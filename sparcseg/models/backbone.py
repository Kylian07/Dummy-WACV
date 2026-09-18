"""Shared encoder/decoder backbone.

Every method in the paper -- single-shot, dense-unrolled, PTEA-lite and
SPARC-Seg -- is built on *this* module with identical weights-initialisation and
identical parameter count in the perceptual path.  That is the only way the
comparison isolates the reasoning mechanism rather than backbone capacity.

The encoder produces the "raw evidence" g_phi(x): a stride-4 feature map with
``sketch_dim`` channels.  The reasoning loop then operates entirely at stride 4
(64x64 for a 256x256 input), which is what makes K unrolled steps affordable on
a single T4.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------
# Offline-friendly pretrained weight loading
# --------------------------------------------------------------------------
def _find_local_resnet_weights(arch: str = "resnet34") -> Optional[Path]:
    """Look for ImageNet weights already on disk.

    Kaggle notebooks frequently run with internet disabled.  Several widely
    mirrored Kaggle datasets ship torchvision checkpoints; we scan for them so
    the notebook does not silently fall back to random initialisation (which
    would cost ~4-6 Dice points and make the numbers non-comparable to the
    literature).
    """
    patterns = [f"{arch}-", f"{arch}_", arch]
    roots = [
        Path(torch.hub.get_dir()) / "checkpoints",
        Path.home() / ".cache/torch/hub/checkpoints",
        Path("/kaggle/input"),
    ]
    for root in roots:
        if not root.is_dir():
            continue
        try:
            for dirpath, _dirnames, filenames in os.walk(root):
                for fn in filenames:
                    low = fn.lower()
                    if low.endswith((".pth", ".pt")) and any(p in low for p in patterns):
                        return Path(dirpath) / fn
        except Exception:
            continue
    return None


def build_resnet(arch: str = "resnet34", pretrained: bool = True) -> Tuple[nn.Module, List[int]]:
    import torchvision

    ctor = getattr(torchvision.models, arch)
    channels = {
        "resnet18": [64, 64, 128, 256, 512],
        "resnet34": [64, 64, 128, 256, 512],
        "resnet50": [64, 256, 512, 1024, 2048],
    }[arch]

    net = None
    if pretrained:
        try:
            weights_enum = getattr(torchvision.models, f"{arch.capitalize()}_Weights", None)
            if weights_enum is None:
                weights_enum = getattr(
                    torchvision.models, f"ResNet{arch.replace('resnet','')}_Weights"
                )
            net = ctor(weights=weights_enum.IMAGENET1K_V1)
            print(f"[backbone] loaded torchvision ImageNet weights for {arch}")
        except Exception as e:
            local = _find_local_resnet_weights(arch)
            if local is not None:
                net = ctor(weights=None)
                try:
                    sd = torch.load(local, map_location="cpu", weights_only=True)
                except TypeError:
                    sd = torch.load(local, map_location="cpu")
                if isinstance(sd, dict) and "state_dict" in sd:
                    sd = sd["state_dict"]
                missing, unexpected = net.load_state_dict(sd, strict=False)
                print(f"[backbone] loaded local ImageNet weights from {local} "
                      f"(missing={len(missing)}, unexpected={len(unexpected)})")
            else:
                warnings.warn(
                    f"\n{'!' * 70}\n"
                    f"Could not obtain ImageNet weights for {arch} ({type(e).__name__}).\n"
                    f"Falling back to RANDOM INITIALISATION. Absolute Dice will be\n"
                    f"several points below published numbers. Turn Kaggle internet ON\n"
                    f"(Settings -> Internet) or attach a torchvision-weights dataset.\n"
                    f"{'!' * 70}",
                    RuntimeWarning,
                )
                net = ctor(weights=None)
    if net is None:
        net = ctor(weights=None)
    return net, channels


class ResNetEncoder(nn.Module):
    """ResNet trunk exposing the five standard pyramid levels."""

    def __init__(self, arch: str = "resnet34", pretrained: bool = True,
                 in_channels: int = 3) -> None:
        super().__init__()
        net, self.out_channels = build_resnet(arch, pretrained)
        if in_channels != 3:
            old = net.conv1
            new = nn.Conv2d(in_channels, old.out_channels, old.kernel_size,
                            old.stride, old.padding, bias=False)
            with torch.no_grad():
                w = old.weight.mean(dim=1, keepdim=True).repeat(1, in_channels, 1, 1)
                new.weight.copy_(w * (3.0 / in_channels))
            net.conv1 = new
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu)
        self.pool = net.maxpool
        self.layer1, self.layer2 = net.layer1, net.layer2
        self.layer3, self.layer4 = net.layer3, net.layer4

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        f0 = self.stem(x)                 # stride  2
        f1 = self.layer1(self.pool(f0))   # stride  4
        f2 = self.layer2(f1)              # stride  8
        f3 = self.layer3(f2)              # stride 16
        f4 = self.layer4(f3)              # stride 32
        return [f0, f1, f2, f3, f4]


def conv_block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class UNetDecoderToStride4(nn.Module):
    """Decode the pyramid down to a stride-4 map with ``out_dim`` channels.

    This map is the working sketch S_0 = g_phi(x): the raw image evidence that
    the reasoning loop then revises.
    """

    def __init__(self, enc_channels: Sequence[int], out_dim: int = 64,
                 width: int = 128) -> None:
        super().__init__()
        c0, c1, c2, c3, c4 = enc_channels
        self.up3 = conv_block(c4 + c3, width)
        self.up2 = conv_block(width + c2, width)
        self.up1 = conv_block(width + c1, width)
        self.project = nn.Sequential(
            nn.Conv2d(width, out_dim, 1, bias=False),
            nn.BatchNorm2d(out_dim),
        )

    @staticmethod
    def _up_cat(x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return torch.cat([x, skip], dim=1)

    def forward(self, feats: Sequence[torch.Tensor]) -> torch.Tensor:
        f0, f1, f2, f3, f4 = feats
        x = self.up3(self._up_cat(f4, f3))
        x = self.up2(self._up_cat(x, f2))
        x = self.up1(self._up_cat(x, f1))
        return self.project(x)


class ReadoutHead(nn.Module):
    """S -> mask logits at full resolution.

    Kept deliberately shallow (two 3x3 convs and two upsamples): if the readout
    were a deep network it could repair an uninformative sketch, and the causal
    claims about S would lose their force.
    """

    def __init__(self, in_dim: int = 64, width: int = 32, n_classes: int = 1,
                 scale: int = 4) -> None:
        super().__init__()
        self.scale = scale
        self.body = nn.Sequential(
            nn.Conv2d(in_dim, width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Conv2d(width, n_classes, 1)

    def forward(self, s: torch.Tensor,
                out_size: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        x = self.body(s)
        logits = self.classifier(x)
        if out_size is None:
            out_size = (logits.shape[-2] * self.scale, logits.shape[-1] * self.scale)
        return F.interpolate(logits, size=out_size, mode="bilinear", align_corners=False)

    def logits_at_sketch_res(self, s: torch.Tensor) -> torch.Tensor:
        """Mask logits at stride-4 resolution, used inside the energy so the
        shape prior does not pay for a 16x upsample at every descent step."""
        return self.classifier(self.body(s))


class Encoder(nn.Module):
    """Convenience wrapper: image -> evidence sketch g_phi(x)."""

    def __init__(self, arch: str = "resnet34", pretrained: bool = True,
                 sketch_dim: int = 64, in_channels: int = 3, width: int = 128) -> None:
        super().__init__()
        self.trunk = ResNetEncoder(arch, pretrained, in_channels)
        self.decoder = UNetDecoderToStride4(self.trunk.out_channels, sketch_dim, width)
        self.sketch_dim = sketch_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.trunk(x))
