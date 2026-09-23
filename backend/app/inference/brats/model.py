"""The trained 3D U-Net, transcribed from notebook Section 12.

This is a faithful reconstruction of the architecture the supplied checkpoint
was trained with — not a substitute. MONAI UNet, nnU-Net, SegResNet, UNETR and
Swin UNETR are all explicitly out of scope (PRD ADR-001): the checkpoint's
`state_dict` keys are produced by the module tree below and nothing else.

The expected parameter count for the trained configuration
(base_channels=32, max_channels=320, num_stages=5) is 18,774,756, matching the
notebook's own Section 12.3 output.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from app.inference.brats.config import BratsModelConfig


class ConvBlock3D(nn.Module):
    """Two 3x3x3 convs, each -> InstanceNorm3d -> LeakyReLU.

    `bias=False` on the convolutions is deliberate and load-bearing for
    checkpoint compatibility: InstanceNorm3d re-centres each channel per
    sample, so a preceding conv bias is a no-op and was never trained.
    """

    def __init__(
        self, in_channels: int, out_channels: int, negative_slope: float = 0.01
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.LeakyReLU(negative_slope, inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.LeakyReLU(negative_slope, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down3D(nn.Module):
    """Strided-conv downsampling by 2x (not max-pool)."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.down = nn.Conv3d(channels, channels, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(x)


class Up3D(nn.Module):
    """Upsample 2x, concatenate the matching encoder skip, then a ConvBlock3D."""

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        mode: str = "transposed_conv",
        negative_slope: float = 0.01,
    ) -> None:
        super().__init__()
        if mode == "transposed_conv":
            self.up: nn.Module = nn.ConvTranspose3d(
                in_channels, in_channels, kernel_size=2, stride=2
            )
        elif mode == "trilinear":
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
                nn.Conv3d(in_channels, in_channels, kernel_size=1),
            )
        else:
            raise ValueError(f"Unknown upsample_mode: {mode}")
        self.conv = ConvBlock3D(in_channels + skip_channels, out_channels, negative_slope)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Guard against odd-dimension off-by-ones. Never fires at the trained
        # patch size (128^3 divides evenly by 2^4), kept for robustness.
        diff = [skip.shape[i] - x.shape[i] for i in (2, 3, 4)]
        if any(diff):
            x = F.pad(
                x,
                [
                    diff[2] // 2,
                    diff[2] - diff[2] // 2,
                    diff[1] // 2,
                    diff[1] - diff[1] // 2,
                    diff[0] // 2,
                    diff[0] - diff[0] // 2,
                ],
            )
        return self.conv(torch.cat([skip, x], dim=1))


class UNet3D(nn.Module):
    """Custom 3D U-Net. Every dimension is read from `BratsModelConfig`."""

    def __init__(self, config: BratsModelConfig) -> None:
        super().__init__()
        channels = [
            min(config.base_channels * (2**i), config.max_channels)
            for i in range(config.num_stages)
        ]
        self.channels = channels

        self.encoder_blocks = nn.ModuleList()
        self.downs = nn.ModuleList()
        in_ch = config.in_channels
        for i, ch in enumerate(channels):
            self.encoder_blocks.append(ConvBlock3D(in_ch, ch, config.negative_slope))
            if i < len(channels) - 1:
                self.downs.append(Down3D(ch))
            in_ch = ch

        self.decoder_blocks = nn.ModuleList(
            [
                Up3D(
                    channels[i],
                    channels[i - 1],
                    channels[i - 1],
                    config.upsample_mode,
                    config.negative_slope,
                )
                for i in range(len(channels) - 1, 0, -1)
            ]
        )
        self.final_conv = nn.Conv3d(channels[0], config.num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips: list[torch.Tensor] = []
        for i, enc_block in enumerate(self.encoder_blocks):
            x = enc_block(x)
            if i < len(self.encoder_blocks) - 1:
                skips.append(x)
                x = self.downs[i](x)
        for dec_block, skip in zip(self.decoder_blocks, reversed(skips)):
            x = dec_block(x, skip)
        return self.final_conv(x)


def build_model(config: BratsModelConfig) -> UNet3D:
    """Construct the network in the exact shape the checkpoint expects."""
    return UNet3D(config)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
