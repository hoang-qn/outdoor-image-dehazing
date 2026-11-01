"""Lightweight dehazing architectures from recent papers.

SLRNet is based on the authors' public implementation:
https://github.com/Unk1ndledAC/SLRNet

WLD-Net is based on the authors' public implementation:
https://github.com/AliMurtaza29/WLD-Net
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class AdaptiveFeatureUnit(nn.Module):
    """Asymmetric feature projection and global context gate used by SLRNet."""

    def __init__(self, channels: int, reduction: int = 2, gate_groups: int = 4) -> None:
        super().__init__()
        split_channels = channels // reduction
        hidden_gate = max(4, channels // gate_groups)
        self.proj = nn.Conv2d(channels, split_channels, 1, bias=False)
        self.norm_proj = nn.BatchNorm2d(split_channels)
        self.refine = nn.Conv2d(
            split_channels,
            channels - split_channels,
            3,
            padding=1,
            groups=split_channels,
            bias=False,
        )
        self.norm_refine = nn.BatchNorm2d(channels - split_channels)
        self.gate_pool = nn.AdaptiveAvgPool2d(1)
        self.gate_reduce = nn.Conv2d(split_channels, hidden_gate, 1, bias=False)
        self.gate_act = nn.GELU()
        self.gate_expand = nn.Conv2d(hidden_gate, channels, 1)
        self.gate_sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projected = self.norm_proj(self.proj(x))
        refined = self.norm_refine(self.refine(projected))
        features = torch.cat((projected, refined), dim=1)
        gate = self.gate_pool(projected)
        gate = self.gate_sigmoid(self.gate_expand(self.gate_act(self.gate_reduce(gate))))
        return features * gate


class _SLRResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.activation = nn.GELU()
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv2(self.activation(self.conv1(x)))


class SLRNet(nn.Module):
    """Super Lightweight Residual Network for RGB image dehazing."""

    def __init__(self, base_channels: int = 32, num_blocks: int = 5) -> None:
        super().__init__()
        self.in_proj = nn.Conv2d(3, base_channels, 3, padding=1)
        self.enhancer = AdaptiveFeatureUnit(base_channels)
        self.backbone = nn.ModuleList(
            [_SLRResidualBlock(base_channels) for _ in range(num_blocks)]
        )
        self.refiner = AdaptiveFeatureUnit(base_channels)
        self.out_proj = nn.Conv2d(base_channels, 3, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.in_proj(x)
        early_features = self.enhancer(features)
        features = early_features
        for block in self.backbone:
            features = block(features)
        features = self.refiner(features) + early_features
        return self.out_proj(features).clamp(0.0, 1.0)


def _haar_dwt(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """One-level fixed Haar transform, returning low and concatenated high bands."""
    x_top = x[:, :, 0::2, :] * 0.5
    x_bottom = x[:, :, 1::2, :] * 0.5
    x1, x2 = x_top[:, :, :, 0::2], x_bottom[:, :, :, 0::2]
    x3, x4 = x_top[:, :, :, 1::2], x_bottom[:, :, :, 1::2]
    low = x1 + x2 + x3 + x4
    high = torch.cat((-x1 - x2 + x3 + x4, -x1 + x2 - x3 + x4, x1 - x2 - x3 + x4), dim=1)
    return low, high


class _WaveletSplit(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, sampling: int = 2) -> None:
        super().__init__()
        self.low_projection = nn.Conv2d(in_channels, out_channels, 1)
        self.high_projection = nn.Conv2d(in_channels * 3, out_channels, 1)
        self.spatial_projection = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, out_channels, 3, stride=sampling, padding=0),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        low, high = _haar_dwt(x)
        low = self.low_projection(low) + self.spatial_projection(x)
        return low, self.high_projection(high)


class _FactorizedConv(nn.Module):
    """Reflection-padded 1x3 then 3x1 convolution used in WLD-Net."""

    def __init__(self, in_channels: int, out_channels: int, *, activation: bool) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.ReflectionPad2d((0, 1, 0, 1)),
            nn.Conv2d(in_channels, out_channels, (1, 3), bias=False),
            nn.ReflectionPad2d((1, 0, 1, 0)),
            nn.Conv2d(out_channels, out_channels, (3, 1), bias=False),
        ]
        if activation:
            layers.append(nn.ReLU(inplace=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class _ChannelAttention(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size, padding=(kernel_size - 1) // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.pool(x).squeeze(-1).transpose(1, 2)
        weights = torch.sigmoid(self.conv(weights)).transpose(1, 2).unsqueeze(-1)
        return x * weights


class _PixelAttention(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 1, bias=False)
        self.conv2 = nn.Conv2d(channels, channels, 1, bias=False)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(self.conv2(self.activation(self.conv1(x))))


class _SpatialAttention(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(
            channels, channels, kernel_size, padding=kernel_size // 2, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = torch.sigmoid(self.conv(self.pool(x).flatten(2))).unsqueeze(-1)
        return x * weights


class _DualAttention(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channel = _ChannelAttention(channels)
        self.pixel = _PixelAttention(channels)
        self.spatial = _SpatialAttention(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attended = self.channel(x)
        return torch.cat((self.pixel(attended) + x, self.spatial(attended) + x), dim=1)


class _WaveletFeatureBlock(nn.Module):
    def __init__(self, channels: int = 32) -> None:
        super().__init__()
        self.factorized = _FactorizedConv(channels, channels, activation=True)
        self.attention = _DualAttention(channels)
        self.project = nn.Conv2d(channels * 2, channels, 1, bias=False)
        self.norm = nn.InstanceNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.project(self.attention(self.factorized(x)))
        return features + self.norm(x)


class _LowFrequencyGroup(nn.Module):
    def __init__(self, channels: int, num_blocks: int = 3) -> None:
        super().__init__()
        self.blocks = nn.Sequential(*[_WaveletFeatureBlock(channels) for _ in range(num_blocks)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class _HighFrequencyProcess(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.pad = nn.ReflectionPad2d(1)
        self.conv = nn.Conv2d(channels, channels, 3, padding=0)
        self.norm = nn.InstanceNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.norm(self.conv(self.pad(x)))) + x


class _HighFrequencyDown(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.pad = nn.ReflectionPad2d(1)
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=0)
        self.factorized = _FactorizedConv(channels, channels, activation=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.factorized(self.conv(self.pad(x)))


class _WaveletUp(nn.Module):
    def __init__(self, channels: int, *, fuse_three: bool = False) -> None:
        super().__init__()
        self.fuse_three = fuse_three
        self.pre = _FactorizedConv(channels, channels, activation=False)
        self.pixel = _PixelAttention(channels)
        self.up = nn.ConvTranspose2d(
            channels, channels, 3, stride=2, padding=1, output_padding=1, bias=False
        )
        self.post = _FactorizedConv(channels, channels, activation=True)

    def forward(self, *features: torch.Tensor) -> torch.Tensor:
        merged = features[0]
        for feature in features[1:]:
            merged = merged + feature
        return self.post(self.up(self.pixel(self.pre(merged))))


class WLDNet(nn.Module):
    """WaveLiteDehaze-Net with a [0, 1] RGB input/output interface."""

    def __init__(
        self,
        channels: int = 32,
        blocks_per_level: tuple[int, int, int, int] = (3, 3, 3, 3),
    ) -> None:
        super().__init__()
        self.splits = nn.ModuleList(
            [_WaveletSplit(3 if level == 0 else channels, channels) for level in range(4)]
        )
        self.low_groups = nn.ModuleList(
            [_LowFrequencyGroup(channels, count) for count in blocks_per_level]
        )
        self.high_process = nn.ModuleList([_HighFrequencyProcess(channels) for _ in range(4)])
        self.high_down = nn.ModuleList([_HighFrequencyDown(channels) for _ in range(3)])
        self.high_up = _WaveletUp(channels)
        self.fusions = nn.ModuleList([_WaveletUp(channels, fuse_three=True) for _ in range(3)])
        self.out_channel_attention = _ChannelAttention(channels)
        self.out_pixel_attention = _PixelAttention(channels)
        self.out_conv1 = nn.Conv2d(channels, channels // 2, 3, padding=1)
        self.out_conv2 = nn.Conv2d(channels // 2, 3, 3, padding=1)

    def _forward_normalized(self, x: torch.Tensor) -> torch.Tensor:
        low_features: list[torch.Tensor] = []
        high_features: list[torch.Tensor] = []
        prior_high: torch.Tensor | None = None
        low = x
        for level, split in enumerate(self.splits):
            low, high = split(low)
            low = self.low_groups[level](low)
            if prior_high is not None:
                high = high + prior_high
            high = self.high_process[level](high)
            low_features.append(low)
            high_features.append(high)
            if level < 3:
                prior_high = self.high_down[level](high)

        decoded = self.high_up(high_features[3], low_features[3])
        for level, fusion in zip((2, 1, 0), self.fusions):
            decoded = fusion(decoded, high_features[level], low_features[level])
        decoded = self.out_channel_attention(decoded)
        decoded = self.out_pixel_attention(decoded)
        decoded = self.out_conv2(F.relu(self.out_conv1(decoded)))
        return torch.tanh(decoded + x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_height, original_width = x.shape[-2:]
        normalized = x.mul(2.0).sub(1.0)
        pad_h, pad_w = (-original_height) % 16, (-original_width) % 16
        if pad_h or pad_w:
            mode = "reflect" if original_height > pad_h and original_width > pad_w else "replicate"
            normalized = F.pad(normalized, (0, pad_w, 0, pad_h), mode=mode)
        output = self._forward_normalized(normalized)
        output = output[..., :original_height, :original_width]
        return output.add(1.0).mul(0.5)


def slrnet(base_channels: int = 32, num_blocks: int = 5) -> SLRNet:
    return SLRNet(base_channels=base_channels, num_blocks=num_blocks)


def wldnet(channels: int = 32) -> WLDNet:
    return WLDNet(channels=channels)
