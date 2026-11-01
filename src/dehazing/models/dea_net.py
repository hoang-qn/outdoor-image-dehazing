"""DEA-Net style image dehazing architecture.

This implementation follows the published DEA-Net design: a three-level
encoder/decoder, detail-enhanced convolution blocks, detail-enhanced attention
blocks, and content-guided fusion. It is kept independent of the upstream
repository so it can use this project's training and evaluation pipeline.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class DetailEnhancedConv(nn.Module):
    """Combine vanilla and difference-convolution kernels into one 3x3 conv."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.center = nn.Conv2d(channels, channels, 3, padding=1, bias=True)
        self.horizontal = nn.Parameter(torch.empty(channels, channels, 3))
        self.vertical = nn.Parameter(torch.empty(channels, channels, 3))
        self.angular = nn.Conv2d(channels, channels, 3, padding=1, bias=True)
        self.vanilla = nn.Conv2d(channels, channels, 3, padding=1, bias=True)
        self.horizontal_bias = nn.Parameter(torch.zeros(channels))
        self.vertical_bias = nn.Parameter(torch.zeros(channels))
        self.deploy_conv: nn.Conv2d | None = None
        nn.init.kaiming_uniform_(self.horizontal, a=5**0.5)
        nn.init.kaiming_uniform_(self.vertical, a=5**0.5)
        bias_bound = (channels * 3) ** -0.5
        nn.init.uniform_(self.horizontal_bias, -bias_bound, bias_bound)
        nn.init.uniform_(self.vertical_bias, -bias_bound, bias_bound)

    def _effective_weight(self) -> tuple[torch.Tensor, torch.Tensor]:
        center_weight = self.center.weight.clone()
        center_weight[:, :, 1, 1] -= self.center.weight.sum(dim=(2, 3))

        horizontal_line = self.horizontal.unsqueeze(-1)
        horizontal_weight = F.pad(horizontal_line, (0, 2, 0, 0)) - F.pad(
            horizontal_line, (2, 0, 0, 0)
        )
        vertical_line = self.vertical.unsqueeze(-2)
        vertical_weight = F.pad(vertical_line, (0, 0, 0, 2)) - F.pad(
            vertical_line, (0, 0, 2, 0)
        )

        angular_flat = self.angular.weight.flatten(2)
        angular_permutation = torch.tensor((3, 0, 1, 6, 4, 2, 7, 8, 5), device=angular_flat.device)
        angular_weight = (angular_flat - angular_flat.index_select(2, angular_permutation)).view_as(
            self.angular.weight
        )
        weight = center_weight + horizontal_weight + vertical_weight + angular_weight + self.vanilla.weight
        bias = self.center.bias + self.horizontal_bias + self.vertical_bias + self.angular.bias + self.vanilla.bias
        return weight, bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.deploy_conv is not None:
            return self.deploy_conv(x)
        weight, bias = self._effective_weight()
        return F.conv2d(x, weight, bias, padding=1)

    def switch_to_deploy(self) -> None:
        """Fold difference-convolution branches into a single inference conv."""
        if self.deploy_conv is not None:
            return
        weight, bias = self._effective_weight()
        deploy = nn.Conv2d(weight.shape[1], weight.shape[0], 3, padding=1).to(
            device=weight.device, dtype=weight.dtype
        )
        with torch.no_grad():
            deploy.weight.copy_(weight)
            deploy.bias.copy_(bias)
        self.deploy_conv = deploy
        for name in ("center", "horizontal", "vertical", "angular", "vanilla",
                     "horizontal_bias", "vertical_bias"):
            delattr(self, name)


class DetailBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = DetailEnhancedConv(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=True)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.activation(self.conv1(x)) + x
        return self.conv2(residual) + x


class SpatialAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(2, 1, 7, padding=3, padding_mode="reflect")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = torch.cat((x.mean(dim=1, keepdim=True), x.amax(dim=1, keepdim=True)), dim=1)
        return self.conv(pooled)


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.layers = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class PixelAttention(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(2 * channels, channels, 7, padding=3,
                              padding_mode="reflect", groups=channels)

    def forward(self, x: torch.Tensor, attention: torch.Tensor) -> torch.Tensor:
        paired = torch.stack((x, attention), dim=2).flatten(1, 2)
        return torch.sigmoid(self.conv(paired))


class DetailAttentionBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        self.conv1 = DetailEnhancedConv(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=True)
        self.activation = nn.ReLU(inplace=True)
        self.spatial_attention = SpatialAttention()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.pixel_attention = PixelAttention(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.activation(self.conv1(x)) + x
        features = self.conv2(features)
        attention = self.spatial_attention(features) + self.channel_attention(features)
        return features * self.pixel_attention(features, attention) + x


class ContentGuidedFusion(nn.Module):
    def __init__(self, channels: int, reduction: int) -> None:
        super().__init__()
        self.spatial_attention = SpatialAttention()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.pixel_attention = PixelAttention(channels)
        self.project = nn.Conv2d(channels, channels, 1)

    def forward(self, encoder: torch.Tensor, decoder: torch.Tensor) -> torch.Tensor:
        merged = encoder + decoder
        attention = self.spatial_attention(merged) + self.channel_attention(merged)
        # PixelAttention already applies sigmoid. The original CGAFusion
        # applies sigmoid to that output once more before mixing the branches.
        gate = torch.sigmoid(self.pixel_attention(merged, attention))
        return self.project(merged + gate * encoder + (1.0 - gate) * decoder)


class DeaNet(nn.Module):
    """Three-level DEA-Net for RGB single-image dehazing."""

    def __init__(self, base_dim: int = 32) -> None:
        super().__init__()
        d1, d2, d3 = base_dim, base_dim * 2, base_dim * 4
        self.down1 = nn.Conv2d(3, d1, 3, padding=1)
        self.down2 = nn.Sequential(nn.Conv2d(d1, d2, 3, stride=2, padding=1), nn.ReLU(True))
        self.down3 = nn.Sequential(nn.Conv2d(d2, d3, 3, stride=2, padding=1), nn.ReLU(True))

        self.down_level1 = nn.Sequential(*(DetailBlock(d1) for _ in range(4)))
        self.up_level1 = nn.Sequential(*(DetailBlock(d1) for _ in range(4)))
        self.level2_project = nn.Conv2d(d2, d2, 3, padding=1)
        self.down_level2 = nn.Sequential(*(DetailBlock(d2) for _ in range(4)))
        self.up_level2 = nn.Sequential(*(DetailBlock(d2) for _ in range(4)))
        self.level3_project = nn.Conv2d(d3, d3, 3, padding=1)
        self.level3 = nn.Sequential(*(DetailAttentionBlock(d3) for _ in range(8)))

        self.fuse_level3 = ContentGuidedFusion(d3, reduction=8)
        self.fuse_level2 = ContentGuidedFusion(d2, reduction=4)
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(d3, d2, 3, stride=2, padding=1, output_padding=1), nn.ReLU(True)
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(d2, d1, 3, stride=2, padding=1, output_padding=1), nn.ReLU(True)
        )
        self.output = nn.Conv2d(d1, 3, 3, padding=1)

    @staticmethod
    def _pad_to_multiple(x: torch.Tensor, multiple: int = 4) -> torch.Tensor:
        height, width = x.shape[-2:]
        pad_h, pad_w = (-height) % multiple, (-width) % multiple
        if not (pad_h or pad_w):
            return x
        mode = "reflect" if height > pad_h and width > pad_w else "replicate"
        return F.pad(x, (0, pad_w, 0, pad_h), mode=mode)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_h, original_w = x.shape[-2:]
        x = self._pad_to_multiple(x)

        level1 = self.down_level1(self.down1(x))

        # DEA-Net's content-guided fusion uses the encoder features directly
        # after downsampling. Keep these tensors separate from the features
        # that pass through the per-level enhancement blocks.
        level2_skip = self.down2(level1)
        level2 = self.down_level2(self.level2_project(level2_skip))
        level3_skip = self.down3(level2)
        level3_input = self.level3_project(level3_skip)
        level3 = self.level3(level3_input)
        level3 = self.fuse_level3(level3_skip, level3)

        decoder2 = self.up_level2(self.up1(level3))
        if decoder2.shape[-2:] != level2_skip.shape[-2:]:
            decoder2 = F.interpolate(
                decoder2, size=level2_skip.shape[-2:], mode="bilinear", align_corners=False
            )
        decoder2 = self.fuse_level2(level2_skip, decoder2)
        decoder1 = self.up_level1(self.up2(decoder2))
        output = self.output(decoder1)
        return output[..., :original_h, :original_w]

    def reparameterize(self) -> None:
        """Convert all DEConv blocks into equivalent single convolutions."""
        for module in tuple(self.modules()):
            if isinstance(module, DetailEnhancedConv):
                module.switch_to_deploy()


def dea_net(base_dim: int = 32) -> DeaNet:
    return DeaNet(base_dim=base_dim)
