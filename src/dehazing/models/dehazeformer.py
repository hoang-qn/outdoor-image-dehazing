"""DehazeFormer-T implementation, following the authors' public architecture."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def _trunc_normal_(tensor: torch.Tensor, std: float = 0.02) -> torch.Tensor:
    return nn.init.trunc_normal_(tensor, std=std)


class RevisedLayerNorm(nn.Module):
    """Feature-map normalization with learned rescaling and rebiasing."""

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.meta1 = nn.Conv2d(1, dim, 1)
        self.meta2 = nn.Conv2d(1, dim, 1)
        _trunc_normal_(self.meta1.weight)
        nn.init.constant_(self.meta1.bias, 1)
        _trunc_normal_(self.meta2.weight)
        nn.init.constant_(self.meta2.bias, 0)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        std = torch.sqrt((x - mean).square().mean(dim=(1, 2, 3), keepdim=True) + self.eps)
        return (x - mean) / std * self.weight + self.bias, self.meta1(std), self.meta2(mean)


class Mlp(nn.Module):
    def __init__(self, network_depth: int, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(nn.Conv2d(dim, hidden_dim, 1), nn.ReLU(inplace=True), nn.Conv2d(hidden_dim, dim, 1))
        gain = (8 * network_depth) ** (-1 / 4)
        for layer in self.layers:
            if isinstance(layer, nn.Conv2d):
                fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(layer.weight)
                _trunc_normal_(layer.weight, gain * math.sqrt(2.0 / (fan_in + fan_out)))
                nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


def _window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    batch, height, width, channels = x.shape
    x = x.view(batch, height // window_size, window_size, width // window_size, window_size, channels)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size * window_size, channels)


def _window_reverse(windows: torch.Tensor, window_size: int, height: int, width: int) -> torch.Tensor:
    batch = int(windows.shape[0] / (height * width / window_size / window_size))
    x = windows.view(batch, height // window_size, width // window_size, window_size, window_size, -1)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(batch, height, width, -1)


class WindowAttention(nn.Module):
    def __init__(self, dim: int, window_size: int, num_heads: int) -> None:
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        coords = torch.stack(torch.meshgrid(torch.arange(window_size), torch.arange(window_size), indexing="ij"))
        coords = coords.flatten(1)
        relative = (coords[:, :, None] - coords[:, None, :]).permute(1, 2, 0).contiguous()
        relative = relative.sign() * torch.log1p(relative.abs())
        self.register_buffer("relative_positions", relative, persistent=False)
        self.position_mlp = nn.Sequential(nn.Linear(2, 256), nn.ReLU(inplace=True), nn.Linear(256, num_heads))

    def forward(self, qkv: torch.Tensor) -> torch.Tensor:
        batch_windows, tokens, _ = qkv.shape
        qkv = qkv.reshape(batch_windows, tokens, 3, self.num_heads, self.dim // self.num_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        attention = (q * self.scale) @ k.transpose(-2, -1)
        bias = self.position_mlp(self.relative_positions).permute(2, 0, 1).contiguous()
        attention = (attention + bias.unsqueeze(0)).softmax(dim=-1)
        return (attention @ v).transpose(1, 2).reshape(batch_windows, tokens, self.dim)


class Dcp3DPositionEmbedding(nn.Module):
    """Fixed sine/cosine embedding of x, y, and dark-channel haze density."""

    def __init__(self, dim: int, density_scale: float) -> None:
        super().__init__()
        if dim % 6:
            raise ValueError("DCP 3D positional embedding requires a channel count divisible by 6.")
        self.frequencies_per_axis = dim // 6
        self.density_scale = density_scale

    def forward(self, density: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = features.shape
        density = F.interpolate(density.float(), size=(height, width), mode="area")
        device = features.device
        y = torch.arange(height, device=device, dtype=torch.float32).view(1, 1, height, 1)
        x = torch.arange(width, device=device, dtype=torch.float32).view(1, 1, 1, width)
        y = y.expand(batch, 1, height, width)
        x = x.expand(batch, 1, height, width)
        haze = density * self.density_scale
        frequencies = torch.exp(
            -math.log(10000.0)
            * torch.arange(self.frequencies_per_axis, device=device, dtype=torch.float32)
            / self.frequencies_per_axis
        ).view(1, -1, 1, 1)

        def encode(coordinate: torch.Tensor) -> torch.Tensor:
            angles = coordinate * frequencies
            return torch.stack((angles.sin(), angles.cos()), dim=2).flatten(1, 2)

        return torch.cat((encode(x), encode(y), encode(haze)), dim=1).to(dtype=features.dtype)


class AttentionBlock(nn.Module):
    def __init__(
        self,
        network_depth: int,
        dim: int,
        num_heads: int,
        mlp_ratio: float,
        window_size: int,
        shift_size: int,
        use_attention: bool,
        dcp_position: bool = False,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.shift_size = shift_size
        self.use_attention = use_attention
        self.norm = RevisedLayerNorm(dim) if use_attention else nn.Identity()
        self.value = nn.Conv2d(dim, dim, 1)
        self.qk = nn.Conv2d(dim, dim * 2, 1) if use_attention else None
        self.depthwise = nn.Conv2d(dim, dim, 5, padding=2, groups=dim, padding_mode="reflect")
        self.proj = nn.Conv2d(dim, dim, 1)
        self.window_attention = WindowAttention(dim, window_size, num_heads) if use_attention else None
        self.dcp_position_gain = nn.Parameter(torch.tensor(0.1)) if use_attention and dcp_position else None
        self.mlp = Mlp(network_depth, dim, int(dim * mlp_ratio))
        gain = (8 * network_depth) ** (-1 / 4)
        for layer in (self.value, self.qk, self.depthwise, self.proj):
            if layer is not None:
                fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(layer.weight)
                std = math.sqrt(2.0 / (fan_in + fan_out)) if layer is self.qk else gain * math.sqrt(2.0 / (fan_in + fan_out))
                _trunc_normal_(layer.weight, std)
                nn.init.zeros_(layer.bias)

    def _pad_windows(self, x: torch.Tensor) -> torch.Tensor:
        _, _, height, width = x.shape
        pad_h = (self.window_size - height % self.window_size) % self.window_size
        pad_w = (self.window_size - width % self.window_size) % self.window_size
        if self.shift_size:
            pad = (
                self.shift_size,
                (self.window_size - self.shift_size + pad_w) % self.window_size,
                self.shift_size,
                (self.window_size - self.shift_size + pad_h) % self.window_size,
            )
        else:
            pad = (0, pad_w, 0, pad_h)
        return F.pad(x, pad, mode="reflect") if any(pad) else x

    def forward(self, x: torch.Tensor, position_embedding: torch.Tensor | None = None) -> torch.Tensor:
        residual = x
        if self.use_attention:
            normalized, rescale, rebias = self.norm(x)
            if self.dcp_position_gain is not None:
                if position_embedding is None:
                    raise ValueError("DCP positional embedding is required for this attention block.")
                normalized = normalized + torch.tanh(self.dcp_position_gain) * position_embedding
            value = self.value(normalized)
            qkv = torch.cat((self.qk(normalized), value), dim=1)
            padded = self._pad_windows(qkv)
            padded_h, padded_w = padded.shape[-2:]
            windows = _window_partition(padded.permute(0, 2, 3, 1), self.window_size)
            attended = self.window_attention(windows)
            attended = _window_reverse(attended, self.window_size, padded_h, padded_w)
            _, _, height, width = qkv.shape
            attended = attended[:, self.shift_size : self.shift_size + height,
                                self.shift_size : self.shift_size + width].permute(0, 3, 1, 2)
            x = residual + self.proj(self.depthwise(value) + attended * rescale + rebias)
        else:
            x = residual + self.proj(self.depthwise(self.value(x)))
        return x + self.mlp(x)


class BasicLayer(nn.Module):
    def __init__(
        self, dim: int, depth: int, num_heads: int, mlp_ratio: float,
        attn_ratio: float, window_size: int, dcp_position: bool = False,
    ) -> None:
        super().__init__()
        attn_depth = int(attn_ratio * depth)
        self.blocks = nn.ModuleList(
            AttentionBlock(
                network_depth=16,
                dim=dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                window_size=window_size,
                shift_size=0 if i % 2 == 0 else window_size // 2,
                use_attention=i >= depth - attn_depth,
                dcp_position=dcp_position,
            )
            for i in range(depth)
        )

    def forward(self, x: torch.Tensor, position_embedding: torch.Tensor | None = None) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, position_embedding=position_embedding)
        return x


class PatchEmbed(nn.Module):
    def __init__(self, in_chans: int, embed_dim: int, stride: int = 1, kernel_size: int = 3) -> None:
        super().__init__()
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size, stride=stride,
            padding=(kernel_size - stride + 1) // 2, padding_mode="reflect",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class PatchUnEmbed(nn.Module):
    def __init__(self, embed_dim: int, out_chans: int, scale: int = 1, kernel_size: int = 3) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(embed_dim, out_chans * scale**2, kernel_size,
                      padding=kernel_size // 2, padding_mode="reflect"),
            nn.PixelShuffle(scale),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class SKFusion(nn.Module):
    def __init__(self, dim: int, height: int = 2, reduction: int = 8) -> None:
        super().__init__()
        self.height = height
        hidden = max(dim // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(nn.Conv2d(dim, hidden, 1, bias=False), nn.ReLU(),
                                 nn.Conv2d(hidden, dim * height, 1, bias=False))

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        batch, channels, height, width = features[0].shape
        stacked = torch.stack(features, dim=1)
        weights = self.mlp(self.pool(stacked.sum(dim=1)))
        weights = weights.view(batch, self.height, channels, 1, 1).softmax(dim=1)
        return (stacked * weights).sum(dim=1)


class SpatialChannelAdaptiveFusion(nn.Module):
    """Fuse skip and decoder features with channel and spatial attention maps."""

    def __init__(self, dim: int, height: int = 2, reduction: int = 8) -> None:
        super().__init__()
        self.height = height
        hidden = max(dim // reduction, 4)
        self.channel_logits = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, hidden, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden, dim * height, 1, bias=False),
        )
        self.spatial_logits = nn.Sequential(
            nn.Conv2d(dim, hidden, 3, padding=1, padding_mode="reflect", bias=False),
            nn.GELU(),
            nn.Conv2d(hidden, height, 1),
        )

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        batch, channels, height, width = features[0].shape
        stacked = torch.stack(features, dim=1)
        summary = stacked.sum(dim=1)
        channel_logits = self.channel_logits(summary).reshape(batch, self.height, channels, 1, 1)
        spatial_logits = self.spatial_logits(summary).reshape(batch, self.height, 1, height, width)
        weights = (channel_logits + spatial_logits).softmax(dim=1)
        return (stacked * weights).sum(dim=1)


class FixedHaarDetailBranch(nn.Module):
    """Extract fixed Haar high-frequency features and project them to decoder width."""

    def __init__(self, out_channels: int = 24) -> None:
        super().__init__()
        kernels = torch.tensor(
            [
                [[1, 1], [1, 1]],
                [[-1, -1], [1, 1]],
                [[-1, 1], [-1, 1]],
                [[1, -1], [-1, 1]],
            ],
            dtype=torch.float32,
        ) / 2.0
        self.register_buffer("haar", kernels[:, None].repeat(3, 1, 1, 1), persistent=False)
        self.project = nn.Sequential(
            nn.Conv2d(9, out_channels, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(
                out_channels, out_channels, 3, padding=1,
                groups=out_channels, padding_mode="reflect",
            ),
        )

    def forward(self, image: torch.Tensor, output_size: tuple[int, int]) -> torch.Tensor:
        coefficients = F.conv2d(image, self.haar.to(dtype=image.dtype), stride=2, groups=3)
        batch, _, height, width = coefficients.shape
        coefficients = coefficients.reshape(batch, 3, 4, height, width)
        high_frequency = coefficients[:, :, 1:].reshape(batch, 9, height, width)
        detail = self.project(high_frequency)
        return F.interpolate(detail, size=output_size, mode="bilinear", align_corners=False)


class WaveletResidualBlock(nn.Module):
    """A lightweight depthwise-separable residual transform for wavelet bands."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            channels, channels, 3, padding=1, groups=channels, padding_mode="reflect",
        )
        self.activation = nn.GELU()
        self.pointwise = nn.Conv2d(channels, channels, 1)
        # Start as a no-op so the added branch does not disturb the baseline.
        nn.init.zeros_(self.pointwise.weight)
        nn.init.zeros_(self.pointwise.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.activation(self.depthwise(x)))


class MultiLevelWaveletRefinement(nn.Module):
    """Refine two levels of fixed Haar coefficients on decoder features."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        filters = torch.tensor(
            [
                [[1, 1], [1, 1]],
                [[-1, -1], [1, 1]],
                [[-1, 1], [-1, 1]],
                [[1, -1], [-1, 1]],
            ],
            dtype=torch.float32,
        ) / 2.0
        self.register_buffer("haar", filters[:, None], persistent=False)
        self.low_frequency_refine = WaveletResidualBlock(channels)
        self.level2_detail_refine = WaveletResidualBlock(channels * 3)
        self.level1_detail_refine = WaveletResidualBlock(channels * 3)
        self.output_gate = nn.Parameter(torch.tensor(0.1))

    def _dwt(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int]]:
        batch, channels, height, width = x.shape
        original_size = (height, width)
        pad_h, pad_w = height % 2, width % 2
        if pad_h or pad_w:
            mode = "reflect" if height > 1 and width > 1 else "replicate"
            x = F.pad(x, (0, pad_w, 0, pad_h), mode=mode)
        filters = self.haar.to(device=x.device, dtype=x.dtype).repeat(channels, 1, 1, 1)
        coefficients = F.conv2d(x, filters, stride=2, groups=channels)
        _, _, low_h, low_w = coefficients.shape
        coefficients = coefficients.reshape(batch, channels, 4, low_h, low_w)
        low_frequency = coefficients[:, :, 0]
        detail = coefficients[:, :, 1:].reshape(batch, channels * 3, low_h, low_w)
        return low_frequency, detail, original_size

    def _idwt(
        self,
        low_frequency: torch.Tensor,
        detail: torch.Tensor,
        output_size: tuple[int, int],
    ) -> torch.Tensor:
        batch, channels, low_h, low_w = low_frequency.shape
        detail = detail.reshape(batch, channels, 3, low_h, low_w)
        coefficients = torch.cat((low_frequency.unsqueeze(2), detail), dim=2)
        coefficients = coefficients.reshape(batch, channels * 4, low_h, low_w)
        filters = self.haar.to(device=coefficients.device, dtype=coefficients.dtype).repeat(
            channels, 1, 1, 1,
        )
        reconstructed = F.conv_transpose2d(coefficients, filters, stride=2, groups=channels)
        return reconstructed[..., : output_size[0], : output_size[1]]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        low1, detail1, size1 = self._dwt(x)
        low2, detail2, size2 = self._dwt(low1)

        low2 = low2 + self.low_frequency_refine(low2)
        detail2 = detail2 + self.level2_detail_refine(detail2)
        low1 = self._idwt(low2, detail2, size2)
        detail1 = detail1 + self.level1_detail_refine(detail1)
        reconstructed = self._idwt(low1, detail1, size1)
        return x + torch.tanh(self.output_gate) * (reconstructed - x)


class HazeMapFeatureConditioner(nn.Module):
    """Apply a small feature modulation weighted by the estimated haze map."""

    def __init__(self, channels: int, strength: float = 0.1) -> None:
        super().__init__()
        self.strength = strength
        self.affine = nn.Conv2d(1, channels * 2, 1)
        nn.init.zeros_(self.affine.weight)
        nn.init.zeros_(self.affine.bias)

    def forward(self, features: torch.Tensor, haze_map: torch.Tensor) -> torch.Tensor:
        scale, bias = self.affine(haze_map).chunk(2, dim=1)
        confidence = haze_map.clamp(0, 1)
        return features * (1 + self.strength * confidence * torch.tanh(scale)) + (
            self.strength * confidence * torch.tanh(bias)
        )


class DecoderConditionedSkipModulation(nn.Module):
    """Predict per-pixel affine modulation of a skip from decoder features."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.affine = nn.Conv2d(channels, channels * 2, 1)
        nn.init.zeros_(self.affine.weight)
        nn.init.zeros_(self.affine.bias)

    def forward(self, skip: torch.Tensor, decoder: torch.Tensor) -> torch.Tensor:
        scale, bias = self.affine(decoder).chunk(2, dim=1)
        return skip * (1 + torch.tanh(scale)) + torch.tanh(bias)


class DehazeFormer(nn.Module):
    """DehazeFormer-T and the project's M1-M9 ablation variants."""

    def __init__(
        self,
        window_size: int = 8,
        use_depth: bool = False,
        fusion: str = "sk",
        haze_guidance: bool = False,
        frequency_branch: bool = False,
        wavelet_refinement: bool = False,
        haze_skip_guidance: bool = False,
        physical_consistency: bool = False,
        decoder_skip_modulation: bool = False,
        dcp_position_embedding: bool = False,
    ) -> None:
        super().__init__()
        if fusion not in {"sk", "scaf"}:
            raise ValueError(f"Unsupported fusion type: {fusion!r}.")
        if haze_skip_guidance and not haze_guidance:
            raise ValueError("haze_skip_guidance requires haze_guidance to predict the haze map.")
        if use_depth and (
            haze_guidance or frequency_branch or wavelet_refinement or physical_consistency
            or decoder_skip_modulation or dcp_position_embedding
        ):
            raise ValueError("Depth conditioning and M1-M9 ablations are separate experiments.")
        self.patch_size = 4
        self.window_size = window_size
        self.use_depth = use_depth
        self.haze_guidance = haze_guidance
        self.frequency_branch = frequency_branch
        self.wavelet_refinement = wavelet_refinement
        self.haze_skip_guidance = haze_skip_guidance
        self.physical_consistency = physical_consistency
        self.decoder_skip_modulation = decoder_skip_modulation
        self.dcp_position_embedding = dcp_position_embedding
        dims = (24, 48, 96, 48, 24)
        self.patch_embed = PatchEmbed(4 if use_depth else 3, dims[0], stride=1, kernel_size=3)
        self.layer1 = BasicLayer(dims[0], 4, 2, 2, 0, window_size)
        self.patch_merge1 = PatchEmbed(dims[0], dims[1], stride=2, kernel_size=2)
        self.skip1 = nn.Conv2d(dims[0], dims[0], 1)
        self.layer2 = BasicLayer(dims[1], 4, 4, 4, 0.5, window_size, dcp_position=dcp_position_embedding)
        self.patch_merge2 = PatchEmbed(dims[1], dims[2], stride=2, kernel_size=2)
        self.skip2 = nn.Conv2d(dims[1], dims[1], 1)
        self.layer3 = BasicLayer(dims[2], 4, 6, 4, 1, window_size, dcp_position=dcp_position_embedding)
        if dcp_position_embedding:
            self.dcp_position2 = Dcp3DPositionEmbedding(dims[1], window_size)
            self.dcp_position3 = Dcp3DPositionEmbedding(dims[2], window_size)
        if haze_guidance:
            self.haze_head = nn.Sequential(
                nn.Conv2d(dims[2], 24, 3, padding=1, padding_mode="reflect"),
                nn.GELU(),
                nn.Conv2d(24, 1, 1),
            )
            self.haze_scale = nn.Parameter(torch.tensor(0.1))
        if haze_skip_guidance:
            self.haze_decoder_condition1 = HazeMapFeatureConditioner(dims[3])
            self.haze_skip_condition1 = HazeMapFeatureConditioner(dims[3])
            self.haze_decoder_condition2 = HazeMapFeatureConditioner(dims[4])
            self.haze_skip_condition2 = HazeMapFeatureConditioner(dims[4])
        if physical_consistency:
            self.transmission_head = nn.Sequential(
                nn.Conv2d(dims[2], 24, 3, padding=1, padding_mode="reflect"),
                nn.GELU(),
                nn.Conv2d(24, 1, 1),
            )
            nn.init.constant_(self.transmission_head[-1].bias, 2.0)
            self.airlight_head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(dims[2], 24, 1),
                nn.GELU(),
                nn.Conv2d(24, 3, 1),
            )
        self.patch_split1 = PatchUnEmbed(dims[2], dims[3], scale=2, kernel_size=1)
        fusion_module = SKFusion if fusion == "sk" else SpatialChannelAdaptiveFusion
        self.fusion1 = fusion_module(dims[3])
        if decoder_skip_modulation:
            self.skip_modulation1 = DecoderConditionedSkipModulation(dims[3])
        self.layer4 = BasicLayer(dims[3], 2, 1, 2, 0, window_size)
        self.patch_split2 = PatchUnEmbed(dims[3], dims[4], scale=2, kernel_size=1)
        self.fusion2 = fusion_module(dims[4])
        if decoder_skip_modulation:
            self.skip_modulation2 = DecoderConditionedSkipModulation(dims[4])
        if frequency_branch:
            self.detail_branch = FixedHaarDetailBranch(dims[4])
            self.detail_gate = nn.Parameter(torch.tensor(0.1))
        if wavelet_refinement:
            self.wavelet_refiner = MultiLevelWaveletRefinement(dims[4])
        self.layer5 = BasicLayer(dims[4], 2, 1, 2, 0, window_size)
        self.patch_unembed = PatchUnEmbed(dims[4], 4, scale=1, kernel_size=3)

    def _pad_image(self, x: torch.Tensor) -> torch.Tensor:
        height, width = x.shape[-2:]
        pad_h = (-height) % self.patch_size
        pad_w = (-width) % self.patch_size
        if (pad_h or pad_w) and (height <= pad_h or width <= pad_w):
            raise ValueError("Input image is too small for reflect padding.")
        return F.pad(x, (0, pad_w, 0, pad_h), mode="reflect") if pad_h or pad_w else x

    @staticmethod
    def _dcp_density(hazy: torch.Tensor) -> torch.Tensor:
        # With unit airlight, the dark channel approximates the haze density (1 - T).
        dark_rgb = hazy.float().amin(dim=1, keepdim=True)
        return (-F.max_pool2d(-dark_rgb, kernel_size=15, stride=1, padding=7)).clamp(0, 1)

    def forward(
        self,
        x: torch.Tensor,
        depth: torch.Tensor | None = None,
        return_aux: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        original_h, original_w = x.shape[-2:]
        if self.use_depth:
            if depth is None:
                raise ValueError("This DehazeFormer checkpoint requires a depth map input.")
            if depth.ndim != 4 or depth.shape[1] != 1 or depth.shape[0] != x.shape[0]:
                raise ValueError("depth must have shape [batch, 1, height, width].")
            if depth.shape[-2:] != x.shape[-2:]:
                raise ValueError("RGB input and depth map must have identical spatial dimensions.")
            features = torch.cat((x, depth.to(dtype=x.dtype)), dim=1)
        else:
            if depth is not None:
                raise ValueError("A depth map was supplied to a model configured without depth input.")
            features = x

        features = self._pad_image(features)
        # Keep the reconstruction skip connection RGB-only. Depth conditions
        # feature extraction but is never treated as an output image channel.
        hazy = features[:, :3]
        dcp_density = self._dcp_density(hazy) if self.dcp_position_embedding else None
        skip1 = self.layer1(self.patch_embed(features))
        stage2 = self.patch_merge1(skip1)
        position2 = self.dcp_position2(dcp_density, stage2) if dcp_density is not None else None
        skip2 = self.layer2(stage2, position_embedding=position2)
        stage3 = self.patch_merge2(skip2)
        position3 = self.dcp_position3(dcp_density, stage3) if dcp_density is not None else None
        x = self.layer3(stage3, position_embedding=position3)
        auxiliary: dict[str, torch.Tensor] = {}
        haze_map: torch.Tensor | None = None
        if self.haze_guidance:
            haze_map = torch.sigmoid(self.haze_head(x))
            x = x * (1.0 + torch.tanh(self.haze_scale) * haze_map)
            auxiliary["haze_map"] = haze_map
        if self.physical_consistency and return_aux:
            transmission = 0.05 + 0.95 * torch.sigmoid(self.transmission_head(x))
            transmission = F.interpolate(
                transmission, size=hazy.shape[-2:], mode="bilinear", align_corners=False,
            )
            auxiliary["transmission"] = transmission[
                ..., :original_h, :original_w
            ]
            auxiliary["airlight"] = torch.sigmoid(self.airlight_head(x))
        x = self.patch_split1(x)
        skip2_features = self.skip2(skip2)
        if self.haze_skip_guidance:
            if haze_map is None:
                raise RuntimeError("Haze-map skip guidance was enabled without a haze map.")
            haze_half = F.interpolate(haze_map, size=x.shape[-2:], mode="bilinear", align_corners=False)
            x = self.haze_decoder_condition1(x, haze_half)
            skip2_features = self.haze_skip_condition1(skip2_features, haze_half)
        if self.decoder_skip_modulation:
            skip2_features = self.skip_modulation1(skip2_features, x)
        x = self.layer4(self.fusion1([x, skip2_features]) + x)
        x = self.patch_split2(x)
        skip1_features = self.skip1(skip1)
        if self.haze_skip_guidance:
            if haze_map is None:
                raise RuntimeError("Haze-map skip guidance was enabled without a haze map.")
            haze_full = F.interpolate(haze_map, size=x.shape[-2:], mode="bilinear", align_corners=False)
            x = self.haze_decoder_condition2(x, haze_full)
            skip1_features = self.haze_skip_condition2(skip1_features, haze_full)
        if self.decoder_skip_modulation:
            skip1_features = self.skip_modulation2(skip1_features, x)
        x = self.fusion2([x, skip1_features]) + x
        if self.frequency_branch:
            x = x + torch.tanh(self.detail_gate) * self.detail_branch(hazy, x.shape[-2:])
        if self.wavelet_refinement:
            x = self.wavelet_refiner(x)
        x = self.layer5(x)
        k_and_bias = self.patch_unembed(x)
        k, bias = k_and_bias[:, :1], k_and_bias[:, 1:]
        restored = (k * hazy - bias + hazy)[:, :, :original_h, :original_w]
        return (restored, auxiliary) if return_aux else restored


def dehazeformer_t(
    use_depth: bool = False,
    *,
    fusion: str = "sk",
    haze_guidance: bool = False,
    frequency_branch: bool = False,
    wavelet_refinement: bool = False,
    haze_skip_guidance: bool = False,
    physical_consistency: bool = False,
    decoder_skip_modulation: bool = False,
    dcp_position_embedding: bool = False,
) -> DehazeFormer:
    return DehazeFormer(
        use_depth=use_depth,
        fusion=fusion,
        haze_guidance=haze_guidance,
        frequency_branch=frequency_branch,
        wavelet_refinement=wavelet_refinement,
        haze_skip_guidance=haze_skip_guidance,
        physical_consistency=physical_consistency,
        decoder_skip_modulation=decoder_skip_modulation,
        dcp_position_embedding=dcp_position_embedding,
    )
