"""On-the-fly smooth, spatially varying haze augmentation."""

from __future__ import annotations

import random
from typing import Sequence

import numpy as np
from PIL import Image


class SpatialHazeAugment:
    """Add a smooth, locally varying haze veil to a hazy training image.

    The clear target is deliberately left unchanged by the caller. The veil is
    generated from a low-resolution random field and composited using the
    atmospheric scattering form I_aug = I * (1 - alpha) + A * alpha.
    """

    def __init__(
        self,
        probability: float = 0.75,
        opacity_range: Sequence[float] = (0.08, 0.28),
        spatial_strength: float = 0.6,
        max_opacity: float = 0.45,
        grid_size: Sequence[int] = (3, 7),
        airlight_range: Sequence[float] = (0.78, 1.0),
    ) -> None:
        if not 0.0 <= probability <= 1.0:
            raise ValueError("spatial_haze probability must be between 0 and 1.")
        if len(opacity_range) != 2 or not 0 <= opacity_range[0] <= opacity_range[1] <= 1:
            raise ValueError("spatial_haze opacity_range must be [min, max] within [0, 1].")
        if not 0.0 <= spatial_strength <= 1.0:
            raise ValueError("spatial_haze spatial_strength must be between 0 and 1.")
        if not 0.0 <= max_opacity <= 1.0:
            raise ValueError("spatial_haze max_opacity must be between 0 and 1.")
        if len(grid_size) != 2 or int(grid_size[0]) < 2 or int(grid_size[1]) < int(grid_size[0]):
            raise ValueError("spatial_haze grid_size must be [min, max] with min >= 2.")
        if len(airlight_range) != 2 or not 0 <= airlight_range[0] <= airlight_range[1] <= 1:
            raise ValueError("spatial_haze airlight_range must be [min, max] within [0, 1].")

        self.probability = float(probability)
        self.opacity_range = (float(opacity_range[0]), float(opacity_range[1]))
        self.spatial_strength = float(spatial_strength)
        self.max_opacity = float(max_opacity)
        self.grid_size = (int(grid_size[0]), int(grid_size[1]))
        self.airlight_range = (float(airlight_range[0]), float(airlight_range[1]))

    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() >= self.probability:
            return image

        image_array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        height, width = image_array.shape[:2]

        # Bicubic upsampling gives broad, smooth density changes rather than
        # hard-edged rectangular patches.
        grid_h = random.randint(*self.grid_size)
        grid_w = random.randint(*self.grid_size)
        coarse = np.random.uniform(-1.0, 1.0, size=(grid_h, grid_w)).astype(np.float32)
        field_image = Image.fromarray(np.uint8((coarse + 1.0) * 127.5), mode="L")
        field_image = field_image.resize((width, height), Image.Resampling.BICUBIC)
        field = np.asarray(field_image, dtype=np.float32) / 127.5 - 1.0

        base_opacity = random.uniform(*self.opacity_range)
        alpha = base_opacity * (1.0 + self.spatial_strength * field)
        alpha = np.clip(alpha, 0.0, self.max_opacity)[..., None]

        airlight = np.random.uniform(*self.airlight_range, size=(1, 1, 3)).astype(np.float32)
        hazy = image_array * (1.0 - alpha) + airlight * alpha
        result = np.clip(hazy * 255.0 + 0.5, 0, 255).astype(np.uint8)
        return Image.fromarray(result, mode="RGB")
