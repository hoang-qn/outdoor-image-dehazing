"""Online MCBM-style non-homogeneous haze synthesis for paired training."""

from __future__ import annotations

import random
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image


class MCBMHazeAugment:
    """Synthesize a hazy input from a clear image and aligned relative depth.

    The spatial beta field is sampled from the MCBM density-map bank loaded
    from a configurable local data directory. Depth is used only to synthesize
    RGB training inputs; it is not passed to the DehazeFormer baseline.
    """

    def __init__(
        self,
        map_root: str | Path,
        beta_range: tuple[float, float] = (0.2, 2.8),
        airlight_range: tuple[float, float] = (0.25, 1.8),
        beta_map_scale_range: tuple[float, float] = (0.5, 1.0),
        depth_scale: float = 2.0,
    ) -> None:
        self.map_root = Path(map_root)
        self.map_paths = sorted(self.map_root.glob("*.png"))
        if not self.map_paths:
            raise FileNotFoundError(f"No MCBM density maps found in {self.map_root}.")
        self.beta_range = beta_range
        self.airlight_range = airlight_range
        self.beta_map_scale_range = beta_map_scale_range
        self.depth_scale = depth_scale

    @lru_cache(maxsize=128)
    def _density(self, map_path: Path, size: tuple[int, int]) -> np.ndarray:
        # Reuse resized maps within each persistent DataLoader worker.
        with Image.open(map_path) as image:
            density_image = image.convert("L").resize(size, Image.Resampling.BILINEAR)
        density = np.asarray(density_image, dtype=np.float32) / 255.0
        density_min, density_max = float(density.min()), float(density.max())
        if density_max > density_min:
            density = (density - density_min) / (density_max - density_min)
        return density

    @staticmethod
    def _depth_array(depth: Image.Image) -> np.ndarray:
        array = np.asarray(depth)
        if np.issubdtype(array.dtype, np.integer):
            array = array.astype(np.float32) / float(np.iinfo(array.dtype).max)
        else:
            array = array.astype(np.float32)
        if array.ndim == 3:
            array = array[..., 0]
        return np.clip(array, 0.0, 1.0)

    def __call__(self, clear: Image.Image, inverse_depth: Image.Image) -> Image.Image:
        size = clear.size
        clear_array = np.asarray(clear.convert("RGB"), dtype=np.float32) / 255.0

        # MiDaS maps in this project are normalized relative inverse depth;
        # invert them to obtain a near-to-far depth proxy as in HazeFlow.
        depth_array = 1.0 - self._depth_array(inverse_depth)
        if depth_array.shape[::-1] != size:
            depth_image = Image.fromarray(np.uint8(np.clip(depth_array, 0, 1) * 255))
            depth_array = np.asarray(depth_image.resize(size, Image.Resampling.BILINEAR), dtype=np.float32) / 255.0

        map_path = random.choice(self.map_paths)
        density = self._density(map_path, size)

        beta = random.uniform(*self.beta_range)
        beta_map_scale = random.uniform(*self.beta_map_scale_range)
        beta_field = beta + density * beta_map_scale
        transmission = np.exp(-self.depth_scale * depth_array * beta_field)[..., None]

        airlight = random.uniform(*self.airlight_range)
        airlight_rgb = np.clip(
            airlight + np.random.uniform(-0.025, 0.025, size=(1, 1, 3)), 0.0, 1.8
        )
        hazy = np.clip(clear_array * transmission + airlight_rgb * (1.0 - transmission), 0.0, 1.0)
        return Image.fromarray(np.uint8(np.rint(hazy * 255.0)), mode="RGB")
