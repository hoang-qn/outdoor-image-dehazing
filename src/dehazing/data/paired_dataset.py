"""Paired outdoor dehazing datasets driven by the supplied pairs.csv manifests."""

from __future__ import annotations

import csv
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from dehazing.data.spatial_haze import SpatialHazeAugment
from dehazing.data.mcbm_haze import MCBMHazeAugment


DATASET_DIRS = {
    "reside": "RESIDE-6K-Outdoor",
    "ohaze": "O-HAZE",
    "nhhaze": "NH-HAZE",
    "hdnhhaze": "HD-NH-HAZE",
    "nhhaze2": "NH-HAZE2",
}
DATASET_ALIASES = {
    **{name.lower(): name for name in DATASET_DIRS},
    **{directory.lower(): name for name, directory in DATASET_DIRS.items()},
}


def normalize_dataset_name(dataset: str) -> str:
    """Resolve a short dataset key or its on-disk directory name."""
    normalized = DATASET_ALIASES.get(dataset.lower())
    if normalized is None:
        raise ValueError(
            f"Unsupported dataset {dataset!r}. Choose from {tuple(DATASET_DIRS)} "
            f"or directory names {tuple(DATASET_DIRS.values())}."
        )
    return normalized


class PairedHazeDataset(Dataset[tuple[torch.Tensor, ...]]):
    """Read hazy/clear pairs from a dataset's pairs.csv file.

    The manifest is important for O-HAZE, where hazy and clear filenames differ.
    """

    def __init__(
        self,
        data_root: str | Path,
        dataset: str,
        split: str,
        crop_size: int | None = None,
        horizontal_flip: bool = False,
        additional_manifest: str | Path | None = None,
        spatial_haze: dict[str, object] | None = None,
        depth_root: str | Path | None = None,
        mcbm_haze: dict[str, object] | None = None,
        return_depth: bool | None = None,
        image_cache_root: str | Path | None = None,
    ) -> None:
        dataset = normalize_dataset_name(dataset)
        self.dataset_name = dataset
        self.dataset_root = Path(data_root) / DATASET_DIRS[dataset]
        self.split = split
        self.crop_size = crop_size
        self.horizontal_flip = horizontal_flip
        self.depth_root = Path(depth_root) if depth_root is not None else None
        self.return_depth = self.depth_root is not None if return_depth is None else return_depth
        self.image_cache_root = Path(image_cache_root) if image_cache_root is not None else None
        self.mcbm_haze_augment: MCBMHazeAugment | None = None
        if split == "train" and mcbm_haze and bool(mcbm_haze.get("enabled", False)):
            if self.depth_root is None:
                raise ValueError("MCBM haze synthesis requires data.depth_root with aligned depth maps.")
            self.mcbm_haze_augment = MCBMHazeAugment(
                map_root=mcbm_haze.get("map_root", "data/MCBM"),
                beta_range=tuple(mcbm_haze.get("beta_range", (0.2, 2.8))),  # type: ignore[arg-type]
                airlight_range=tuple(mcbm_haze.get("airlight_range", (0.25, 1.8))),  # type: ignore[arg-type]
                beta_map_scale_range=tuple(mcbm_haze.get("beta_map_scale_range", (0.5, 1.0))),  # type: ignore[arg-type]
                depth_scale=float(mcbm_haze.get("depth_scale", 2.0)),
            )
        self.spatial_haze_augment: SpatialHazeAugment | None = None
        if split == "train" and spatial_haze and bool(spatial_haze.get("enabled", False)):
            self.spatial_haze_augment = SpatialHazeAugment(
                probability=float(spatial_haze.get("probability", 0.75)),
                opacity_range=spatial_haze.get("opacity_range", (0.08, 0.28)),  # type: ignore[arg-type]
                spatial_strength=float(spatial_haze.get("spatial_strength", 0.6)),
                max_opacity=float(spatial_haze.get("max_opacity", 0.45)),
                grid_size=spatial_haze.get("grid_size", (3, 7)),  # type: ignore[arg-type]
                airlight_range=spatial_haze.get("airlight_range", (0.78, 1.0)),  # type: ignore[arg-type]
            )
        manifest = self.dataset_root / "pairs.csv"
        if not manifest.is_file():
            raise FileNotFoundError(f"Pair manifest not found: {manifest}")

        self.pairs: list[tuple[Path, Path]] = []
        with manifest.open("r", encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                row_split = row.get("split", "")
                if split != "all" and row_split != split:
                    continue
                hazy_rel = Path(row["hazy"].replace("\\", "/"))
                clear_rel = Path(row["clear"].replace("\\", "/"))
                hazy_path = self.dataset_root / hazy_rel
                clear_path = self.dataset_root / clear_rel
                if not hazy_path.is_file() or not clear_path.is_file():
                    raise FileNotFoundError(f"Missing image pair from {manifest}: {hazy_path} / {clear_path}")
                self.pairs.append((hazy_path, clear_path))
        self.additional_pairs: list[tuple[Path, Path, Path]] = []
        if additional_manifest is not None:
            extra_manifest = Path(additional_manifest)
            if not extra_manifest.is_file():
                raise FileNotFoundError(f"Additional pair manifest not found: {extra_manifest}")
            with extra_manifest.open("r", encoding="utf-8-sig", newline="") as stream:
                for row in csv.DictReader(stream):
                    row_split = row.get("split", "train")
                    if split != "all" and row_split != split:
                        continue
                    hazy_rel = Path(row["hazy"].replace("\\", "/"))
                    clear_rel = Path(row["clear"].replace("\\", "/"))
                    source_hazy_rel = Path(row.get("source_hazy", row["hazy"]).replace("\\", "/"))
                    hazy_path = self.dataset_root / hazy_rel
                    clear_path = self.dataset_root / clear_rel
                    source_hazy_path = self.dataset_root / source_hazy_rel
                    if not hazy_path.is_file() or not clear_path.is_file():
                        raise FileNotFoundError(
                            f"Missing image pair from {extra_manifest}: {hazy_path} / {clear_path}"
                        )
                    if not source_hazy_path.is_file():
                        raise FileNotFoundError(
                            f"Source hazy image from {extra_manifest} does not exist: {source_hazy_path}"
                        )
                    self.additional_pairs.append((hazy_path, clear_path, source_hazy_path))
        if not self.pairs:
            raise ValueError(f"No image pairs found for dataset={dataset!r}, split={split!r} in {manifest}")

    def __len__(self) -> int:
        mcbm_pairs = len(self.pairs) if self.mcbm_haze_augment is not None else 0
        return len(self.pairs) + mcbm_pairs + len(self.additional_pairs)

    @staticmethod
    def _to_tensor(image: Image.Image) -> torch.Tensor:
        array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        return torch.from_numpy(array.transpose(2, 0, 1).copy())

    @staticmethod
    def _to_depth_tensor(image: Image.Image) -> torch.Tensor:
        array = np.asarray(image)
        if np.issubdtype(array.dtype, np.integer):
            scale = float(np.iinfo(array.dtype).max)
            array = array.astype(np.float32) / scale
        else:
            array = array.astype(np.float32)
        if array.ndim == 3:
            array = array[..., 0]
        return torch.from_numpy(array[None].copy())

    def _cached_crop(
        self, hazy_path: Path, clear_path: Path, depth_path: Path | None, synthetic: bool,
    ) -> tuple[Image.Image, Image.Image, Image.Image | None]:
        """Memory-map lossless arrays and copy only the randomly selected crop."""
        cache = self.image_cache_root / DATASET_DIRS[self.dataset_name]

        def read(path: Path, kind: str, source_root: Path) -> np.ndarray:
            relative = path.relative_to(source_root)
            target = (cache / kind / relative).with_name(relative.name + ".npy")
            if not target.is_file():
                raise FileNotFoundError(f"Training cache missing: {target}; run prepare_training_cache.py.")
            return np.load(target, mmap_mode="r", allow_pickle=False)

        clear_array = read(clear_path, "rgb", self.dataset_root)
        hazy_array = None if synthetic else read(hazy_path, "rgb", self.dataset_root)
        depth_array = (
            read(depth_path, "depth", self.depth_root / DATASET_DIRS[self.dataset_name])
            if depth_path is not None else None
        )
        height, width = clear_array.shape[:2]
        if hazy_array is not None and hazy_array.shape != clear_array.shape:
            raise ValueError(f"Cached paired image dimensions differ: {hazy_path}")
        if depth_array is not None and depth_array.shape[:2] != (height, width):
            raise ValueError(f"Cached depth dimensions differ: {depth_path}")
        size = self.crop_size
        if min(width, height) < size:
            raise ValueError(f"Image {hazy_path} is smaller than crop_size={size}.")
        left, top = random.randint(0, width - size), random.randint(0, height - size)
        region = np.s_[top:top + size, left:left + size]
        clear = Image.fromarray(np.array(clear_array[region], copy=True))
        hazy = clear.copy() if synthetic else Image.fromarray(np.array(hazy_array[region], copy=True))
        depth = Image.fromarray(np.array(depth_array[region], copy=True)) if depth_array is not None else None
        return hazy, clear, depth

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        mcbm_variant = False
        if index < len(self.pairs):
            hazy_path, clear_path = self.pairs[index]
            depth_source_path = hazy_path
        elif self.mcbm_haze_augment is not None and index < 2 * len(self.pairs):
            hazy_path, clear_path = self.pairs[index - len(self.pairs)]
            depth_source_path = hazy_path
            mcbm_variant = True
        else:
            additional_index = index - len(self.pairs)
            if self.mcbm_haze_augment is not None:
                additional_index -= len(self.pairs)
            hazy_path, clear_path, depth_source_path = self.additional_pairs[additional_index]
        depth: Image.Image | None = None
        depth_path: Path | None = None
        if self.depth_root is not None and (self.return_depth or mcbm_variant):
            # Offline haze variants share the scene geometry with their source
            # image, so reuse that source image's aligned depth map.
            relative_hazy = depth_source_path.relative_to(self.dataset_root).with_suffix(".png")
            depth_path = self.depth_root / DATASET_DIRS[self.dataset_name] / relative_hazy
            if not depth_path.is_file():
                raise FileNotFoundError(
                    f"Depth map not found for {hazy_path}: {depth_path}. "
                    "Generate maps with scripts/generate_depth_maps.py first."
                )
        cached_crop = self.image_cache_root is not None and self.crop_size is not None
        if cached_crop:
            hazy, clear, depth = self._cached_crop(hazy_path, clear_path, depth_path, mcbm_variant)
        else:
            with Image.open(hazy_path) as image:
                hazy = image.convert("RGB")
            with Image.open(clear_path) as image:
                clear = image.convert("RGB")
            if depth_path is not None:
                with Image.open(depth_path) as image:
                    depth = image.copy()
        if hazy.size != clear.size:
            raise ValueError(f"Paired image dimensions differ: {hazy_path} {hazy.size} vs {clear_path} {clear.size}")
        if depth is not None and depth.size != hazy.size:
            raise ValueError(f"Depth map dimensions differ: {depth_path} {depth.size} vs {hazy_path} {hazy.size}")

        if self.crop_size is not None and not cached_crop:
            width, height = hazy.size
            if width < self.crop_size or height < self.crop_size:
                raise ValueError(
                    f"Image {hazy_path} is {width}x{height}, smaller than crop_size={self.crop_size}."
                )
            left = random.randint(0, width - self.crop_size)
            top = random.randint(0, height - self.crop_size)
            box = (left, top, left + self.crop_size, top + self.crop_size)
            hazy, clear = hazy.crop(box), clear.crop(box)
            if depth is not None:
                depth = depth.crop(box)
        if self.horizontal_flip and random.random() < 0.5:
            hazy = hazy.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            clear = clear.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            if depth is not None:
                depth = depth.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

        if mcbm_variant:
            if depth is None or self.mcbm_haze_augment is None:
                raise RuntimeError("MCBM training example is missing its aligned depth map.")
            hazy = self.mcbm_haze_augment(clear, depth)

        # Augment only the hazy input. The paired clear image remains the
        # supervised target, and validation/test datasets never enable this.
        if self.spatial_haze_augment is not None:
            hazy = self.spatial_haze_augment(hazy)

        if depth is not None and self.return_depth:
            return self._to_tensor(hazy), self._to_tensor(clear), self._to_depth_tensor(depth), hazy_path.name
        return self._to_tensor(hazy), self._to_tensor(clear), hazy_path.name
