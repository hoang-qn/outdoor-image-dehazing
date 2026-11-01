"""Build lossless, memory-mapped RGB/depth arrays for random-crop training."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import yaml
from PIL import Image
from tqdm import tqdm

from dehazing.data.paired_dataset import DATASET_DIRS, normalize_dataset_name


def prepare(config_path: Path) -> dict[str, int]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data = config["data"]
    cache_root = Path(data["image_cache_root"])
    sources = data["train"].get("datasets", [data["train"]])
    use_mcbm = config["training"].get("augmentation", {}).get("mcbm_haze", {}).get("enabled", False)
    jobs: dict[Path, tuple[Path, str]] = {}
    pairs = 0
    for source in sources:
        directory = DATASET_DIRS[normalize_dataset_name(source["dataset"])]
        dataset_root = Path(data["root"]) / directory
        with (dataset_root / "pairs.csv").open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                if row["split"] != source.get("split", "train"):
                    continue
                pairs += 1
                dimensions = []
                for role in ("hazy", "clear"):
                    relative = Path(row[role].replace("\\", "/"))
                    original = dataset_root / relative
                    with Image.open(original) as image:
                        dimensions.append(image.size)
                    target = (cache_root / directory / "rgb" / relative).with_name(relative.name + ".npy")
                    jobs[target] = (original, "RGB")
                if use_mcbm:
                    relative = Path(row["hazy"].replace("\\", "/")).with_suffix(".png")
                    original = Path(data["depth_root"]) / directory / relative
                    with Image.open(original) as image:
                        dimensions.append(image.size)
                    target = (cache_root / directory / "depth" / relative).with_name(relative.name + ".npy")
                    jobs[target] = (original, "depth")
                if len(set(dimensions)) != 1:
                    raise ValueError(f"Image/depth alignment differs: {directory}, {row['hazy']}: {dimensions}")

    missing_bytes = 0
    pending = []
    for target, (original, mode) in jobs.items():
        stat = original.stat()
        signature = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "mode": mode}
        metadata = target.with_suffix(target.suffix + ".json")
        if target.is_file() and metadata.is_file():
            try:
                if json.loads(metadata.read_text(encoding="utf-8")) == signature:
                    continue
            except (OSError, ValueError):
                pass
        with Image.open(original) as image:
            missing_bytes += image.width * image.height * (3 if mode == "RGB" else 2) + 128
        pending.append((original, target, metadata, mode, signature))
    cache_root.mkdir(parents=True, exist_ok=True)
    if missing_bytes * 1.1 > shutil.disk_usage(cache_root).free:
        raise OSError(f"Cache requires approximately {missing_bytes / 2**30:.2f} GiB of free disk space.")
    print(f"Cache: {pairs} aligned pairs, {len(jobs)} unique arrays; {len(pending)} to build, "
          f"approximately {missing_bytes / 2**30:.2f} GiB. Existing valid arrays are reused.", flush=True)

    def save(job) -> None:
        original, target, metadata, mode, signature = job
        target.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(original) as opened:
            image = opened.convert("RGB") if mode == "RGB" else opened
            array = np.asarray(image)
            if mode == "depth" and array.dtype != np.uint16:
                raise ValueError(f"Expected normalized uint16 depth: {original}, got {array.dtype}")
            temporary = target.with_suffix(target.suffix + ".tmp")
            with temporary.open("wb") as stream:
                np.save(stream, array, allow_pickle=False)
            temporary.replace(target)
        metadata.write_text(json.dumps(signature), encoding="utf-8")

    # Bound peak decoded-image RAM, including very large HD-NH-HAZE pairs.
    with ThreadPoolExecutor(max_workers=2) as executor:
        list(tqdm(executor.map(save, pending), total=len(pending), desc="Training cache"))
    return {"original_training_pairs": pairs, "effective_training_pairs": pairs * (2 if use_mcbm else 1),
            "cache_arrays": len(jobs)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.config), indent=2))
