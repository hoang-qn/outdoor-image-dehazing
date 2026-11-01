"""Generate normalized, full-resolution MiDaS depth maps from hazy inputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from tqdm import tqdm

DATASET_DIRS = {
    "reside": "RESIDE-6K-Outdoor",
    "ohaze": "O-HAZE",
    "nhhaze": "NH-HAZE",
    "nhhaze2": "NH-HAZE2",
    "hdnhhaze": "HD-NH-HAZE",
}


def _images_from_manifest(
    data_root: Path, dataset: str, split: str = "all"
) -> list[tuple[Path, Path]]:
    dataset_root = data_root / DATASET_DIRS[dataset]
    manifest = dataset_root / "pairs.csv"
    if not manifest.is_file():
        raise FileNotFoundError(f"Pair manifest not found: {manifest}")
    images: dict[Path, Path] = {}
    with manifest.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if split != "all" and row.get("split", "") != split:
                continue
            relative = Path(row["hazy"].replace("\\", "/"))
            source = dataset_root / relative
            target = relative.with_suffix(".png")
            if not source.is_file():
                raise FileNotFoundError(f"Hazy image from {manifest} is missing: {source}")
            images[source] = target
    return [(source, images[source]) for source in sorted(images)]


def _normalized_depth(prediction: torch.Tensor) -> np.ndarray:
    values = prediction.detach().float().cpu().numpy()
    low, high = np.percentile(values, (2.0, 98.0))
    if not np.isfinite(low) or not np.isfinite(high) or high - low < 1e-6:
        return np.zeros(values.shape, dtype=np.uint16)
    normalized = np.clip((values - low) / (high - low), 0.0, 1.0)
    return np.round(normalized * 65535.0).astype(np.uint16)


@torch.inference_mode()
def _predict_depth(
    image: Image.Image,
    model: torch.nn.Module,
    transform,
    device: torch.device,
) -> np.ndarray:
    rgb = np.asarray(image, dtype=np.uint8)
    model_input = transform(rgb).to(device)
    prediction = model(model_input)
    prediction = F.interpolate(
        prediction.unsqueeze(1),
        size=(image.height, image.width),
        mode="bicubic",
        align_corners=False,
    )[0, 0]
    return _normalized_depth(prediction)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate offline MiDaS depth maps for hazy images.")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--depth-root", type=Path, default=Path("data/depth_maps"))
    parser.add_argument("--dataset", choices=(*DATASET_DIRS, "all"), default="reside")
    parser.add_argument("--split", choices=("train", "val", "test", "all"), default="all")
    parser.add_argument("--model-type", choices=("MiDaS_small", "DPT_Hybrid"), default="MiDaS_small")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    data_root = args.data_root.resolve()
    depth_root = args.depth_root.resolve()
    datasets = list(DATASET_DIRS) if args.dataset == "all" else [args.dataset]
    inputs: list[tuple[str, Path, Path]] = []
    for dataset in datasets:
        for source, relative_target in _images_from_manifest(data_root, dataset, args.split):
            target = depth_root / DATASET_DIRS[dataset] / relative_target
            if args.overwrite or not target.is_file():
                inputs.append((dataset, source, target))

    if not inputs:
        print(f"Depth maps already exist under {depth_root}; nothing to generate.")
        return

    try:
        model = torch.hub.load("intel-isl/MiDaS", args.model_type, pretrained=True, trust_repo=True)
        transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
    except ImportError as exc:
        raise RuntimeError(
            "MiDaS dependencies are missing. Install this project with depth support: "
            "python -m pip install -r requirements.txt after selecting the PyTorch build for the target machine."
        ) from exc

    transform = transforms.dpt_transform if args.model_type == "DPT_Hybrid" else transforms.small_transform
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    print(f"Estimator: MiDaS {args.model_type}; device: {device}; maps to generate: {len(inputs)}")

    for dataset, source, target in tqdm(inputs, desc="Depth maps"):
        target.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(source) as opened:
            image = opened.convert("RGB")
            depth = _predict_depth(image, model, transform, device)
        Image.fromarray(depth, mode="I;16").save(target)

    metadata_path = depth_root / "depth_metadata.json"
    previous_metadata = {}
    if metadata_path.is_file():
        previous_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    generated_splits = previous_metadata.get("generated_splits")
    if generated_splits is None:
        generated_splits = {
            name: ["all"] for name in previous_metadata.get("datasets", [])
        }
    for dataset in datasets:
        if args.split == "all":
            generated_splits[dataset] = ["all"]
        elif "all" not in generated_splits.get(dataset, []):
            generated_splits[dataset] = sorted(
                set(generated_splits.get(dataset, [])) | {args.split}
            )
    metadata = {
        "estimator": "MiDaS",
        "model_type": args.model_type,
        "device": str(device),
        "input": "hazy RGB image",
        "normalization": "per-image 2nd-98th percentile, clipped to [0,1], stored as uint16 PNG",
        "datasets": sorted(generated_splits),
        "generated_splits": generated_splits,
    }
    depth_root.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Saved depth maps and metadata under {depth_root}")


if __name__ == "__main__":
    main()
