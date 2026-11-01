"""Evaluate a configured dehazing model on paired dehazing datasets."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
from fvcore.nn import FlopCountAnalysis
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from dehazing.data.paired_dataset import DATASET_DIRS, PairedHazeDataset
from dehazing.metrics.image_metrics import image_quality
from dehazing.models import build_model


def _positions(length: int, tile_size: int, stride: int) -> list[int]:
    if length <= tile_size:
        return [0]
    starts = list(range(0, length - tile_size + 1, stride))
    if starts[-1] != length - tile_size:
        starts.append(length - tile_size)
    return starts


@torch.inference_mode()
def estimate_gflops(
    model: torch.nn.Module, device: torch.device, input_size: int, use_depth: bool
) -> tuple[float, list[str]]:
    """Estimate supported operator FLOPs for one standard RGB forward pass."""
    sample = torch.zeros((1, 3, input_size, input_size), device=device)
    inputs: torch.Tensor | tuple[torch.Tensor, torch.Tensor] = sample
    if use_depth:
        depth = torch.zeros((1, 1, input_size, input_size), device=device)
        inputs = (sample, depth)
    analysis = FlopCountAnalysis(model, inputs)
    total_flops = analysis.total()
    unsupported = sorted(str(op) for op in analysis.unsupported_ops())
    return float(total_flops / 1e9), unsupported


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def benchmark_fixed_input(
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int,
    input_size: int,
    warmup_iterations: int,
    benchmark_iterations: int,
    use_depth: bool,
) -> dict[str, float | int | str | bool]:
    """Measure steady-state FP32 model throughput at one fixed input shape."""
    sample = torch.rand((batch_size, 3, input_size, input_size), device=device)
    depth = torch.rand((batch_size, 1, input_size, input_size), device=device) if use_depth else None

    def forward() -> torch.Tensor:
        return model(sample, depth) if depth is not None else model(sample)
    previous_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    previous_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    previous_cudnn_benchmark = torch.backends.cudnn.benchmark
    try:
        # Keep the benchmark strictly FP32, including on Ampere GPUs.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = True
        for _ in range(warmup_iterations):
            output = forward()
        del output
        _synchronize(device)

        start = time.perf_counter()
        for _ in range(benchmark_iterations):
            output = forward()
        _synchronize(device)
        elapsed = time.perf_counter() - start
        del output
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_matmul_tf32
        torch.backends.cudnn.allow_tf32 = previous_cudnn_tf32
        torch.backends.cudnn.benchmark = previous_cudnn_benchmark

    total_images = batch_size * benchmark_iterations
    return {
        "precision": "FP32 (TF32 disabled)",
        "batch_size": batch_size,
        "input_size": f"{input_size}x{input_size}",
        "warmup_iterations": warmup_iterations,
        "benchmark_iterations": benchmark_iterations,
        "measured_images": total_images,
        "mean_batch_latency_ms": float(elapsed * 1000 / benchmark_iterations),
        "mean_image_latency_ms": float(elapsed * 1000 / total_images),
        "fps": float(total_images / elapsed),
        "timing_scope": "model only; input stays on device; excludes image I/O, transfers, tiling, and metrics",
    }


@torch.inference_mode()
def tiled_predict(
    model: torch.nn.Module,
    image: torch.Tensor,
    device: torch.device,
    amp: bool,
    tile_size: int,
    overlap: int,
    depth: torch.Tensor | None = None,
) -> torch.Tensor:
    """Restore one image, using overlapping tiles to cap memory on high-res sets."""
    _, _, height, width = image.shape
    if tile_size <= 0 or (height <= tile_size and width <= tile_size):
        with torch.amp.autocast(device_type=device.type, enabled=amp):
            rgb = image.to(device)
            depth_device = depth.to(device) if depth is not None else None
            prediction = model(rgb, depth_device) if depth_device is not None else model(rgb)
            return prediction.float().clamp_(0, 1).cpu()
    if overlap < 0 or overlap >= tile_size:
        raise ValueError("tile-overlap must be non-negative and smaller than tile-size")
    stride = max(1, tile_size - overlap)
    rows = _positions(height, tile_size, stride)
    cols = _positions(width, tile_size, stride)
    result = torch.zeros((1, 3, height, width), dtype=torch.float32)
    weights = torch.zeros((1, 1, height, width), dtype=torch.float32)

    for top in rows:
        for left in cols:
            bottom = min(top + tile_size, height)
            right = min(left + tile_size, width)
            tile = image[:, :, top:bottom, left:right].to(device)
            depth_tile = depth[:, :, top:bottom, left:right].to(device) if depth is not None else None
            with torch.amp.autocast(device_type=device.type, enabled=amp):
                prediction = (
                    model(tile, depth_tile) if depth_tile is not None else model(tile)
                ).float().clamp_(0, 1).cpu()
            tile_h, tile_w = bottom - top, right - left
            wy = torch.ones(tile_h)
            wx = torch.ones(tile_w)
            feather = min(overlap, tile_h // 2, tile_w // 2)
            if feather:
                ramp = torch.linspace(1.0 / (feather + 1), 1.0, feather)
                if top > 0:
                    wy[:feather] = ramp
                if bottom < height:
                    wy[-feather:] = ramp.flip(0)
                if left > 0:
                    wx[:feather] = ramp
                if right < width:
                    wx[-feather:] = ramp.flip(0)
            weight = (wy[:, None] * wx[None, :])[None, None]
            result[:, :, top:bottom, left:right] += prediction * weight
            weights[:, :, top:bottom, left:right] += weight
    return result / weights.clamp_min(1e-8)


def _save_prediction(image: np.ndarray, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    pixels = np.round(np.clip(image, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(pixels, mode="RGB").save(destination)


def evaluate_dataset(
    model: torch.nn.Module,
    data_root: Path,
    dataset_name: str,
    split: str,
    output_dir: Path,
    device: torch.device,
    amp: bool,
    tile_size: int,
    overlap: int,
    save_images: bool,
    gflops: float,
    gflops_input_size: int,
    depth_root: Path | None,
) -> dict[str, float | int | str]:
    dataset = PairedHazeDataset(data_root, dataset_name, split, depth_root=depth_root)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    dataset_label = DATASET_DIRS[dataset_name]
    pred_dir = output_dir / "predictions" / dataset_label
    rows: list[dict[str, object]] = []
    inference_seconds = 0.0

    model.eval()
    for batch in tqdm(loader, desc=f"Evaluate {dataset_label}"):
        if depth_root is not None:
            hazy, clear, depth, filename = batch
        else:
            hazy, clear, filename = batch
            depth = None
        _synchronize(device)
        inference_start = time.perf_counter()
        prediction = tiled_predict(model, hazy, device, amp, tile_size, overlap, depth)
        _synchronize(device)
        image_inference_seconds = time.perf_counter() - inference_start
        inference_seconds += image_inference_seconds
        prediction_np = prediction[0].numpy().transpose(1, 2, 0)
        clear_np = clear[0].numpy().transpose(1, 2, 0)
        score_psnr, score_ssim = image_quality(prediction_np, clear_np)
        image_name = filename[0]
        rows.append({
            "filename": image_name,
            "psnr": score_psnr,
            "ssim": score_ssim,
            "gflops_per_256x256": gflops,
            "inference_time_ms": image_inference_seconds * 1000.0,
        })
        if save_images:
            _save_prediction(prediction_np, pred_dir / image_name)

    summary = {
        "dataset": dataset_label,
        "split": split,
        "images": len(rows),
        "psnr_mean": float(np.mean([row["psnr"] for row in rows])),
        "ssim_mean": float(np.mean([row["ssim"] for row in rows])),
        "gflops_per_256x256": gflops,
        "gflops_input_resolution": f"{gflops_input_size}x{gflops_input_size}",
        "fps": float(len(rows) / inference_seconds) if inference_seconds > 0 else 0.0,
        "mean_inference_time_ms": float(inference_seconds * 1000.0 / max(len(rows), 1)),
        "inference_precision": "AMP autocast" if amp else "FP32",
        "inference_batch_size": 1,
        "inference_tile_size": tile_size,
        "inference_tile_overlap": overlap,
        "inference_timing_scope": (
            "per original-resolution image; includes tile transfers, model forward, and tile blending; "
            "excludes disk I/O, metric calculation, and image saving"
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / f"{dataset_name}_{split}_per_image.csv"
    with metrics_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("filename", "psnr", "ssim", "gflops_per_256x256", "inference_time_ms"),
        )
        writer.writeheader()
        writer.writerows(rows)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a dehazing model on paired datasets.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--depth-root", type=Path,
                        help="Root directory containing generated depth maps; required for depth checkpoints.")
    parser.add_argument(
        "--dataset",
        choices=("reside", "ohaze", "nhhaze", "all"),
        default="all",
        help="all evaluates RESIDE test, O-HAZE, and NH-HAZE.",
    )
    parser.add_argument(
        "--split",
        help="Split for RESIDE; defaults to test. O-HAZE and NH-HAZE use unsplit.",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--tile-size", type=int, default=512, help="Set 0 to run full-resolution inference.")
    parser.add_argument("--tile-overlap", type=int, default=64)
    parser.add_argument("--flops-size", type=int, default=256,
                        help="Square RGB input size used for the standardized GFLOPs estimate.")
    parser.add_argument("--benchmark-batch-size", type=int, default=16,
                        help="Batch size for the fixed-shape FP32 throughput benchmark.")
    parser.add_argument("--benchmark-size", type=int, default=512,
                        help="Square input size for the fixed-shape FP32 throughput benchmark.")
    parser.add_argument("--benchmark-warmup", type=int, default=50,
                        help="Unmeasured warm-up iterations for the fixed-shape benchmark.")
    parser.add_argument("--benchmark-iterations", type=int, default=200,
                        help="Measured iterations for the fixed-shape benchmark.")
    parser.add_argument(
        "--skip-fixed-benchmark",
        action="store_true",
        help="Skip the separate fixed-input benchmark and reuse it from an existing summary when available.",
    )
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-save-images", action="store_true")
    args = parser.parse_args()
    if args.benchmark_batch_size < 1 or args.benchmark_size < 1:
        parser.error("benchmark batch size and input size must be positive")
    if args.benchmark_warmup < 0 or args.benchmark_iterations < 1:
        parser.error("benchmark warmup must be non-negative and iterations must be positive")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda" and not args.no_amp
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model", checkpoint)
    checkpoint_config = checkpoint.get("config", {})
    model_config = checkpoint_config.get("model", {})
    model_name = str(model_config.get("name", "dehazeformer_t"))
    use_depth = bool(model_config.get("use_depth", False))
    depth_root = None
    if use_depth:
        configured_depth_root = checkpoint_config.get("data", {}).get("depth_root")
        depth_root = args.depth_root or (Path(configured_depth_root) if configured_depth_root else args.data_root / "depth_maps")
    elif args.depth_root is not None:
        parser.error("--depth-root was set, but the checkpoint is not configured for depth input.")
    model = build_model(
        model_name,
        use_depth=use_depth,
        base_dim=int(model_config.get("base_dim", 32)),
    ).to(device)
    model.load_state_dict(state_dict)
    if hasattr(model, "reparameterize"):
        model.reparameterize()
    model.eval()
    gflops, unsupported_flop_ops = estimate_gflops(model, device, args.flops_size, use_depth)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    run_dir = args.output_dir or args.checkpoint.resolve().parent.parent / "evaluation"
    fixed_input_benchmark = None
    if args.skip_fixed_benchmark:
        existing_summary = run_dir / "summary.json"
        if existing_summary.is_file():
            with existing_summary.open("r", encoding="utf-8") as stream:
                fixed_input_benchmark = json.load(stream).get("fixed_input_benchmark")
    else:
        fixed_input_benchmark = benchmark_fixed_input(
            model,
            device,
            args.benchmark_batch_size,
            args.benchmark_size,
            args.benchmark_warmup,
            args.benchmark_iterations,
            use_depth,
        )

    if args.dataset == "all":
        datasets = ["reside", "ohaze", "nhhaze"]
    else:
        datasets = [args.dataset]
    summaries = []
    for dataset_name in datasets:
        split = args.split or ("test" if dataset_name == "reside" else "unsplit")
        summaries.append(evaluate_dataset(
            model, args.data_root, dataset_name, split, run_dir,
            device, amp, args.tile_size, args.tile_overlap, not args.no_save_images,
            gflops, args.flops_size, depth_root,
        ))

    summary_path = run_dir / "summary.json"
    hardware = torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor() or "CPU"
    summary = {
        "model": model_name,
        "parameters": parameter_count,
        "parameters_millions": parameter_count / 1_000_000.0,
        "hardware": hardware,
        "torch_version": torch.__version__,
        "mixed_precision": amp,
        "depth_conditioned": use_depth,
        "gflops_method": "fvcore FlopCountAnalysis; unsupported operators are not counted",
        "unsupported_flop_ops": unsupported_flop_ops,
        "fixed_input_benchmark": fixed_input_benchmark,
        "tile_size": args.tile_size,
        "tile_overlap": args.tile_overlap,
        "datasets": summaries,
    }
    with summary_path.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"Saved per-image metrics and predictions under {run_dir}")


if __name__ == "__main__":
    main()
