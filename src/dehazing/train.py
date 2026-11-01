"""Train a configured dehazing model on a prepared paired-image split."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from torch.nn import functional as F
from torch.utils.data import ConcatDataset, DataLoader, Sampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dehazing.data.paired_dataset import DATASET_DIRS, PairedHazeDataset, normalize_dataset_name
from dehazing.evaluate import tiled_predict
from dehazing.metrics.image_metrics import ssim_tensor
from dehazing.models import build_model
from dehazing.utils.reproducibility import seed_everything


def _worker_seed(worker_id: int) -> None:
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


class DomainBalancedBatchSampler(Sampler[list[int]]):
    """Draw a fixed number of examples from each concatenated dataset per batch."""

    def __init__(
        self,
        dataset_lengths: list[int],
        samples_per_batch: list[int],
        steps_per_epoch: int | None = None,
        seed: int = 42,
    ) -> None:
        if len(dataset_lengths) != len(samples_per_batch) or len(dataset_lengths) < 2:
            raise ValueError("Domain-balanced sampling needs matching lists for at least two datasets.")
        if any(length <= 0 for length in dataset_lengths):
            raise ValueError(f"Every sampled dataset must be non-empty; got lengths={dataset_lengths}.")
        if any(count <= 0 for count in samples_per_batch):
            raise ValueError(f"Every domain must contribute at least one sample; got {samples_per_batch}.")
        if steps_per_epoch is not None and steps_per_epoch <= 0:
            raise ValueError("training.sampling.steps_per_epoch must be positive or null.")

        self.dataset_lengths = dataset_lengths
        self.samples_per_batch = samples_per_batch
        self.steps_per_epoch = steps_per_epoch or math.ceil(dataset_lengths[0] / samples_per_batch[0])
        self.seed = seed
        self.epoch = 0
        self.offsets: list[int] = []
        offset = 0
        for length in dataset_lengths:
            self.offsets.append(offset)
            offset += length

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.steps_per_epoch

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        pools: list[list[int]] = [[] for _ in self.dataset_lengths]

        def draw(domain_id: int, count: int) -> list[int]:
            selected: list[int] = []
            while len(selected) < count:
                if not pools[domain_id]:
                    pools[domain_id] = list(range(self.dataset_lengths[domain_id]))
                    rng.shuffle(pools[domain_id])
                selected.append(pools[domain_id].pop() + self.offsets[domain_id])
            return selected

        for _ in range(self.steps_per_epoch):
            batch: list[int] = []
            for domain_id, count in enumerate(self.samples_per_batch):
                batch.extend(draw(domain_id, count))
            rng.shuffle(batch)
            yield batch


def _unpack_batch(batch: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    if len(batch) == 4:
        hazy, clear, depth, _ = batch
        return hazy, clear, depth
    hazy, clear, _ = batch
    return hazy, clear, None


def _validation_image_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    tile_size: int,
    window_size: int = 11,
) -> tuple[float, float, float]:
    """Compute full-image L1, PSNR, and SSIM with bounded temporary memory."""
    _, _, height, width = prediction.shape
    tile_size = max(tile_size, window_size)
    abs_error_sum = 0.0
    squared_error_sum = 0.0
    value_count = 0
    for top in range(0, height, tile_size):
        for left in range(0, width, tile_size):
            pred_tile = prediction[:, :, top : top + tile_size, left : left + tile_size]
            target_tile = target[:, :, top : top + tile_size, left : left + tile_size]
            difference = pred_tile - target_tile
            abs_error_sum += float(difference.abs().sum().item())
            squared_error_sum += float(difference.square().sum().item())
            value_count += difference.numel()

    l1_value = abs_error_sum / max(value_count, 1)
    mse = squared_error_sum / max(value_count, 1)
    psnr_value = float("inf") if mse == 0 else float(10.0 * np.log10(1.0 / mse))

    # SSIM uses valid 11x11 windows. Partition its score map into blocks while
    # including the window halo, so this matches a full-image SSIM calculation.
    map_height = height - window_size + 1
    map_width = width - window_size + 1
    ssim_sum = 0.0
    ssim_count = 0
    for top in range(0, map_height, tile_size):
        score_bottom = min(top + tile_size, map_height)
        input_bottom = score_bottom + window_size - 1
        for left in range(0, map_width, tile_size):
            score_right = min(left + tile_size, map_width)
            input_right = score_right + window_size - 1
            score = ssim_tensor(
                prediction[:, :, top:input_bottom, left:input_right],
                target[:, :, top:input_bottom, left:input_right],
                window_size=window_size,
            )
            block_count = (score_bottom - top) * (score_right - left)
            ssim_sum += float(score.item()) * block_count
            ssim_count += block_count
    return l1_value, psnr_value, ssim_sum / max(ssim_count, 1)


def _forward(model: nn.Module, hazy: torch.Tensor, depth: torch.Tensor | None) -> torch.Tensor:
    return model(hazy, depth) if depth is not None else model(hazy)


def _haze_proxy(hazy: torch.Tensor, clear: torch.Tensor, output_size: tuple[int, int]) -> torch.Tensor:
    """Build the notebook's normalized |hazy - clear| auxiliary haze target."""
    proxy = (hazy.float() - clear.float()).abs().mean(dim=1, keepdim=True)
    proxy = F.interpolate(proxy, size=output_size, mode="area")
    denominator = proxy.amax(dim=(2, 3), keepdim=True).clamp_min(1e-6)
    return (proxy / denominator).clamp(0.0, 1.0)


def _initialize_from_checkpoint(model: nn.Module, checkpoint_path: Path, use_depth: bool) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    source = checkpoint.get("model", checkpoint)
    target = model.state_dict()
    adapted_depth_stem = False
    for name, target_value in target.items():
        if name not in source:
            raise KeyError(f"Initialization checkpoint is missing model parameter {name!r}.")
        source_value = source[name]
        if target_value.shape == source_value.shape:
            target[name] = source_value
        elif (
            use_depth
            and name == "patch_embed.proj.weight"
            and target_value.ndim == 4
            and target_value.shape[1] == 4
            and source_value.ndim == 4
            and source_value.shape[1] == 3
            and target_value.shape[0] == source_value.shape[0]
            and target_value.shape[2:] == source_value.shape[2:]
        ):
            target_value.zero_()
            target_value[:, :3] = source_value
            target[name] = target_value
            adapted_depth_stem = True
        else:
            raise ValueError(
                f"Cannot initialize {name}: checkpoint shape {tuple(source_value.shape)} "
                f"does not match model shape {tuple(target_value.shape)}."
            )
    model.load_state_dict(target, strict=True)
    print(f"Initialized from {checkpoint_path} (depth stem adapted: {adapted_depth_stem}).")


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp: bool,
    perceptual_model: nn.Module | None = None,
    tile_size: int = 0,
    tile_overlap: int = 64,
) -> dict[str, float | None]:
    model.eval()
    l1_values: list[float] = []
    lpips_values: list[float] = []
    psnr_values: list[float] = []
    ssim_values: list[float] = []
    for batch in tqdm(loader, desc="Validation", leave=False):
        hazy, clear, depth = _unpack_batch(batch)
        restored = tiled_predict(model, hazy, device, amp, tile_size, tile_overlap, depth)
        for prediction, target in zip(restored, clear):
            prediction = prediction.unsqueeze(0).float().clamp(0, 1)
            target = target.unsqueeze(0).float()
            l1_value, psnr_value, ssim_value = _validation_image_metrics(
                prediction, target, tile_size if tile_size > 0 else 512
            )
            l1_values.append(l1_value)
            psnr_values.append(psnr_value)
            ssim_values.append(ssim_value)
            if perceptual_model is not None:
                _, _, height, width = prediction.shape
                lpips_tile_size = max(tile_size, 64)
                lpips_weighted_sum = 0.0
                lpips_pixel_count = 0
                for top in range(0, height, lpips_tile_size):
                    for left in range(0, width, lpips_tile_size):
                        pred_tile = prediction[:, :, top : top + lpips_tile_size,
                                               left : left + lpips_tile_size].to(device)
                        target_tile = target[:, :, top : top + lpips_tile_size,
                                             left : left + lpips_tile_size].to(device)
                        lpips_value = perceptual_model(pred_tile, target_tile, normalize=True)
                        pixel_count = pred_tile.shape[-2] * pred_tile.shape[-1]
                        lpips_weighted_sum += float(lpips_value.mean().item()) * pixel_count
                        lpips_pixel_count += pixel_count
                lpips_values.append(lpips_weighted_sum / max(lpips_pixel_count, 1))
    return {
        "l1": float(np.mean(l1_values)),
        "lpips": float(np.mean(lpips_values)) if lpips_values else None,
        "psnr": float(np.mean(psnr_values)),
        "ssim": float(np.mean(ssim_values)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the configured dehazing model.")
    parser.add_argument("--config", type=Path, default=Path("configs/experiment/baseline.yaml"))
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--crop-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", type=Path, help="Resume from a saved last/best checkpoint.")
    parser.add_argument("--stop-file", type=Path,
                        help="If created, finish the current epoch, save checkpoints, and then stop.")
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision on CUDA.")
    args = parser.parse_args()

    with args.config.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    data_config = config["data"]
    data_root = args.data_root or Path(data_config["root"])
    train_data_config = data_config.get("train", {})
    val_data_config = data_config.get("validation", {})
    train_sources = train_data_config.get("datasets")
    if train_sources is None:
        train_sources = [train_data_config]
    if not isinstance(train_sources, list) or not train_sources:
        raise ValueError("data.train.datasets must be a non-empty list when configured.")
    default_train_dataset = str(train_sources[0].get("dataset", "reside"))
    val_sources = val_data_config.get("datasets")
    if val_sources is None:
        val_sources = [val_data_config or {"dataset": default_train_dataset, "split": "val"}]
    if not isinstance(val_sources, list) or not val_sources:
        raise ValueError("data.validation.datasets must be a non-empty list when configured.")
    model_config = config.get("model", {})
    use_depth = bool(model_config.get("use_depth", False))
    training = config["training"]
    if training.get("cpu_threads"):
        torch.set_num_threads(int(training["cpu_threads"]))
    mcbm_haze_config = training.get("augmentation", {}).get("mcbm_haze", {})
    use_mcbm_haze = bool(mcbm_haze_config.get("enabled", False))
    depth_root_value = config.get("data", {}).get("depth_root")
    if (use_depth or use_mcbm_haze) and not depth_root_value:
        raise ValueError("data.depth_root must be configured for depth input or MCBM haze synthesis.")
    depth_root = Path(depth_root_value) if (use_depth or use_mcbm_haze) else None
    epochs = args.epochs or int(training["epochs"])
    batch_size = args.batch_size or int(training["batch_size"])
    crop_size = args.crop_size or int(training["crop_size"])
    learning_rate = args.learning_rate or float(training["learning_rate"])
    num_workers = args.num_workers if args.num_workers is not None else int(config["data"]["num_workers"])
    validation_num_workers = int(config["data"].get("validation_num_workers", num_workers))
    validation_tile_size = int(config["data"].get("validation_tile_size", 0))
    validation_tile_overlap = int(config["data"].get("validation_tile_overlap", 64))
    output_dir = args.output_dir or Path(config["output"]["root"]) / config["output"]["experiment_name"]
    run_dir = output_dir
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    seed = int(config["seed"])
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and bool(training.get("mixed_precision", True)) and not args.no_amp
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = bool(training.get("deterministic", True))
    print(f"Device: {device}; mixed precision: {use_amp}; seed: {seed}")

    white_haze_config = training.get("augmentation", {}).get("white_haze_density", {})
    augmentation_config = training.get("augmentation", {})
    spatial_haze_config = augmentation_config.get("spatial_haze", {})
    train_datasets = []
    train_source_labels = []
    train_source_names = []
    for source in train_sources:
        dataset_name = normalize_dataset_name(str(source.get("dataset", "reside")))
        split = str(source.get("split", "train"))
        additional_manifest = source.get("additional_manifest")
        if additional_manifest is None and len(train_sources) == 1 and white_haze_config.get("enabled", False):
            additional_manifest = white_haze_config.get("offline_manifest")
        dataset = PairedHazeDataset(
            data_root,
            dataset_name,
            split,
            crop_size,
            horizontal_flip=bool(augmentation_config.get("horizontal_flip", True)),
            additional_manifest=additional_manifest,
            spatial_haze=spatial_haze_config,
            depth_root=depth_root,
            mcbm_haze=mcbm_haze_config,
            return_depth=use_depth,
            image_cache_root=config["data"].get("image_cache_root"),
        )
        train_datasets.append(dataset)
        train_source_names.append(DATASET_DIRS[dataset_name])
        train_source_labels.append(f"{DATASET_DIRS[dataset_name]}/{split}: {len(dataset)} pairs")
        if dataset.mcbm_haze_augment is not None:
            print(
                f"MCBM haze synthesis enabled for {DATASET_DIRS[dataset_name]}/{split}: "
                f"{len(dataset.mcbm_haze_augment.map_paths)} density maps; "
                f"{len(dataset.pairs)} original + {len(dataset.pairs)} synthetic pairs."
            )
    train_data = train_datasets[0] if len(train_datasets) == 1 else ConcatDataset(train_datasets)

    if len(train_datasets) == 1 and train_datasets[0].spatial_haze_augment is not None:
        spatial_haze = train_data.spatial_haze_augment
        print(
            "Spatial haze augmentation: "
            f"p={spatial_haze.probability:g}, "
            f"opacity_range={spatial_haze.opacity_range}, "
            f"spatial_strength={spatial_haze.spatial_strength:g}, "
            f"max_opacity={spatial_haze.max_opacity:g}, "
            f"grid_size={spatial_haze.grid_size}, "
            f"airlight_range={spatial_haze.airlight_range}"
        )
    val_datasets = []
    val_source_labels = []
    for source in val_sources:
        dataset_name = normalize_dataset_name(str(source.get("dataset", default_train_dataset)))
        split = str(source.get("split", "val"))
        dataset = PairedHazeDataset(
            data_root, dataset_name, split,
            depth_root=depth_root if use_depth else None,
            return_depth=use_depth,
        )
        val_datasets.append(dataset)
        val_source_labels.append(f"{DATASET_DIRS[dataset_name]}/{split}: {len(dataset)} pairs")
    val_data = val_datasets[0] if len(val_datasets) == 1 else ConcatDataset(val_datasets)
    print(f"Training pairs: {len(train_data)} ({'; '.join(train_source_labels)}).")
    print(f"Validation pairs: {len(val_data)} ({'; '.join(val_source_labels)}).")
    generator = torch.Generator().manual_seed(seed)
    loader_args = {
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": num_workers > 0,
        "worker_init_fn": _worker_seed,
    }
    if num_workers > 0:
        loader_args["prefetch_factor"] = int(config["data"].get("prefetch_factor", 2))
    validation_loader_args = {
        **loader_args,
        "num_workers": validation_num_workers,
        "persistent_workers": validation_num_workers > 0,
    }
    validation_loader_args.pop("prefetch_factor", None)
    if validation_num_workers > 0:
        validation_loader_args["prefetch_factor"] = int(config["data"].get("validation_prefetch_factor", 1))
    sampling_config = training.get("sampling", {})
    sampling_strategy = str(sampling_config.get("strategy", "shuffle")).lower()
    if sampling_strategy == "domain_balanced":
        samples_by_domain = sampling_config.get("samples_per_batch", {})
        if not isinstance(samples_by_domain, dict):
            raise ValueError("training.sampling.samples_per_batch must map dataset names to counts.")
        missing_domains = [name for name in train_source_names if name not in samples_by_domain]
        extra_domains = sorted(set(samples_by_domain) - set(train_source_names))
        if missing_domains or extra_domains:
            raise ValueError(
                "Domain-balanced sampling domains must match data.train.datasets exactly; "
                f"missing={missing_domains}, extra={extra_domains}."
            )
        samples_per_batch = [int(samples_by_domain[name]) for name in train_source_names]
        if sum(samples_per_batch) != batch_size:
            raise ValueError(
                "The sum of training.sampling.samples_per_batch must equal training.batch_size; "
                f"got {sum(samples_per_batch)} and {batch_size}."
            )
        configured_steps = sampling_config.get("steps_per_epoch")
        batch_sampler = DomainBalancedBatchSampler(
            [len(dataset) for dataset in train_datasets],
            samples_per_batch,
            int(configured_steps) if configured_steps is not None else None,
            seed,
        )
        train_loader = DataLoader(
            train_data, batch_sampler=batch_sampler, generator=generator, **loader_args,
        )
        sampled_per_epoch = [count * len(batch_sampler) for count in samples_per_batch]
        domain_summary = "; ".join(
            f"{name}: {count}/{len(dataset)} per batch, {total} draws/epoch"
            for name, count, total, dataset in zip(
                train_source_names, samples_per_batch, sampled_per_epoch, train_datasets
            )
        )
        print(
            f"Training sampler: domain_balanced; {len(batch_sampler)} steps/epoch; "
            f"batch size {sum(samples_per_batch)}. {domain_summary}."
        )
    elif sampling_strategy == "shuffle":
        batch_sampler = None
        train_loader = DataLoader(
            train_data, batch_size=batch_size, shuffle=True, drop_last=False,
            generator=generator, **loader_args,
        )
    else:
        raise ValueError(
            f"Unsupported training.sampling.strategy={sampling_strategy!r}; "
            "expected 'shuffle' or 'domain_balanced'."
        )
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False, **validation_loader_args)

    model_name = str(model_config.get("name", "dehazeformer_t"))
    model = build_model(
        model_name,
        use_depth=use_depth,
        base_dim=int(model_config.get("base_dim", 32)),
    ).to(device)
    initialization_checkpoint = model_config.get("init_checkpoint")
    if initialization_checkpoint:
        _initialize_from_checkpoint(model, Path(initialization_checkpoint), use_depth)
    optimizer_args = {}
    if training.get("fused_optimizer", False) and device.type == "cuda":
        optimizer_args["fused"] = True
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=float(training.get("weight_decay", 1e-4)),
        **optimizer_args,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    criterion = nn.L1Loss()
    loss_config = training.get("loss", {})
    loss_type = str(loss_config.get("type", "l1")).lower()
    perceptual_weight = float(loss_config.get("lpips_weight", 0.0))
    haze_aux_weight = float(training.get("haze_aux_weight", 0.0))
    physical_consistency_weight = float(training.get("physical_consistency_weight", 0.0))
    if haze_aux_weight < 0:
        raise ValueError("training.haze_aux_weight must be non-negative.")
    if haze_aux_weight > 0 and not bool(getattr(model, "haze_guidance", False)):
        raise ValueError("training.haze_aux_weight requires a model with haze guidance (M2, M3, or M5).")
    if physical_consistency_weight < 0:
        raise ValueError("training.physical_consistency_weight must be non-negative.")
    if physical_consistency_weight > 0 and not bool(getattr(model, "physical_consistency", False)):
        raise ValueError(
            "training.physical_consistency_weight requires a model with physical-consistency heads (M6)."
        )
    perceptual_model: nn.Module | None = None
    if loss_type == "l1_lpips":
        if perceptual_weight <= 0:
            raise ValueError("training.loss.lpips_weight must be > 0 for loss.type=l1_lpips.")
        from lpips import LPIPS

        perceptual_model = LPIPS(net=str(loss_config.get("lpips_net", "alex")), verbose=False)
        perceptual_model = perceptual_model.to(device).eval()
        for parameter in perceptual_model.parameters():
            parameter.requires_grad_(False)
    elif loss_type != "l1":
        raise ValueError(f"Unsupported training.loss.type={loss_type!r}; expected 'l1' or 'l1_lpips'.")
    print(
        f"Loss: {loss_type}"
        + (f" (LPIPS weight: {perceptual_weight:g})" if perceptual_model is not None else "")
        + (f" + {haze_aux_weight:g} × haze auxiliary L1" if haze_aux_weight > 0 else "")
        + (
            f" + {physical_consistency_weight:g} × atmospheric-scattering L1"
            if physical_consistency_weight > 0 else ""
        )
    )
    start_epoch = 1
    best_psnr = -float("inf")
    epochs_without_improvement = 0
    early_stopping = training.get("early_stopping", {})
    patience = int(early_stopping.get("patience", 0))
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint.get("scaler", {}))
        start_epoch = int(checkpoint["epoch"]) + 1
        best_psnr = float(checkpoint.get("best_psnr", best_psnr))
        epochs_without_improvement = int(checkpoint.get("epochs_without_improvement", 0))
        print(f"Resumed at epoch {start_epoch} from {args.resume}")

    writer = SummaryWriter(log_dir=str(run_dir / "tensorboard"))
    csv_path = run_dir / "metrics.csv"
    csv_fields = ["epoch", "train_l1", "train_ssim", "val_l1", "val_ssim", "val_psnr", "learning_rate"]
    if perceptual_model is not None:
        csv_fields[2:2] = ["train_lpips", "train_total_loss"]
        csv_fields[6:6] = ["val_lpips"]
    if haze_aux_weight > 0:
        csv_fields[2:2] = ["train_haze_aux", "train_total_loss"]
    if physical_consistency_weight > 0:
        csv_fields[2:2] = ["train_physical_consistency", "train_total_loss"]
    csv_fields = list(dict.fromkeys(csv_fields))
    log_timing = bool(training.get("log_timing", False))
    if log_timing:
        csv_fields += ["train_seconds", "validation_seconds", "epoch_seconds", "train_samples_per_second"]
    write_csv_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with (run_dir / "metrics.jsonl").open("a", encoding="utf-8") as log_file:
        with csv_path.open("a", encoding="utf-8", newline="") as csv_stream:
            csv_writer = csv.DictWriter(csv_stream, fieldnames=csv_fields)
            if write_csv_header:
                csv_writer.writeheader()
            for epoch in range(start_epoch, epochs + 1):
                epoch_started = time.perf_counter()
                if batch_sampler is not None:
                    batch_sampler.set_epoch(epoch - 1)
                model.train()
                # Accumulate on device; one transfer per epoch avoids scalar CUDA
                # synchronization after every loss and SSIM calculation.
                totals = torch.zeros(6, device=device, dtype=torch.float64)
                sample_count = 0
                progress = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
                for step, batch in enumerate(progress, start=1):
                    hazy, clear, depth = _unpack_batch(batch)
                    hazy, clear = hazy.to(device, non_blocking=True), clear.to(device, non_blocking=True)
                    depth = depth.to(device, non_blocking=True) if depth is not None else None
                    optimizer.zero_grad(set_to_none=True)
                    with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                        if haze_aux_weight > 0 or physical_consistency_weight > 0:
                            prediction, auxiliary = model(hazy, return_aux=True)
                        else:
                            prediction = _forward(model, hazy, depth)
                            auxiliary = {}
                    prediction_float, clear_float = prediction.float(), clear.float()
                    l1_loss = criterion(prediction_float, clear_float)
                    haze_aux_loss = torch.zeros((), device=device)
                    physical_consistency_loss = torch.zeros((), device=device)
                    if haze_aux_weight > 0:
                        if "haze_map" not in auxiliary:
                            raise RuntimeError("M2/M3/M5 must return a haze_map for auxiliary supervision.")
                        target_haze = _haze_proxy(hazy, clear, auxiliary["haze_map"].shape[-2:])
                        haze_aux_loss = criterion(auxiliary["haze_map"].float(), target_haze)
                    if physical_consistency_weight > 0:
                        if "transmission" not in auxiliary or "airlight" not in auxiliary:
                            raise RuntimeError("M6 must return transmission and airlight for its consistency loss.")
                        transmission = auxiliary["transmission"].float()
                        airlight = auxiliary["airlight"].float()
                        clean_for_physics = prediction_float.clamp(0.0, 1.0)
                        hazy_reconstruction = transmission * clean_for_physics + (1.0 - transmission) * airlight
                        physical_consistency_loss = criterion(hazy_reconstruction, hazy.float())
                    total_loss = (
                        l1_loss
                        + haze_aux_weight * haze_aux_loss
                        + physical_consistency_weight * physical_consistency_loss
                    )
                    if perceptual_model is not None:
                        lpips_loss = perceptual_model(
                            prediction_float.clamp(0, 1), clear_float, normalize=True
                        ).mean()
                        total_loss = total_loss + perceptual_weight * lpips_loss
                    else:
                        lpips_loss = None
                    scaler.scale(total_loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    batch_count = hazy.shape[0]
                    totals[0] += total_loss.detach().double() * batch_count
                    totals[1] += l1_loss.detach().double() * batch_count
                    if haze_aux_weight > 0:
                        totals[3] += haze_aux_loss.detach().double() * batch_count
                    if physical_consistency_weight > 0:
                        totals[4] += physical_consistency_loss.detach().double() * batch_count
                    if lpips_loss is not None:
                        totals[2] += lpips_loss.detach().double() * batch_count
                    totals[5] += ssim_tensor(prediction.detach().clamp(0, 1), clear).sum().double()
                    sample_count += batch_count
                    if step % max(1, int(training.get("progress_interval", 1))) == 0:
                        progress.set_postfix(
                            loss=f"{total_loss.detach().item():.4f}",
                            lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                        )

                loss_sum, l1_sum, lpips_sum, haze_aux_sum, physical_consistency_sum, ssim_sum = totals.cpu().tolist()
                train_seconds = time.perf_counter() - epoch_started
                scheduler.step()
                val_metrics = validate(
                    model, val_loader, device, use_amp, perceptual_model,
                    validation_tile_size, validation_tile_overlap,
                )
                train_loss = loss_sum / max(sample_count, 1)
                train_l1 = l1_sum / max(sample_count, 1)
                train_lpips = lpips_sum / max(sample_count, 1) if perceptual_model is not None else None
                train_haze_aux = haze_aux_sum / max(sample_count, 1) if haze_aux_weight > 0 else None
                train_physical_consistency = (
                    physical_consistency_sum / max(sample_count, 1)
                    if physical_consistency_weight > 0 else None
                )
                train_ssim = ssim_sum / max(sample_count, 1)
                record = {
                    "epoch": epoch,
                    "train_l1": train_l1,
                    "train_ssim": train_ssim,
                    "val_l1": val_metrics["l1"],
                    "val_ssim": val_metrics["ssim"],
                    "val_psnr": val_metrics["psnr"],
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                }
                if log_timing:
                    epoch_seconds = time.perf_counter() - epoch_started
                    record.update(
                        train_seconds=train_seconds,
                        validation_seconds=epoch_seconds - train_seconds,
                        epoch_seconds=epoch_seconds,
                        train_samples_per_second=sample_count / max(train_seconds, 1e-6),
                    )
                if perceptual_model is not None:
                    record.update(
                        train_lpips=train_lpips,
                        train_total_loss=train_loss,
                        val_lpips=val_metrics["lpips"],
                    )
                if train_haze_aux is not None:
                    record.update(train_haze_aux=train_haze_aux, train_total_loss=train_loss)
                if train_physical_consistency is not None:
                    record.update(
                        train_physical_consistency=train_physical_consistency,
                        train_total_loss=train_loss,
                    )
                log_file.write(json.dumps(record) + "\n")
                log_file.flush()
                csv_writer.writerow(record)
                csv_stream.flush()
                writer.add_scalar("train/l1", train_l1, epoch)
                writer.add_scalar("train/total_loss", train_loss, epoch)
                writer.add_scalar("train/ssim", train_ssim, epoch)
                writer.add_scalar("validation/l1", val_metrics["l1"], epoch)
                writer.add_scalar("validation/ssim", val_metrics["ssim"], epoch)
                writer.add_scalar("validation/psnr", val_metrics["psnr"], epoch)
                if train_lpips is not None:
                    writer.add_scalar("train/lpips", train_lpips, epoch)
                    writer.add_scalar("validation/lpips", val_metrics["lpips"], epoch)
                if train_haze_aux is not None:
                    writer.add_scalar("train/haze_aux", train_haze_aux, epoch)
                if train_physical_consistency is not None:
                    writer.add_scalar("train/physical_consistency", train_physical_consistency, epoch)
                print(
                    f"epoch={epoch} train_l1={train_l1:.5f} "
                    + (f"train_lpips={train_lpips:.5f} " if train_lpips is not None else "")
                    + (f"train_haze_aux={train_haze_aux:.5f} " if train_haze_aux is not None else "")
                    + (
                        f"train_physical_consistency={train_physical_consistency:.5f} "
                        if train_physical_consistency is not None else ""
                    )
                    + f"train_ssim={train_ssim:.5f} val_l1={val_metrics['l1']:.5f} "
                    + (f"val_lpips={val_metrics['lpips']:.5f} " if val_metrics["lpips"] is not None else "")
                    + f"val_ssim={val_metrics['ssim']:.5f} "
                    + f"val_psnr={val_metrics['psnr']:.4f}"
                    + (f" epoch_seconds={record['epoch_seconds']:.1f}" if log_timing else "")
                )

                improved = val_metrics["psnr"] > best_psnr
                if improved:
                    best_psnr = val_metrics["psnr"]
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1

                state = {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(),
                    "best_psnr": best_psnr,
                    "epochs_without_improvement": epochs_without_improvement,
                    "config": config,
                }
                torch.save(state, checkpoint_dir / "last.pth")
                if improved:
                    torch.save(state, checkpoint_dir / "best.pth")
                if patience > 0 and epochs_without_improvement >= patience:
                    print(
                        f"Early stopping: validation PSNR did not improve for {patience} epochs. "
                        f"Best validation PSNR: {best_psnr:.4f} dB."
                    )
                    break
                if args.stop_file and args.stop_file.exists():
                    print("Stop request received; checkpoint saved at the end of this epoch.")
                    break
    writer.close()
    print(f"Training finished. Best validation PSNR: {best_psnr:.4f} dB")


if __name__ == "__main__":
    main()
