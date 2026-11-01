"""Prepare train-only depth/cache, tune throughput, and run M6 + MCBM for 50 epochs."""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
CONFIG = PROJECT_ROOT / "configs/experiment/dehazeformer_t_m6_mcbm_full50.yaml"
RUN_DIR = PROJECT_ROOT / "runs/dehazeformer_t_m6_mcbm_full50"


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def training_dataset(config):
    from torch.utils.data import ConcatDataset
    from dehazing.data.paired_dataset import PairedHazeDataset

    data, training = config["data"], config["training"]
    return ConcatDataset([
        PairedHazeDataset(
            data["root"], source["dataset"], source.get("split", "train"),
            crop_size=training["crop_size"], horizontal_flip=True,
            depth_root=data["depth_root"], return_depth=False,
            mcbm_haze=training["augmentation"]["mcbm_haze"],
            image_cache_root=data["image_cache_root"],
        ) for source in data["train"]["datasets"]
    ])


def probe(batch_size: int, workers: int, output: Path) -> None:
    import torch
    from torch.utils.data import DataLoader
    from dehazing.metrics.image_metrics import ssim_tensor
    from dehazing.models import build_model
    from dehazing.train import _worker_seed
    from dehazing.utils.reproducibility import seed_everything

    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    result = {"batch_size": batch_size, "num_workers": workers, "status": "failed"}
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU is required for this experiment.")
        torch.set_num_threads(config["training"]["cpu_threads"])
        seed_everything(config["seed"])
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = bool(config["training"].get("deterministic", True))
        kwargs = {"num_workers": workers, "pin_memory": True,
                  "persistent_workers": workers > 0, "worker_init_fn": _worker_seed}
        if workers:
            kwargs["prefetch_factor"] = 2
        loader = DataLoader(training_dataset(config), batch_size=batch_size, shuffle=True,
                            drop_last=True, generator=torch.Generator().manual_seed(42), **kwargs)
        iterator = iter(loader)
        device = torch.device("cuda")
        model = build_model("dehazeformer_t_m6").to(device).train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4, fused=True)
        scaler = torch.amp.GradScaler("cuda")
        loss_fn = torch.nn.L1Loss()
        warmup, iterations = 4, 12
        loss = None
        checksum = torch.zeros((), device=device)
        torch.cuda.reset_peak_memory_stats()
        started = None
        for step in range(warmup + iterations):
            if step == warmup:
                torch.cuda.synchronize()
                started = time.perf_counter()
            hazy, clear, _ = next(iterator)
            hazy, clear = hazy.to(device, non_blocking=True), clear.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                prediction, aux = model(hazy, return_aux=True)
            reconstructed = aux["transmission"].float() * prediction.float().clamp(0, 1)
            reconstructed += (1 - aux["transmission"].float()) * aux["airlight"].float()
            loss = loss_fn(prediction.float(), clear) + 0.05 * loss_fn(reconstructed, hazy)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            checksum += ssim_tensor(prediction.detach().clamp(0, 1), clear).sum()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        if not torch.isfinite(loss).item() or not torch.isfinite(checksum).item():
            raise RuntimeError("Non-finite training loss/SSIM during throughput probe.")
        result.update(
            status="ok", samples_per_second=iterations * batch_size / elapsed,
            measured_steps=iterations, measured_seconds=elapsed,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            last_loss=float(loss.detach().item()), gpu=torch.cuda.get_device_name(),
        )
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    write_json(output, result)
    print(json.dumps(result), flush=True)


def validate_cached_samples(config) -> None:
    import random
    import numpy as np
    import torch
    from dehazing.data.paired_dataset import PairedHazeDataset

    cached = training_dataset(config)
    for dataset in cached.datasets:
        raw = PairedHazeDataset(
            config["data"]["root"], dataset.dataset_name, "train", crop_size=256,
            horizontal_flip=True, depth_root=config["data"]["depth_root"], return_depth=False,
            mcbm_haze=config["training"]["augmentation"]["mcbm_haze"],
        )
        for index in (0, len(dataset.pairs), len(dataset) - 1):
            random.seed(123); np.random.seed(123)
            reference = raw[index]
            random.seed(123); np.random.seed(123)
            accelerated = dataset[index]
            if not all(torch.equal(a, b) for a, b in zip(reference[:2], accelerated[:2])):
                raise AssertionError(f"Cache changed original/MCBM pixels for {dataset.dataset_name}, {index}")
    print("Verified: cached original and MCBM crops match uncached pixels on all three domains.", flush=True)


def run(resume: bool) -> None:
    run_started = time.time()
    os.chdir(PROJECT_ROOT)
    background = RUN_DIR / "background"
    background.mkdir(parents=True, exist_ok=True)
    status_path = background / "pipeline_status.json"
    effective_path = RUN_DIR / "effective_config.yaml"
    stop_path = background / "train.stop"
    started_at = datetime.now().astimezone().isoformat()
    state = {"started_at": started_at, "pipeline_pid": os.getpid()}

    def status(phase, **extra):
        state.update(status=phase, updated_at=datetime.now().astimezone().isoformat(), **extra)
        write_json(status_path, state)
        print(f"[{phase}] {extra}", flush=True)
        if phase not in ("completed", "failed", "stopped") and stop_path.exists():
            raise InterruptedError("Stop requested during preparation.")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    env.update(PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
    try:
        if (RUN_DIR / "checkpoints/last.pth").exists() and not resume:
            raise FileExistsError("Existing checkpoint found. Use -Resume to continue this run.")
        if resume and not (RUN_DIR / "checkpoints/last.pth").is_file():
            raise FileNotFoundError("No last.pth is available for resume.")
        config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
        for dataset in ("reside", "nhhaze2", "hdnhhaze"):
            status("preparing_depth", dataset=dataset)
            subprocess.run([
                sys.executable, "-u", "scripts/generate_depth_maps.py",
                "--dataset", dataset, "--split", "train",
            ], cwd=PROJECT_ROOT, env=env, check=True)
        status("preparing_cache")
        from prepare_training_cache import prepare
        counts = prepare(CONFIG)
        validate_cached_samples(config)
        write_json(RUN_DIR / "data_preparation.json", counts)

        if resume:
            config = yaml.safe_load(effective_path.read_text(encoding="utf-8"))
        else:
            import torch
            free, total = torch.cuda.mem_get_info()
            memory_budget = min(int(total * 0.85), int(free * 0.90))
            results = []

            def measure(batch, workers):
                status("tuning", batch_size=batch, num_workers=workers)
                output = RUN_DIR / "tuning" / f"batch{batch}_workers{workers}.json"
                output.parent.mkdir(parents=True, exist_ok=True)
                with output.with_suffix(".log").open("w", encoding="utf-8") as log:
                    completed = subprocess.run([
                        sys.executable, "-u", str(Path(__file__).resolve()),
                        "--probe-batch", str(batch), "--probe-workers", str(workers),
                        "--probe-output", str(output),
                    ], cwd=PROJECT_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                if completed.returncode != 0 or not output.exists():
                    result = {"status": "failed", "batch_size": batch, "num_workers": workers,
                              "error": f"Probe exit code {completed.returncode}; see {output.with_suffix('.log')}"}
                else:
                    result = json.loads(output.read_text(encoding="utf-8"))
                result["within_memory_budget"] = (
                    result.get("status") == "ok" and result["peak_reserved_bytes"] <= memory_budget
                )
                results.append(result)
                print(json.dumps(result), flush=True)
                return result

            for batch in (8, 16, 24, 32):
                if results and results[-1].get("status") == "ok":
                    previous = results[-1]
                    projected = previous["peak_allocated_bytes"] * batch / previous["batch_size"]
                    if projected > memory_budget:
                        break
                result = measure(batch, 4)
                if not result["within_memory_budget"]:
                    break
            eligible = [r for r in results if r["within_memory_budget"]]
            if not eligible:
                raise RuntimeError("No stable batch fits the dedicated-VRAM safety budget; see tuning logs.")
            best = max(eligible, key=lambda r: r["samples_per_second"])
            for workers in (0, 2):
                measure(best["batch_size"], workers)
            best = max((r for r in results if r["within_memory_budget"]), key=lambda r: r["samples_per_second"])
            config = copy.deepcopy(config)
            config["training"]["batch_size"] = best["batch_size"]
            config["data"]["num_workers"] = best["num_workers"]
            effective_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
            write_json(RUN_DIR / "tuning/results.json", {
                "memory_budget_bytes": memory_budget, "selection": best, "candidates": results,
                "scope": "real cached/MCBM batches, forward + L1/physics + backward + optimizer + SSIM; excludes startup and validation",
            })

        command = [sys.executable, "-u", "-m", "dehazing.train", "--config", str(effective_path),
                   "--output-dir", str(RUN_DIR), "--stop-file", str(stop_path)]
        if resume:
            command += ["--resume", str(RUN_DIR / "checkpoints/last.pth")]
        status("starting_training", batch_size=config["training"]["batch_size"],
               num_workers=config["data"]["num_workers"], epochs=50, **counts)
        with (background / "train.stdout.log").open("a", encoding="utf-8") as stdout, \
             (background / "train.stderr.log").open("a", encoding="utf-8") as stderr:
            process = subprocess.Popen(command, cwd=PROJECT_ROOT, env=env, stdout=stdout, stderr=stderr)
            (background / "train.pid").write_text(str(process.pid), encoding="ascii")
            write_json(background / "launch.json", {
                "pid": process.pid, "python": sys.executable, "config": str(effective_path),
                "run_dir": str(RUN_DIR), "started_at": datetime.now().astimezone().isoformat(),
                "arguments": command[1:],
            })
            # Child is now responsible for handling the epoch-boundary stop file.
            state.update(status="training", train_pid=process.pid)
            write_json(status_path, state)
            code = process.wait()
        (background / "train.pid").unlink(missing_ok=True)
        if code:
            raise RuntimeError(f"Training failed (exit {code}); inspect train.stderr.log.")
        status("stopped" if stop_path.exists() else "completed", elapsed_seconds=time.time() - run_started)
    except InterruptedError as exc:
        status("stopped", error=str(exc))
    except BaseException as exc:
        status("failed", error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--probe-batch", type=int)
    parser.add_argument("--probe-workers", type=int, default=4)
    parser.add_argument("--probe-output", type=Path)
    args = parser.parse_args()
    os.chdir(PROJECT_ROOT)
    run_started = time.time()
    if args.probe_batch:
        if args.probe_output is None:
            parser.error("--probe-output is required with --probe-batch")
        probe(args.probe_batch, args.probe_workers, args.probe_output)
    else:
        run(args.resume)
