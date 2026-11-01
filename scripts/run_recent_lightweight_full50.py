"""Run SLRNet and WLD-Net sequentially for 50 epochs each."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch


EXPERIMENTS = (
    ("SLRNet", "configs/experiment/slrnet_recent_full50.yaml", "runs/slrnet_recent_full50"),
    ("WLD-Net", "configs/experiment/wldnet_recent_full50.yaml", "runs/wldnet_recent_full50"),
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_status(path: Path, status: dict[str, object]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--python-executable", type=Path, required=True)
    args = parser.parse_args()

    root = args.project_root.resolve()
    python = args.python_executable.resolve()
    background = root / "runs/recent_lightweight_full50_bs8/background"
    background.mkdir(parents=True, exist_ok=True)
    status_path = background / "suite_status.json"
    suite_started = time.time()
    models: list[dict[str, object]] = [
        {"name": name, "run_dir": run_dir, "state": "queued"}
        for name, _, run_dir in EXPERIMENTS
    ]
    status: dict[str, object] = {
        "state": "running",
        "started_at": utc_now(),
        "current_model": None,
        "models": models,
    }
    write_status(status_path, status)

    environment = os.environ.copy()
    source_path = str(root / "src")
    old_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        source_path + os.pathsep + old_pythonpath if old_pythonpath else source_path
    )

    for index, (name, config_rel, run_rel) in enumerate(EXPERIMENTS):
        config_path = root / config_rel
        run_dir = root / run_rel
        run_dir.mkdir(parents=True, exist_ok=True)
        model_record = models[index]
        model_record.update(state="running", started_at=utc_now())
        status["current_model"] = name
        write_status(status_path, status)

        stdout_path = run_dir / "background/train.stdout.log"
        stderr_path = run_dir / "background/train.stderr.log"
        (run_dir / "background").mkdir(parents=True, exist_ok=True)
        # Fail safely instead of overwriting an earlier training run.
        checkpoint_dir = run_dir / "checkpoints"
        metrics_path = run_dir / "metrics.csv"
        last_checkpoint = checkpoint_dir / "last.pth"
        resume_path: Path | None = None
        if last_checkpoint.exists():
            saved_state = torch.load(last_checkpoint, map_location="cpu", weights_only=False)
            saved_epoch = int(saved_state.get("epoch", 0))
            if saved_epoch >= 50:
                model_record.update(
                    state="completed_existing_run",
                    finished_at=utc_now(),
                    last_epoch=saved_epoch,
                )
                write_status(status_path, status)
                continue
            resume_path = last_checkpoint
            model_record["resume_from_epoch"] = saved_epoch
        elif metrics_path.exists():
            model_record.update(state="skipped_existing_run", finished_at=utc_now())
            model_record["error"] = "Metrics exist but no last checkpoint is available; refusing to overwrite."
            write_status(status_path, status)
            continue

        command = [
            str(python), "-u", "-m", "dehazing.train",
            "--config", str(config_path),
            "--output-dir", str(run_dir),
            "--epochs", "50",
        ]
        if resume_path is not None:
            command.extend(["--resume", str(resume_path)])
        started = time.time()
        log_mode = "a" if resume_path is not None else "w"
        with stdout_path.open(log_mode, encoding="utf-8", buffering=1) as stdout, \
                stderr_path.open(log_mode, encoding="utf-8", buffering=1) as stderr:
            process = subprocess.Popen(command, cwd=root, env=environment, stdout=stdout, stderr=stderr)
            (run_dir / "background/train.pid").write_text(str(process.pid), encoding="ascii")
            model_record["pid"] = process.pid
            model_record["command"] = command
            write_status(status_path, status)
            exit_code = process.wait()

        model_record.update(
            state="completed" if exit_code == 0 else "failed",
            finished_at=utc_now(),
            elapsed_seconds=round(time.time() - started, 2),
            exit_code=exit_code,
        )
        write_status(status_path, status)

    errors = [
        record for record in models
        if record.get("state") not in {"completed", "completed_existing_run"}
    ]
    status.update(
        state="completed" if not errors else "completed_with_errors",
        current_model=None,
        finished_at=utc_now(),
        elapsed_seconds=round(time.time() - suite_started, 2),
    )
    write_status(status_path, status)
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
