"""Run the M1, M2, and M3 screening experiments sequentially on one GPU."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path


EXPERIMENTS = (
    ("M1", "configs/experiment/dehazeformer_t_m1_pilot.yaml", "dehazeformer_t_m1_pilot12"),
    ("M2", "configs/experiment/dehazeformer_t_m2_pilot.yaml", "dehazeformer_t_m2_pilot12"),
    ("M3", "configs/experiment/dehazeformer_t_m3_pilot.yaml", "dehazeformer_t_m3_pilot12"),
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary_path.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run M1, M2, and M3 sequentially.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--python-executable", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=12)
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    python_executable = args.python_executable.resolve()
    suite_dir = project_root / "runs" / "dehazeformer_t_m1_m3_pilot12"
    suite_background = suite_dir / "background"
    suite_background.mkdir(parents=True, exist_ok=True)
    status_path = suite_background / "suite_status.json"
    suite_stop_path = suite_background / "stop_after_current_model"
    completed: list[str] = []
    started_at = datetime.now().astimezone().isoformat()

    env = os.environ.copy()
    src_path = str(project_root / "src")
    previous_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = src_path + (os.pathsep + previous_pythonpath if previous_pythonpath else "")

    try:
        for model_id, config_relative, run_name in EXPERIMENTS:
            config_path = project_root / config_relative
            run_dir = project_root / "runs" / run_name
            if not config_path.is_file():
                raise FileNotFoundError(f"Experiment config not found: {config_path}")
            if run_dir.exists() and any(run_dir.iterdir()):
                raise FileExistsError(
                    f"Refusing to overwrite existing {model_id} output: {run_dir}. "
                    "Choose new run names or move the previous run first."
                )

        for index, (model_id, config_relative, run_name) in enumerate(EXPERIMENTS, start=1):
            if suite_stop_path.exists():
                _write_json(status_path, {
                    "status": "stopped_between_models",
                    "completed_models": completed,
                    "next_model": model_id,
                    "started_at": started_at,
                    "updated_at": datetime.now().astimezone().isoformat(),
                })
                print(f"Suite stop requested; skipping {model_id} and remaining experiments.", flush=True)
                return

            config_path = project_root / config_relative
            run_dir = project_root / "runs" / run_name

            run_background = run_dir / "background"
            run_background.mkdir(parents=True, exist_ok=True)
            stdout_path = run_background / "train.stdout.log"
            stderr_path = run_background / "train.stderr.log"
            pid_path = run_background / "train.pid"
            stop_path = run_background / "train.stop"
            if stop_path.exists():
                stop_path.unlink()

            command = [
                str(python_executable), "-m", "dehazing.train",
                "--config", str(config_path),
                "--output-dir", str(run_dir),
                "--stop-file", str(stop_path),
            ]
            if args.epochs > 0:
                command.extend(("--epochs", str(args.epochs)))

            _write_json(status_path, {
                "status": "running",
                "current_model": model_id,
                "current_run": run_name,
                "experiment_index": index,
                "experiment_count": len(EXPERIMENTS),
                "completed_models": completed,
                "started_at": started_at,
                "updated_at": datetime.now().astimezone().isoformat(),
            })
            print(f"[{index}/{len(EXPERIMENTS)}] Starting {model_id}: {run_name}", flush=True)

            launch_info = {
                "model": model_id,
                "python": str(python_executable),
                "config": str(config_path),
                "run_dir": str(run_dir),
                "started_at": datetime.now().astimezone().isoformat(),
                "arguments": command,
            }
            with stdout_path.open("w", encoding="utf-8") as stdout_file, stderr_path.open(
                "w", encoding="utf-8"
            ) as stderr_file:
                process = subprocess.Popen(
                    command,
                    cwd=project_root,
                    env=env,
                    stdout=stdout_file,
                    stderr=stderr_file,
                )
                pid_path.write_text(str(process.pid), encoding="ascii")
                launch_info["pid"] = process.pid
                _write_json(run_background / "launch.json", launch_info)
                return_code = process.wait()

            if pid_path.exists():
                pid_path.unlink()
            if return_code != 0:
                _write_json(status_path, {
                    "status": "failed",
                    "failed_model": model_id,
                    "return_code": return_code,
                    "completed_models": completed,
                    "updated_at": datetime.now().astimezone().isoformat(),
                })
                print(f"{model_id} failed with exit code {return_code}; suite stopped.", flush=True)
                raise SystemExit(return_code)

            completed.append(model_id)
            _write_json(status_path, {
                "status": "running",
                "current_model": None,
                "completed_models": completed,
                "next_model": EXPERIMENTS[index][0] if index < len(EXPERIMENTS) else None,
                "started_at": started_at,
                "updated_at": datetime.now().astimezone().isoformat(),
            })
            print(f"Completed {model_id}.", flush=True)

        _write_json(status_path, {
            "status": "completed",
            "completed_models": completed,
            "started_at": started_at,
            "finished_at": datetime.now().astimezone().isoformat(),
        })
        print("All M1/M2/M3 pilot experiments completed.", flush=True)
    except SystemExit:
        raise
    except Exception as error:
        _write_json(status_path, {
            "status": "failed",
            "error": str(error),
            "completed_models": completed,
            "updated_at": datetime.now().astimezone().isoformat(),
        })
        print(f"Ablation suite failed: {error}", flush=True)
        raise
    finally:
        suite_pid_path = suite_background / "suite.pid"
        if suite_pid_path.exists() and suite_pid_path.read_text(encoding="ascii").strip() == str(os.getpid()):
            suite_pid_path.unlink()


if __name__ == "__main__":
    main()
