from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
METRICS = ROOT / "runs/dehazeformer_t_mcbm/metrics.csv"
NAVY = "#123B62"
BLUE = "#2878B5"
ORANGE = "#E7862A"
GRID = "#DCE3EA"
TEXT = "#263746"


def main() -> None:
    with METRICS.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key, value in tuple(row.items()):
            row[key] = float(value) if value not in (None, "") else np.nan

    best = max(rows, key=lambda row: row["val_psnr"])
    best_epoch = int(best["epoch"])
    epochs = [row["epoch"] for row in rows]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.labelcolor": TEXT,
            "text.color": TEXT,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.4))
    fig.subplots_adjust(left=0.075, right=0.98, top=0.77, bottom=0.14, wspace=0.24)

    header = fig.add_axes([0, 0.89, 1, 0.11])
    header.set_facecolor(NAVY)
    header.set_xticks([])
    header.set_yticks([])
    for spine in header.spines.values():
        spine.set_visible(False)
    header.text(
        0.025,
        0.5,
        "Quá trình train: DehazeFormer-T + MCBM trên RESIDE",
        color="white",
        fontsize=17,
        weight="medium",
        va="center",
        ha="left",
    )
    fig.text(
        0.5,
        0.84,
        f"50 epoch  |  Best checkpoint theo validation PSNR: epoch {best_epoch} ({best['val_psnr']:.3f} dB)",
        ha="center",
        va="center",
        fontsize=9,
        color=TEXT,
    )

    panels = [
        (axes[0], "L1 / MAE", "train_l1", "val_l1", "L1 theo epoch"),
        (axes[1], "SSIM", "train_ssim", "val_ssim", "SSIM theo epoch"),
    ]
    for ax, ylabel, train_key, val_key, title in panels:
        train = [row[train_key] for row in rows]
        val = [row[val_key] for row in rows]
        ax.plot(epochs, train, color=BLUE, linewidth=1.8, label=f"Train {ylabel.split()[0]}")
        ax.plot(epochs, val, color=ORANGE, linewidth=1.8, linestyle="--", label=f"Val {ylabel.split()[0]}")
        ax.axvline(best_epoch, color="#657786", linewidth=1.0, linestyle=":", alpha=0.85)
        ax.scatter(
            [best_epoch],
            [best[val_key]],
            marker="D",
            s=34,
            color=ORANGE,
            edgecolor="white",
            linewidth=0.6,
            zorder=4,
            label="Best val PSNR",
        )
        ax.set_title(title, color=NAVY, fontsize=11, weight="semibold", loc="left", pad=8)
        ax.set_xlabel("Epoch", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_xlim(0, 50)
        ax.set_xticks(np.arange(0, 51, 5))
        ax.grid(True, color=GRID, linewidth=0.7)
        ax.set_axisbelow(True)
        ax.tick_params(colors=TEXT, labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("#B8C4CF")
            spine.set_linewidth(0.8)
        ax.legend(loc="best", frameon=False, fontsize=8.5, ncol=1)

    axes[1].set_ylim(0.50, 1.0)
    axes[1].set_yticks(np.arange(0.50, 1.001, 0.05))
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "dehazeformer_t_mcbm_curves.png", dpi=300, facecolor="white", bbox_inches="tight")
    fig.savefig(OUT / "dehazeformer_t_mcbm_curves.svg", facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved chart; {len(rows)} epochs; best validation PSNR epoch={best_epoch}")


if __name__ == "__main__":
    main()
