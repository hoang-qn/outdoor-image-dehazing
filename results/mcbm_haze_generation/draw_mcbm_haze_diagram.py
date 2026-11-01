from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent

CLEAR_PATH = ROOT / "data/RESIDE-6K-Outdoor/train/clear/1.jpg"
INVERSE_DEPTH_PATH = ROOT / "data/depth_maps/RESIDE-6K-Outdoor/train/hazy/1.png"
DENSITY_PATH = ROOT / "data/MCBM/0.png"

# Fixed values make this a reproducible visual example. Training samples draw
# beta, density scale, density map, and airlight randomly from configured ranges.
BETA = 1.60
DENSITY_SCALE = 0.90
DEPTH_SCALE = 2.0
AIRLIGHT = 0.92

NAVY = "#13263f"
INK = "#1e3047"
MUTED = "#61738a"
BLUE = "#3578c8"
TEAL = "#149b91"
GOLD = "#e7a83e"
CORAL = "#df715d"
PAPER = "#f4f7fb"
WHITE = "#ffffff"
EDGE = "#dce5ef"


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0


def normalize01(values: np.ndarray) -> np.ndarray:
    low, high = float(values.min()), float(values.max())
    if high <= low:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - low) / (high - low)).astype(np.float32)


def main() -> None:
    clear = load_rgb(CLEAR_PATH)
    h, w = clear.shape[:2]

    with Image.open(INVERSE_DEPTH_PATH) as im:
        inverse_depth_image = im.convert("L").resize((w, h), Image.Resampling.BILINEAR)
        inverse_depth = np.asarray(inverse_depth_image, dtype=np.float32) / 255.0
    # The project stores a normalized relative inverse-depth map from MiDaS;
    # MCBMHazeAugment inverts it to create the depth proxy used in scattering.
    depth = 1.0 - np.clip(inverse_depth, 0.0, 1.0)

    with Image.open(DENSITY_PATH) as im:
        density_image = im.convert("L").resize((w, h), Image.Resampling.BILINEAR)
        density = np.asarray(density_image, dtype=np.float32) / 255.0
    density = normalize01(density)

    beta_field = BETA + DENSITY_SCALE * density
    transmission = np.exp(-DEPTH_SCALE * depth * beta_field)
    hazy = np.clip(
        clear * transmission[..., None]
        + AIRLIGHT * (1.0 - transmission[..., None]),
        0.0,
        1.0,
    )

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.unicode_minus": False,
            "svg.fonttype": "none",
        }
    )
    fig = plt.figure(figsize=(16, 9.5), facecolor=PAPER)
    fig.text(
        0.5,
        0.955,
        "TẠO ẢNH HAZE MỚI BẰNG DENSITY MAP VÀ DEPTH MAP",
        ha="center",
        va="center",
        fontsize=20,
        fontweight="bold",
        color=NAVY,
    )
    fig.text(
        0.5,
        0.918,
        "Minh họa theo pipeline MCBM trong project · dùng ảnh sạch RESIDE và các map đã căn chỉnh cùng kích thước",
        ha="center",
        va="center",
        fontsize=10.5,
        color=MUTED,
    )

    def card(x: float, y: float, w0: float, h0: float, title: str, subtitle: str, accent: str) -> None:
        p = FancyBboxPatch(
            (x, y),
            w0,
            h0,
            boxstyle="round,pad=0.008,rounding_size=0.018",
            transform=fig.transFigure,
            linewidth=1.0,
            edgecolor=EDGE,
            facecolor=WHITE,
            zorder=-2,
        )
        fig.add_artist(p)
        fig.add_artist(
            FancyBboxPatch(
                (x, y + h0 - 0.012),
                w0,
                0.012,
                boxstyle="round,pad=0.002,rounding_size=0.008",
                transform=fig.transFigure,
                linewidth=0,
                facecolor=accent,
                zorder=-1,
            )
        )
        fig.text(x + w0 / 2, y + h0 - 0.045, title, ha="center", va="center", fontsize=12, fontweight="bold", color=INK)
        fig.text(x + w0 / 2, y + 0.018, subtitle, ha="center", va="bottom", fontsize=8.2, color=MUTED)

    def image_axis(x: float, y: float, size: float, image: np.ndarray, cmap: str | None = None) -> None:
        ax = fig.add_axes([x, y, size, size])
        ax.imshow(image, cmap=cmap, vmin=0 if cmap else None, vmax=1 if cmap else None)
        ax.set_axis_off()
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(EDGE)
            spine.set_linewidth(0.7)

    card_y, card_h, card_w = 0.595, 0.285, 0.245
    image_size = 0.18
    input_cards = [
        (0.045, "Ảnh sạch  J", "Radiance / ảnh GT", BLUE, clear, None),
        (0.3775, "Depth proxy  d(x)", "d = 1 − MiDaS inverse depth", TEAL, depth, "gray"),
        (0.710, "Density map  D(x)", "Chuẩn hóa về [0, 1]", GOLD, density, "gray"),
    ]
    for x, title, subtitle, color, image, cmap in input_cards:
        card(x, card_y, card_w, card_h, title, subtitle, color)
        image_axis(x + (card_w - image_size) / 2, card_y + 0.040, image_size, image, cmap)

    # Central model block shows the actual equations and random parameters.
    formula_x, formula_y, formula_w, formula_h = 0.105, 0.345, 0.79, 0.175
    formula_box = FancyBboxPatch(
        (formula_x, formula_y),
        formula_w,
        formula_h,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        transform=fig.transFigure,
        linewidth=1.2,
        edgecolor="#cbd9e8",
        facecolor="#eaf1f9",
        zorder=-2,
    )
    fig.add_artist(formula_box)
    fig.text(0.5, formula_y + formula_h - 0.034, "MÔ HÌNH TÁN XẠ MCBM", ha="center", va="center", fontsize=11, fontweight="bold", color=NAVY)
    fig.text(
        0.30,
        formula_y + 0.077,
        r"$\beta_{field}(x)=\beta+\alpha D(x)$",
        ha="center",
        va="center",
        fontsize=14,
        color=INK,
    )
    fig.text(
        0.30,
        formula_y + 0.036,
        r"$T(x)=\exp[-s\,d(x)\,\beta_{field}(x)]$",
        ha="center",
        va="center",
        fontsize=13,
        color=INK,
    )
    fig.add_artist(
        FancyArrowPatch(
            (0.505, formula_y + 0.076),
            (0.565, formula_y + 0.076),
            transform=fig.transFigure,
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=1.7,
            color=BLUE,
        )
    )
    fig.text(
        0.725,
        formula_y + 0.065,
        r"$I_{hazy}(x)=\mathrm{clip}(J(x)T(x)+A(1-T(x)),0,1)$",
        ha="center",
        va="center",
        fontsize=12.2,
        color=INK,
    )

    out_y, out_h, out_w = 0.035, 0.29, 0.245
    out_size = 0.165
    out_cards = [
        (0.235, "Transmission  T(x)", "T thấp → ảnh hưởng haze mạnh hơn", CORAL, transmission, "gray"),
        (0.520, "Ảnh haze tổng hợp  Iₕₐzy", "Đầu vào haze mới cho huấn luyện", BLUE, hazy, None),
    ]
    for x, title, subtitle, color, image, cmap in out_cards:
        card(x, out_y, out_w, out_h, title, subtitle, color)
        image_axis(x + (out_w - out_size) / 2, out_y + 0.055, out_size, image, cmap)

    def arrow(start: tuple[float, float], end: tuple[float, float], color: str = MUTED, rad: float = 0.0) -> None:
        fig.add_artist(
            FancyArrowPatch(
                start,
                end,
                transform=fig.transFigure,
                arrowstyle="-|>",
                mutation_scale=12,
                connectionstyle=f"arc3,rad={rad}",
                linewidth=1.45,
                color=color,
                alpha=0.88,
            )
        )

    # Inputs feed the scattering block; the transmission and reconstructed haze
    # are shown as its two outputs. The connector labels make the data flow clear.
    arrow((0.167, card_y - 0.008), (0.40, formula_y + formula_h + 0.008), BLUE, -0.10)
    arrow((0.500, card_y - 0.008), (0.36, formula_y + formula_h + 0.008), TEAL, 0.08)
    arrow((0.833, card_y - 0.008), (0.40, formula_y + formula_h + 0.008), GOLD, 0.18)
    arrow((0.30, formula_y - 0.008), (0.357, out_y + out_h + 0.006), CORAL, 0.04)
    arrow((0.72, formula_y - 0.008), (0.642, out_y + out_h + 0.006), BLUE, -0.04)

    # Compact parameter note, useful when the figure is reused in a report.
    fig.text(
        0.5,
        0.012,
        f"Ví dụ cố định: β={BETA:.2f}, α={DENSITY_SCALE:.2f}, s={DEPTH_SCALE:.1f}, A={AIRLIGHT:.2f}. Khi train, map và các tham số được lấy mẫu ngẫu nhiên theo cấu hình MCBM.",
        ha="center",
        va="center",
        fontsize=8.2,
        color=MUTED,
    )

    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "mcbm_haze_generation.png", dpi=220, facecolor=fig.get_facecolor(), bbox_inches="tight")
    fig.savefig(OUT / "mcbm_haze_generation.svg", facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print(OUT / "mcbm_haze_generation.png")
    print(OUT / "mcbm_haze_generation.svg")


if __name__ == "__main__":
    main()
