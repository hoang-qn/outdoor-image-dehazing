"""Generate offline full-image white-haze variants at several density levels."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def make_preview(
    dataset_root: Path,
    output_dir: Path,
    source_rows: list[dict[str, str]],
    levels: list[float],
    count: int,
    seed: int,
) -> Path:
    selected = random.Random(seed + 1).sample(source_rows, min(count, len(source_rows)))
    labels = ["Original haze", *(f"White haze {level:.0%}" for level in levels), "Clear target"]
    cell_width, cell_height = 220, 264
    margin, gap = 16, 10
    columns = len(labels)
    sheet = Image.new(
        "RGB",
        (margin * 2 + columns * cell_width + (columns - 1) * gap,
         margin * 2 + len(selected) * cell_height),
        (245, 245, 245),
    )
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()

    for row_index, source_row in enumerate(selected):
        row_top = margin + row_index * cell_height
        source_hazy = Path(source_row["hazy"].replace("\\", "/"))
        source_clear = Path(source_row["clear"].replace("\\", "/"))
        stem = Path(source_row["hazy"]).stem
        image_paths = [dataset_root / source_hazy]
        image_paths.extend(
            output_dir / "hazy" / f"level_{round(level * 100):02d}" / f"{stem}_haze{round(level * 100):02d}.jpg"
            for level in levels
        )
        image_paths.append(dataset_root / source_clear)
        draw.text((margin, row_top), f"Sample {source_row.get('scene_id', source_hazy.stem)}", fill=(20, 20, 20), font=font)

        for column, (image_path, label) in enumerate(zip(image_paths, labels)):
            left = margin + column * (cell_width + gap)
            draw.text((left, row_top + 18), label, fill=(30, 30, 30), font=font)
            with Image.open(image_path) as image:
                thumbnail = image.convert("RGB")
            thumbnail.thumbnail((cell_width, 220), Image.Resampling.LANCZOS)
            x = left + (cell_width - thumbnail.width) // 2
            y = row_top + 38 + (220 - thumbnail.height) // 2
            sheet.paste(thumbnail, (x, y))

    preview_path = output_dir / "preview.png"
    sheet.save(preview_path)
    return preview_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Create offline full-image white-haze RESIDE pairs.")
    parser.add_argument("--dataset-root", type=Path, default=Path("data/RESIDE-6K-Outdoor"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/RESIDE-6K-Outdoor/augmentations/white_haze_density"),
    )
    parser.add_argument("--opacity-levels", type=float, nargs="+", default=[0.10, 0.20, 0.30, 0.40, 0.50])
    parser.add_argument("--preview-count", type=int, default=4)
    parser.add_argument("--quality", type=int, default=95)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if any(not 0 < level <= 1 for level in args.opacity_levels):
        parser.error("all opacity levels must be in (0, 1]")
    if len(set(args.opacity_levels)) != len(args.opacity_levels):
        parser.error("opacity levels must be distinct")
    if args.preview_count < 1:
        parser.error("--preview-count must be at least 1")
    if not 1 <= args.quality <= 100:
        parser.error("--quality must be between 1 and 100")

    dataset_root = args.dataset_root.resolve()
    output_dir = args.output_dir.resolve()
    source_manifest = dataset_root / "pairs.csv"
    output_manifest = output_dir / "pairs.csv"
    if not source_manifest.is_file():
        raise FileNotFoundError(f"RESIDE pair manifest not found: {source_manifest}")
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. Use --overwrite to replace generated files."
        )

    with source_manifest.open("r", encoding="utf-8-sig", newline="") as stream:
        source_rows = [row for row in csv.DictReader(stream) if row.get("split") == "train"]
    if not source_rows:
        raise ValueError(f"No train pairs found in {source_manifest}")

    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    generated_rows: list[dict[str, str]] = []
    for pair_index, row in enumerate(source_rows, start=1):
        source_hazy = Path(row["hazy"].replace("\\", "/"))
        source_clear = Path(row["clear"].replace("\\", "/"))
        source_hazy_path = dataset_root / source_hazy
        source_clear_path = dataset_root / source_clear
        if not source_hazy_path.is_file() or not source_clear_path.is_file():
            raise FileNotFoundError(f"Missing source pair: {source_hazy_path} / {source_clear_path}")

        with Image.open(source_hazy_path) as source:
            original_hazy = source.convert("RGB")
        with Image.open(source_clear_path) as source:
            clear = source.convert("RGB")
        if original_hazy.size != clear.size:
            raise ValueError(f"Source pair dimensions differ: {source_hazy_path} / {source_clear_path}")

        stem = source_hazy_path.stem
        for level_index, opacity in enumerate(args.opacity_levels, start=1):
            level_percent = round(opacity * 100)
            level_name = f"level_{level_percent:02d}"
            generated_name = f"{stem}_haze{level_percent:02d}.jpg"
            output_hazy = output_dir / "hazy" / level_name / generated_name
            output_hazy.parent.mkdir(parents=True, exist_ok=True)
            white = Image.new("RGB", original_hazy.size, (255, 255, 255))
            augmented = Image.blend(original_hazy, white, opacity)
            augmented.save(output_hazy, format="JPEG", quality=args.quality)
            generated_rows.append({
                "split": "train",
                "hazy": f"augmentations/white_haze_density/hazy/{level_name}/{generated_name}",
                "clear": source_clear.as_posix(),
                "source_hazy": source_hazy.as_posix(),
                "source_clear": source_clear.as_posix(),
                "source_scene_id": row.get("scene_id", str(pair_index)),
                "density_level": str(level_index),
                "white_opacity": f"{opacity:.2f}",
                "seed": str(args.seed),
            })

        if pair_index % 100 == 0 or pair_index == len(source_rows):
            print(f"Processed {pair_index}/{len(source_rows)} source pairs")

    fieldnames = list(generated_rows[0])
    with output_manifest.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(generated_rows)

    config = {
        "method": "blend the complete hazy input with white airlight",
        "formula": "I_aug = (1 - opacity) * I_hazy + opacity * 255",
        "source_manifest": str(source_manifest),
        "source_pairs": len(source_rows),
        "generated_pairs": len(generated_rows),
        "opacity_levels": args.opacity_levels,
        "jpeg_quality": args.quality,
        "seed": args.seed,
    }
    (output_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    preview_path = make_preview(dataset_root, output_dir, source_rows, args.opacity_levels,
                                args.preview_count, args.seed)
    print(f"Offline pairs manifest: {output_manifest}")
    print(f"Preview: {preview_path}")
    print(f"Generated variants: {len(generated_rows)}")


if __name__ == "__main__":
    main()
