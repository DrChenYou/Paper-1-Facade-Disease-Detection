#!/usr/bin/env python3
"""Validate YOLO annotations and create an auditable 8:1:1 dataset split."""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from facade_damage.dataset import pair_image_labels, split_records  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--labels-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and report split sizes without copying files.",
    )
    return parser.parse_args()


def write_split(output_dir: Path, split_map: dict) -> None:
    output_dir = output_dir.expanduser().resolve()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    if output_dir.exists():
        raise FileExistsError(
            f"Output already exists: {output_dir}. Choose a new directory to avoid mixing splits."
        )

    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent))
    try:
        rows: list[dict[str, str | int]] = []
        for split_name, records in split_map.items():
            image_target = staging / "images" / split_name
            label_target = staging / "labels" / split_name
            image_target.mkdir(parents=True, exist_ok=True)
            label_target.mkdir(parents=True, exist_ok=True)

            for record in records:
                shutil.copy2(record.image, image_target / record.image.name)
                shutil.copy2(record.label, label_target / record.label.name)
                rows.append(
                    {
                        "image": record.image.name,
                        "label": record.label.name,
                        "split": split_name,
                        "objects": record.object_count,
                        "image_sha256": record.image_sha256,
                    }
                )

        with (staging / "manifest.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=["image", "label", "split", "objects", "image_sha256"],
            )
            writer.writeheader()
            writer.writerows(rows)

        staging.replace(output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> None:
    args = parse_args()
    records = pair_image_labels(
        args.images_dir,
        args.labels_dir,
        num_classes=args.num_classes,
    )
    split_map = split_records(
        records,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    counts = ", ".join(f"{name}={len(items)}" for name, items in split_map.items())
    print(f"Validated {len(records)} image-label pairs ({counts}).")
    if args.dry_run:
        return

    write_split(args.output_dir, split_map)
    print(f"Prepared dataset at {args.output_dir.expanduser().resolve()}")


if __name__ == "__main__":
    main()

