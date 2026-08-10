#!/usr/bin/env python3
"""Evaluate trained weights on the held-out test split."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from facade_damage.yolo_cli import (  # noqa: E402
    require_upstream_script,
    resolve_dataset_yaml,
    run_upstream,
)


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yolov5-dir", type=Path, default=Path("third_party/yolov5"))
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("configs/building_disease.yaml"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--device", default="")
    parser.add_argument("--name", default="facade_test")
    parser.add_argument("--project", type=Path, default=Path("runs/val"))
    return parser.parse_known_args()


def main() -> None:
    args, extra = parse_args()
    script = require_upstream_script(args.yolov5_dir, "val.py")
    data_yaml = resolve_dataset_yaml(
        args.data,
        PROJECT_ROOT / "runs" / "resolved_configs" / "building_disease.yaml",
    )
    arguments = [
        "--data",
        str(data_yaml),
        "--weights",
        str(args.weights.expanduser().resolve()),
        "--task",
        "test",
        "--batch-size",
        str(args.batch_size),
        "--imgsz",
        str(args.img_size),
        "--name",
        args.name,
        "--project",
        str(args.project.expanduser().resolve()),
    ]
    if args.device:
        arguments.extend(["--device", args.device])
    run_upstream(script, [*arguments, *extra])


if __name__ == "__main__":
    main()

