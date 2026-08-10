#!/usr/bin/env python3
"""Run YOLOv5 inference on an image, directory, video, URL, or camera source."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from facade_damage.yolo_cli import require_upstream_script, run_upstream  # noqa: E402


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yolov5-dir", type=Path, default=Path("third_party/yolov5"))
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--conf-thres", type=float, default=0.25)
    parser.add_argument("--iou-thres", type=float, default=0.45)
    parser.add_argument("--device", default="")
    parser.add_argument("--name", default="facade_predictions")
    parser.add_argument("--project", type=Path, default=Path("runs/detect"))
    return parser.parse_known_args()


def main() -> None:
    args, extra = parse_args()
    script = require_upstream_script(args.yolov5_dir, "detect.py")
    arguments = [
        "--weights",
        str(args.weights.expanduser().resolve()),
        "--source",
        args.source,
        "--imgsz",
        str(args.img_size),
        "--conf-thres",
        str(args.conf_thres),
        "--iou-thres",
        str(args.iou_thres),
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
