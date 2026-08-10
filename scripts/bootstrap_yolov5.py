#!/usr/bin/env python3
"""Clone a pinned upstream YOLOv5 release without modifying user data."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=Path("third_party/yolov5"))
    parser.add_argument("--tag", default="v7.0", help="Upstream tag or branch to clone.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    destination = args.destination.expanduser().resolve()
    if destination.exists():
        if (destination / ".git").is_dir() and (destination / "train.py").is_file():
            print(f"YOLOv5 checkout already exists at {destination}")
            return
        raise FileExistsError(
            f"Destination already exists and is not a YOLOv5 checkout: {destination}"
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "git",
        "clone",
        "--depth",
        "1",
        "--branch",
        args.tag,
        "https://github.com/ultralytics/yolov5.git",
        str(destination),
    ]
    subprocess.run(command, check=True)
    print(f"Cloned YOLOv5 {args.tag} to {destination}")


if __name__ == "__main__":
    main()

