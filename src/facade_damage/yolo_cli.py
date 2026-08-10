"""Shared helpers for calling a pinned upstream YOLOv5 checkout."""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

import yaml


def require_upstream_script(yolov5_dir: Path, script_name: str) -> Path:
    yolov5_dir = Path(yolov5_dir).expanduser().resolve()
    script = yolov5_dir / script_name
    if not script.is_file():
        raise FileNotFoundError(
            f"Could not find {script_name} in {yolov5_dir}. "
            "Run scripts/bootstrap_yolov5.py first."
        )
    return script


def resolve_dataset_yaml(source: Path, destination: Path) -> Path:
    """Write a runtime YAML whose dataset root is absolute."""

    source = Path(source).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Dataset configuration does not exist: {source}")

    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Dataset configuration must contain a mapping: {source}")
    root = Path(payload.get("path", ".")).expanduser()
    if not root.is_absolute():
        root = (source.parent / root).resolve()
    payload["path"] = str(root)

    destination = Path(destination).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return destination


def run_upstream(script: Path, arguments: Sequence[str]) -> None:
    command = [sys.executable, str(script), *arguments]
    subprocess.run(command, cwd=script.parent, check=True)
