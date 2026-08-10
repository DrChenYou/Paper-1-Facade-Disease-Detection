from pathlib import Path

import yaml

from facade_damage.yolo_cli import resolve_dataset_yaml


def test_resolve_dataset_yaml_makes_root_absolute(tmp_path: Path) -> None:
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    source = config_dir / "dataset.yaml"
    source.write_text(
        "path: ../data/processed\ntrain: images/train\nval: images/val\nnc: 2\n",
        encoding="utf-8",
    )
    destination = tmp_path / "runs" / "resolved.yaml"

    result = resolve_dataset_yaml(source, destination)
    payload = yaml.safe_load(result.read_text(encoding="utf-8"))

    assert Path(payload["path"]).is_absolute()
    assert Path(payload["path"]) == (tmp_path / "data" / "processed").resolve()

