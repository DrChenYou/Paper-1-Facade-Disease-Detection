from pathlib import Path

import pytest

from facade_damage.dataset import (
    AnnotationError,
    pair_image_labels,
    parse_yolo_label,
    split_records,
)


def write_pair(images_dir: Path, labels_dir: Path, index: int, label: str) -> None:
    (images_dir / f"facade_{index:02d}.jpg").write_bytes(f"image-{index}".encode())
    (labels_dir / f"facade_{index:02d}.txt").write_text(label, encoding="utf-8")


def test_parse_yolo_label_accepts_two_classes(tmp_path: Path) -> None:
    label = tmp_path / "valid.txt"
    label.write_text("0 0.50 0.50 0.20 0.30\n1 0.25 0.25 0.10 0.10\n", encoding="utf-8")
    assert parse_yolo_label(label, num_classes=2) == 2


@pytest.mark.parametrize(
    "row",
    [
        "2 0.5 0.5 0.2 0.2",
        "0 1.2 0.5 0.2 0.2",
        "0 0.5 0.5 0.0 0.2",
        "0 0.05 0.5 0.2 0.2",
        "0 0.5 0.5 0.2",
    ],
)
def test_parse_yolo_label_rejects_invalid_rows(tmp_path: Path, row: str) -> None:
    label = tmp_path / "invalid.txt"
    label.write_text(row, encoding="utf-8")
    with pytest.raises(AnnotationError):
        parse_yolo_label(label, num_classes=2)


def test_pair_and_split_are_deterministic(tmp_path: Path) -> None:
    images = tmp_path / "images"
    labels = tmp_path / "labels"
    images.mkdir()
    labels.mkdir()
    for index in range(10):
        write_pair(images, labels, index, "0 0.5 0.5 0.2 0.2\n")

    records = pair_image_labels(images, labels)
    first = split_records(records, seed=42)
    second = split_records(records, seed=42)

    assert {name: len(items) for name, items in first.items()} == {
        "train": 8,
        "val": 1,
        "test": 1,
    }
    assert first == second
    assert set(first["train"]).isdisjoint(first["test"])


def test_missing_label_is_reported(tmp_path: Path) -> None:
    images = tmp_path / "images"
    labels = tmp_path / "labels"
    images.mkdir()
    labels.mkdir()
    (images / "unlabelled.jpg").write_bytes(b"image")

    with pytest.raises(FileNotFoundError, match="Missing label"):
        pair_image_labels(images, labels)

