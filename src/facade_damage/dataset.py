"""Dataset validation and deterministic splitting for YOLO annotations."""

from __future__ import annotations

import hashlib
import random
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


class AnnotationError(ValueError):
    """Raised when a YOLO annotation cannot be validated."""


@dataclass(frozen=True)
class ImageLabelRecord:
    """A validated image-label pair."""

    image: Path
    label: Path
    object_count: int

    @property
    def image_sha256(self) -> str:
        digest = hashlib.sha256()
        with self.image.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()


def parse_yolo_label(path: Path, num_classes: int = 2) -> int:
    """Validate one YOLO label file and return its object count.

    Empty files are accepted as explicitly verified negative images.
    """

    if num_classes <= 0:
        raise ValueError("num_classes must be positive")
    if not path.is_file():
        raise AnnotationError(f"Label file does not exist: {path}")

    object_count = 0
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line:
            continue

        fields = line.split()
        if len(fields) != 5:
            raise AnnotationError(
                f"{path}:{line_number}: expected 5 values, received {len(fields)}"
            )

        try:
            class_value = float(fields[0])
            coordinates = [float(value) for value in fields[1:]]
        except ValueError as error:
            raise AnnotationError(f"{path}:{line_number}: values must be numeric") from error

        class_id = int(class_value)
        if class_value != class_id:
            raise AnnotationError(f"{path}:{line_number}: class ID must be an integer")
        if not 0 <= class_id < num_classes:
            raise AnnotationError(
                f"{path}:{line_number}: class ID {class_id} is outside [0, {num_classes - 1}]"
            )

        x_center, y_center, width, height = coordinates
        if not all(0.0 <= value <= 1.0 for value in coordinates):
            raise AnnotationError(
                f"{path}:{line_number}: normalised coordinates must be within [0, 1]"
            )
        if width <= 0.0 or height <= 0.0:
            raise AnnotationError(f"{path}:{line_number}: width and height must be positive")
        if x_center - width / 2 < 0 or x_center + width / 2 > 1:
            raise AnnotationError(f"{path}:{line_number}: box extends beyond image width")
        if y_center - height / 2 < 0 or y_center + height / 2 > 1:
            raise AnnotationError(f"{path}:{line_number}: box extends beyond image height")

        object_count += 1

    return object_count


def pair_image_labels(
    images_dir: Path,
    labels_dir: Path,
    *,
    num_classes: int = 2,
) -> list[ImageLabelRecord]:
    """Discover images, match same-stem labels, and validate the annotations."""

    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {images_dir}")
    if not labels_dir.is_dir():
        raise FileNotFoundError(f"Label directory does not exist: {labels_dir}")

    images = sorted(
        path
        for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        raise FileNotFoundError(f"No supported images found in: {images_dir}")

    seen_stems: set[str] = set()
    records: list[ImageLabelRecord] = []
    for image in images:
        if image.stem in seen_stems:
            raise ValueError(
                f"Duplicate image stem '{image.stem}'. Use unique stems across image extensions."
            )
        seen_stems.add(image.stem)

        label = labels_dir / f"{image.stem}.txt"
        if not label.is_file():
            raise FileNotFoundError(f"Missing label for {image.name}: {label}")
        records.append(
            ImageLabelRecord(
                image=image,
                label=label,
                object_count=parse_yolo_label(label, num_classes=num_classes),
            )
        )
    return records


def split_records(
    records: Iterable[ImageLabelRecord],
    *,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> dict[str, list[ImageLabelRecord]]:
    """Split records deterministically; the remainder is assigned to test."""

    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1")
    if not 0 <= val_ratio < 1:
        raise ValueError("val_ratio must be in [0, 1)")
    if train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio must be less than 1")

    shuffled = list(records)
    if len(shuffled) < 3:
        raise ValueError("At least three image-label pairs are required for a three-way split")
    random.Random(seed).shuffle(shuffled)

    train_end = int(len(shuffled) * train_ratio)
    val_end = train_end + int(len(shuffled) * val_ratio)
    return {
        "train": shuffled[:train_end],
        "val": shuffled[train_end:val_end],
        "test": shuffled[val_end:],
    }
