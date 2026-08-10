"""Utilities for reproducible building-facade defect detection experiments."""

from .dataset import AnnotationError, ImageLabelRecord, pair_image_labels, split_records
from .metrics import BinaryMetrics, binary_metrics

__all__ = [
    "AnnotationError",
    "BinaryMetrics",
    "ImageLabelRecord",
    "binary_metrics",
    "pair_image_labels",
    "split_records",
]

__version__ = "0.1.0"

