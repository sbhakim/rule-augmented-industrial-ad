"""Image and mask loading utilities."""

from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image


def _resample(name: str) -> int:
    """Resolve Pillow resampling constants across versions."""
    if hasattr(Image, "Resampling"):
        return getattr(Image.Resampling, name)
    return getattr(Image, name)


def load_image_array(path: Path, image_size: Tuple[int, int], grayscale: bool) -> np.ndarray:
    """Load an image, resize it, and normalize to [0, 1]."""
    image = Image.open(path)
    image = image.convert("L" if grayscale else "RGB")
    image = image.resize(image_size, _resample("BILINEAR"))
    array = np.asarray(image, dtype=np.float32) / 255.0
    return array


def load_mask_array(path: Path, image_size: Tuple[int, int]) -> np.ndarray:
    """Load a binary ground-truth mask."""
    mask = Image.open(path).convert("L")
    mask = mask.resize(image_size, _resample("NEAREST"))
    return (np.asarray(mask, dtype=np.uint8) > 0).astype(np.uint8)
