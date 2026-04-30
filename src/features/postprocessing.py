"""Binary-mask cleanup operating between the detector and the rule layer.

This module implements the standard morphological cleanup chain
(opening, closing, small-component filtering) without an external
dependency on OpenCV or scipy. Keeping it dependency-free is a
deliberate choice so the repository can run on minimal Python
installations and so the cleanup behaviour is exactly reproducible
across platforms.

The cleanup is the only place where the raw thresholded anomaly
map is allowed to be modified before the symbolic layer sees it.
Any change to the morphology choices here will affect every
downstream metric that uses the cleaned mask (Dice, IoU, coverage,
the region descriptors fed to the rule engine).
"""

import numpy as np

from .connected_components import label_components


def _shift(mask: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Translate a binary mask by (dy, dx) with zero padding.

    Used as the elementary building block for dilation and erosion
    over a 3x3 structuring element. Implementing the shift by
    explicit slicing avoids the cost of allocating a wrapped roll
    and the risk of values wrapping around image edges.
    """
    shifted = np.zeros_like(mask, dtype=np.uint8)

    src_y_start = max(0, -dy)
    src_y_end = mask.shape[0] - max(0, dy)
    src_x_start = max(0, -dx)
    src_x_end = mask.shape[1] - max(0, dx)

    dst_y_start = max(0, dy)
    dst_y_end = dst_y_start + (src_y_end - src_y_start)
    dst_x_start = max(0, dx)
    dst_x_end = dst_x_start + (src_x_end - src_x_start)

    if src_y_end <= src_y_start or src_x_end <= src_x_start:
        return shifted

    shifted[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = mask[src_y_start:src_y_end, src_x_start:src_x_end]
    return shifted


def binary_dilation(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    """Dilate a binary mask using a 3x3 neighborhood."""
    current = (mask > 0).astype(np.uint8)
    for _ in range(max(0, iterations)):
        neighbors = [_shift(current, dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)]
        current = np.maximum.reduce(neighbors).astype(np.uint8)
    return current


def binary_erosion(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    """Erode a binary mask using a 3x3 neighborhood."""
    current = (mask > 0).astype(np.uint8)
    for _ in range(max(0, iterations)):
        neighbors = [_shift(current, dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)]
        current = np.minimum.reduce(neighbors).astype(np.uint8)
    return current


def binary_opening(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    """Opening removes isolated speckles before region extraction."""
    return binary_dilation(binary_erosion(mask, iterations), iterations)


def binary_closing(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    """Closing fills small holes inside salient anomaly regions."""
    return binary_erosion(binary_dilation(mask, iterations), iterations)


def filter_small_components(mask: np.ndarray, min_component_area_px: int) -> np.ndarray:
    """Keep only connected components above the configured area threshold."""
    filtered = np.zeros_like(mask, dtype=np.uint8)
    for component in label_components(mask):
        if len(component.coordinates) >= min_component_area_px:
            ys = component.coordinates[:, 0]
            xs = component.coordinates[:, 1]
            filtered[ys, xs] = 1
    return filtered


def cleanup_binary_mask(
    mask: np.ndarray,
    opening_iterations: int,
    closing_iterations: int,
    min_component_area_px: int,
) -> np.ndarray:
    """Apply the canonical opening/closing/area-filter chain.

    The order matters: opening first removes isolated speckle that
    would otherwise survive as tiny "regions"; closing then fills
    small holes inside genuine defect blobs; the area filter
    finally drops any remaining components smaller than the
    configured minimum. All three iteration counts are configured
    by the caller so an ablation can disable any step.
    """
    current = (mask > 0).astype(np.uint8)
    current = binary_opening(current, opening_iterations)
    current = binary_closing(current, closing_iterations)
    current = filter_small_components(current, min_component_area_px)
    return current
