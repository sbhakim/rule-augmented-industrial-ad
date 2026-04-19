"""Anomaly-map helpers."""

import numpy as np
from scipy.ndimage import gaussian_filter


def zscore_map(image: np.ndarray, mean_image: np.ndarray, std_image: np.ndarray, eps: float) -> np.ndarray:
    """Compute an absolute z-score anomaly map."""
    return np.abs(image - mean_image) / (std_image + eps)


def smooth_anomaly_map(anomaly_map: np.ndarray, sigma: float) -> np.ndarray:
    """Apply Gaussian smoothing to reduce speckle noise in anomaly maps."""
    if sigma <= 0.0:
        return anomaly_map
    return gaussian_filter(anomaly_map.astype(np.float32), sigma=sigma)


def topk_score(anomaly_map: np.ndarray, topk_ratio: float) -> float:
    """Compute an image score from the average of the top-k anomaly values."""
    flat = anomaly_map.reshape(-1)
    k = max(1, int(len(flat) * topk_ratio))
    values = np.partition(flat, len(flat) - k)[-k:]
    return float(values.mean())
