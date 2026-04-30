"""Abstract surface that every visual backend must implement.

The framework is built around a narrow contract: a fitted backend
produces an image-level score, a pixel-resolution anomaly map, and
a thresholded binary mask. Anything downstream --- region
extraction, the rule engine, the metrics layer --- depends only
on this interface. A user can bring their own backend by
subclassing ``BaseAnomalyModel`` and implementing ``fit``,
``predict``, and ``save``; a matching branch in
``models/factory.py`` is the only other change needed.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class AnomalyPrediction:
    """Single-image output of a fitted backend.

    Returned by ``BaseAnomalyModel.predict``. The pipeline assumes
    ``anomaly_map`` and ``binary_mask`` are at the original image
    resolution, and that ``binary_mask`` was already obtained by
    thresholding ``anomaly_map`` against the model's pixel-level
    threshold.
    """

    score: float
    anomaly_map: np.ndarray
    binary_mask: np.ndarray


class BaseAnomalyModel(ABC):
    """Common interface for the statistics baseline and feature backend.

    Subclasses must populate ``score_threshold`` during ``fit`` so
    the evaluation pipeline can normalize scores against a category-
    specific operating point without inspecting model internals.
    """

    category: str
    score_threshold: float

    @abstractmethod
    def fit(self, images: list[np.ndarray]) -> "BaseAnomalyModel":
        raise NotImplementedError

    @abstractmethod
    def predict(self, image: np.ndarray) -> AnomalyPrediction:
        raise NotImplementedError

    @abstractmethod
    def save(self, output_dir: Path) -> None:
        raise NotImplementedError
