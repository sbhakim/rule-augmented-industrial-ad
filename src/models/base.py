"""Base interfaces for anomaly models."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class AnomalyPrediction:
    """One anomaly prediction."""

    score: float
    anomaly_map: np.ndarray
    binary_mask: np.ndarray


class BaseAnomalyModel(ABC):
    """Abstract anomaly model interface."""

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
