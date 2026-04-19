"""A lightweight per-category normal-image statistics model."""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

from ..features.anomaly_maps import smooth_anomaly_map, topk_score, zscore_map
from ..utils.io import ensure_dir, save_json
from .base import AnomalyPrediction, BaseAnomalyModel


@dataclass
class NormalStatsState:
    """Serializable model state."""

    category: str
    eps: float
    pixel_quantile: float
    score_quantile: float
    pixel_threshold: float
    score_threshold: float
    topk_ratio: float
    smoothing_sigma: float


class CategoryNormalStatsModel(BaseAnomalyModel):
    """Category-wise model built from the mean and std of good images."""

    def __init__(self, category: str, eps: float, pixel_quantile: float, score_quantile: float, topk_ratio: float, smoothing_sigma: float = 0.0):
        self.category = category
        self.eps = eps
        self.pixel_quantile = pixel_quantile
        self.score_quantile = score_quantile
        self.topk_ratio = topk_ratio
        self.smoothing_sigma = smoothing_sigma
        self.mean_image: Optional[np.ndarray] = None
        self.std_image: Optional[np.ndarray] = None
        self.pixel_threshold: Optional[float] = None
        self.score_threshold: Optional[float] = None

    def fit(self, images: List[np.ndarray]) -> "CategoryNormalStatsModel":
        stack = np.stack(images, axis=0).astype(np.float32)
        self.mean_image = stack.mean(axis=0)
        self.std_image = stack.std(axis=0)

        residuals = [smooth_anomaly_map(zscore_map(image, self.mean_image, self.std_image, self.eps), self.smoothing_sigma) for image in images]
        residual_stack = np.stack(residuals, axis=0)
        scores = np.asarray([topk_score(residual, self.topk_ratio) for residual in residuals], dtype=np.float32)

        self.pixel_threshold = float(np.quantile(residual_stack.reshape(-1), self.pixel_quantile))
        self.score_threshold = float(np.quantile(scores, self.score_quantile))
        return self

    def predict(self, image: np.ndarray) -> AnomalyPrediction:
        if self.mean_image is None or self.std_image is None or self.pixel_threshold is None:
            raise RuntimeError("Model must be fitted before prediction.")
        anomaly_map = smooth_anomaly_map(
            zscore_map(image, self.mean_image, self.std_image, self.eps),
            self.smoothing_sigma,
        )
        score = topk_score(anomaly_map, self.topk_ratio)
        binary_mask = (anomaly_map >= self.pixel_threshold).astype(np.uint8)
        return AnomalyPrediction(score=score, anomaly_map=anomaly_map, binary_mask=binary_mask)

    def save(self, output_dir: Path) -> None:
        """Persist the model state."""
        if self.mean_image is None or self.std_image is None or self.pixel_threshold is None or self.score_threshold is None:
            raise RuntimeError("Cannot save an unfitted model.")

        ensure_dir(output_dir)
        np.savez_compressed(
            output_dir / f"{self.category}_normal_stats.npz",
            mean_image=self.mean_image,
            std_image=self.std_image,
        )
        save_json(
            output_dir / f"{self.category}_normal_stats.json",
            asdict(
                NormalStatsState(
                    category=self.category,
                    eps=self.eps,
                    pixel_quantile=self.pixel_quantile,
                    score_quantile=self.score_quantile,
                    pixel_threshold=self.pixel_threshold,
                    score_threshold=self.score_threshold,
                    topk_ratio=self.topk_ratio,
                    smoothing_sigma=self.smoothing_sigma,
                )
            ),
        )

    @classmethod
    def load(cls, category: str, output_dir: Path) -> "CategoryNormalStatsModel":
        """Load a previously saved model."""
        weights = np.load(output_dir / f"{category}_normal_stats.npz")
        import json

        with open(output_dir / f"{category}_normal_stats.json", "r", encoding="utf-8") as handle:
            state = json.load(handle)

        model = cls(
            category=category,
            eps=float(state["eps"]),
            pixel_quantile=float(state.get("pixel_quantile", 0.995)),
            score_quantile=float(state.get("score_quantile", 0.995)),
            topk_ratio=float(state["topk_ratio"]),
            smoothing_sigma=float(state.get("smoothing_sigma", 0.0)),
        )
        model.mean_image = weights["mean_image"]
        model.std_image = weights["std_image"]
        model.pixel_threshold = float(state["pixel_threshold"])
        model.score_threshold = float(state["score_threshold"])
        return model
