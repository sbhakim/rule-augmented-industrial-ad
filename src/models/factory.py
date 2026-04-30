"""Model dispatch by ``model_type`` string.

Both ``create_model`` and ``load_model`` look at the same field of
the config so the choice of backend is a single string in the JSON
file. The feature model is imported lazily inside the dispatch
branch so users running the statistics-only baseline do not pay
the torch import cost.

Adding a new backend means adding a third branch here and a
matching subclass of ``BaseAnomalyModel``; no other module needs
to know about the backend type.
"""

from pathlib import Path

from ..config.settings import ModelConfig
from .base import BaseAnomalyModel
from .normal_stats import CategoryNormalStatsModel


def create_model(category: str, config: ModelConfig) -> BaseAnomalyModel:
    """Build a fresh, unfitted model instance for ``category``."""
    if config.model_type == "normal_stats":
        return CategoryNormalStatsModel(
            category=category,
            eps=config.eps,
            pixel_quantile=config.pixel_quantile,
            score_quantile=config.score_quantile,
            topk_ratio=config.topk_ratio,
            smoothing_sigma=config.smoothing_sigma,
        )
    if config.model_type == "feature":
        from .feature_model import CategoryFeatureModel

        return CategoryFeatureModel(
            category=category,
            backbone=config.backbone,
            feature_layers=config.feature_layers,
            pixel_quantile=config.pixel_quantile,
            score_quantile=config.score_quantile,
            topk_ratio=config.topk_ratio,
            smoothing_sigma=config.smoothing_sigma,
            image_size=(256, 256),
        )
    raise ValueError(f"Unknown model_type: {config.model_type!r}")


def load_model(category: str, config: ModelConfig, models_dir: Path) -> BaseAnomalyModel:
    """Reload a previously fitted model from ``models_dir``."""
    if config.model_type == "normal_stats":
        return CategoryNormalStatsModel.load(category, models_dir)
    if config.model_type == "feature":
        from .feature_model import CategoryFeatureModel

        return CategoryFeatureModel.load(category, models_dir)
    raise ValueError(f"Unknown model_type: {config.model_type!r}")
