"""Model factory for config-driven model selection."""

from pathlib import Path

from ..config.settings import ModelConfig
from .base import BaseAnomalyModel
from .normal_stats import CategoryNormalStatsModel


def create_model(category: str, config: ModelConfig) -> BaseAnomalyModel:
    """Instantiate an anomaly model based on the configured model_type."""
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
    """Load a previously saved model based on the configured model_type."""
    if config.model_type == "normal_stats":
        return CategoryNormalStatsModel.load(category, models_dir)
    if config.model_type == "feature":
        from .feature_model import CategoryFeatureModel

        return CategoryFeatureModel.load(category, models_dir)
    raise ValueError(f"Unknown model_type: {config.model_type!r}")
