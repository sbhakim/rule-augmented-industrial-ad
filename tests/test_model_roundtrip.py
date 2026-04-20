import json

import numpy as np

from src.models.feature_model import CategoryFeatureModel
from src.models.normal_stats import CategoryNormalStatsModel


def test_normal_stats_roundtrip_preserves_inference_settings(tmp_path):
    model = CategoryNormalStatsModel(
        category="demo",
        eps=1e-6,
        pixel_quantile=0.9,
        score_quantile=0.8,
        topk_ratio=0.1,
        smoothing_sigma=2.5,
    )
    model.mean_image = np.zeros((2, 2), dtype=np.float32)
    model.std_image = np.ones((2, 2), dtype=np.float32)
    model.pixel_threshold = 1.0
    model.score_threshold = 2.0

    model.save(tmp_path)
    loaded = CategoryNormalStatsModel.load("demo", tmp_path)

    assert loaded.smoothing_sigma == 2.5
    assert loaded.pixel_quantile == 0.9
    assert loaded.score_quantile == 0.8
    assert loaded.topk_ratio == 0.1


def test_normal_stats_load_supports_legacy_state(tmp_path):
    np.savez_compressed(
        tmp_path / "demo_normal_stats.npz",
        mean_image=np.zeros((2, 2), dtype=np.float32),
        std_image=np.ones((2, 2), dtype=np.float32),
    )
    with open(tmp_path / "demo_normal_stats.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "category": "demo",
                "eps": 1e-6,
                "pixel_threshold": 1.0,
                "score_threshold": 2.0,
                "topk_ratio": 0.1,
            },
            handle,
        )

    loaded = CategoryNormalStatsModel.load("demo", tmp_path)

    assert loaded.smoothing_sigma == 0.0
    assert loaded.pixel_quantile == 0.995
    assert loaded.score_quantile == 0.995


def test_feature_model_roundtrip_preserves_inference_settings(tmp_path):
    model = CategoryFeatureModel(
        category="demo",
        topk_ratio=0.1,
        pixel_quantile=0.91,
        score_quantile=0.81,
        smoothing_sigma=1.5,
        image_size=(128, 128),
    )
    model.mean = np.zeros((3, 4), dtype=np.float32)
    model.cov_inv = np.zeros((4, 3, 3), dtype=np.float32)
    model.dim_indices = np.array([0, 1, 2], dtype=np.int64)
    model.pixel_threshold = 1.0
    model.score_threshold = 2.0
    model.full_embedding_dim = 3
    model.reduced_dim = 3
    model.patch_h = 2
    model.patch_w = 2

    model.save(tmp_path)
    loaded = CategoryFeatureModel.load("demo", tmp_path)

    assert loaded.smoothing_sigma == 1.5
    assert loaded.image_size == (128, 128)
    assert loaded.pixel_quantile == 0.91
    assert loaded.score_quantile == 0.81
    assert loaded.topk_ratio == 0.1


def test_feature_model_load_supports_legacy_state(tmp_path):
    np.savez_compressed(
        tmp_path / "demo_feature_model.npz",
        mean=np.zeros((3, 4), dtype=np.float32),
        cov_inv=np.zeros((4, 3, 3), dtype=np.float32),
        dim_indices=np.array([0, 1, 2], dtype=np.int64),
    )
    with open(tmp_path / "demo_feature_model.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "category": "demo",
                "backbone": "wide_resnet50_2",
                "feature_layers": [1, 2, 3],
                "full_embedding_dim": 3,
                "reduced_dim": 3,
                "pixel_threshold": 1.0,
                "score_threshold": 2.0,
                "topk_ratio": 0.1,
                "patch_h": 2,
                "patch_w": 2,
            },
            handle,
        )

    loaded = CategoryFeatureModel.load("demo", tmp_path)

    assert loaded.smoothing_sigma == 4.0
    assert loaded.image_size == (256, 256)
    assert loaded.pixel_quantile == 0.995
    assert loaded.score_quantile == 0.995
