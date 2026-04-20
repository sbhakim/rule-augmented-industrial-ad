import numpy as np

from src.evaluation.metrics import auroc


def test_auroc_handles_tied_scores():
    y_true = np.array([0, 1, 0, 1], dtype=np.int32)
    y_score = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float64)

    assert auroc(y_true, y_score) == 0.5


def test_auroc_matches_known_example():
    y_true = np.array([0, 0, 1, 1], dtype=np.int32)
    y_score = np.array([0.1, 0.4, 0.35, 0.8], dtype=np.float64)

    assert auroc(y_true, y_score) == 0.75
