"""Evaluation metrics for detection, localization, and explanation coverage.

The metrics here are deliberately implemented from primitives so the
repository has no hard dependency on ``scikit-learn`` or any other
heavy stack at evaluation time. Numerical conventions follow the
references most users will compare against (rank-sum AUROC, MVTec PRO
integrated to a finite false-positive budget, set-based Dice/IoU on
binary masks). Edge cases that would otherwise raise are mapped to
NaN so a category with no anomalies in the test split does not
poison aggregate reporting.
"""

from typing import Dict, Iterable, Tuple

import numpy as np
from scipy import ndimage


def auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Mann-Whitney rank-sum form of AUROC.

    Equivalent to the trapezoidal area under the ROC curve but
    computed from average ranks. Average-rank handling means
    repeated score values are scored consistently regardless of
    sort stability, which matters because anomaly maps often
    contain large blocks of identical low scores.
    """
    y_true = np.asarray(y_true).astype(np.int32)
    y_score = np.asarray(y_score).astype(np.float64)
    pos = y_true == 1
    neg = y_true == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    # Walk the sorted scores in groups of ties and assign each
    # tied block its average rank. This is the standard fix for
    # the Mann-Whitney estimator under ties.
    order = np.argsort(y_score, kind="mergesort")
    sorted_scores = y_score[order]
    ranks = np.empty_like(order, dtype=np.float64)
    start = 0
    while start < len(sorted_scores):
        end = start + 1
        while end < len(sorted_scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        average_rank = (start + end + 1) / 2.0
        ranks[order[start:end]] = average_rank
        start = end
    pos_rank_sum = ranks[pos].sum()
    return float((pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def dice_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Dice score for binary masks."""
    y_true = y_true.astype(bool)
    y_pred = y_pred.astype(bool)
    intersection = np.logical_and(y_true, y_pred).sum()
    total = y_true.sum() + y_pred.sum()
    if total == 0:
        return 1.0
    return float((2.0 * intersection) / total)


def iou_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute IoU for binary masks."""
    y_true = y_true.astype(bool)
    y_pred = y_pred.astype(bool)
    union = np.logical_or(y_true, y_pred).sum()
    if union == 0:
        return 1.0
    intersection = np.logical_and(y_true, y_pred).sum()
    return float(intersection / union)


def coverage_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Recall of ground-truth anomaly pixels by the predicted mask.

    Used as the explanation-side coverage signal: how much of the
    real defect is contained in the flagged regions that survive
    cleanup and reach the symbolic layer. Distinct from pixel-level
    AUROC because it is computed against the post-thresholded,
    post-cleanup mask rather than the continuous anomaly map.
    Returns NaN for nominal images so they can be filtered out
    cleanly when aggregating.
    """
    y_true = y_true.astype(bool)
    y_pred = y_pred.astype(bool)
    positive = int(y_true.sum())
    if positive == 0:
        return float("nan")
    inside = int(np.logical_and(y_true, y_pred).sum())
    return float(inside / positive)


def pro_score(
    masks: Iterable[np.ndarray],
    scores: Iterable[np.ndarray],
    max_fpr: float = 0.3,
    num_thresholds: int = 100,
) -> float:
    """Per-region overlap integrated over a bounded false-positive budget.

    The PRO metric weights every defect region equally regardless of
    its size, which is the opposite of pixel-level recall and is the
    reason it is preferred in industrial inspection. The (FPR, PRO)
    curve is built by sweeping a set of score thresholds; the result
    is the area under that curve restricted to the low-FPR regime
    where deployment usually operates, normalized so the value lies
    in [0, 1].
    """
    masks = [np.asarray(m, dtype=bool) for m in masks]
    scores = [np.asarray(s, dtype=np.float64) for s in scores]
    if not masks:
        return float("nan")

    # Sample the threshold set as quantiles of the pooled score
    # distribution so that points along the (FPR, PRO) curve are
    # spread roughly evenly rather than clustering near the
    # extremes of the score range.
    all_scores = np.concatenate([s.reshape(-1) for s in scores])
    score_min = float(np.min(all_scores))
    score_max = float(np.max(all_scores))
    if score_max <= score_min:
        return float("nan")
    percentiles = np.linspace(100.0, 0.0, num=num_thresholds)
    thresholds = np.percentile(all_scores, percentiles)

    # Cache the connected-component decomposition per image once;
    # the same regions are reused at every threshold below.
    region_cache = []
    total_nominal = 0
    for mask, score_map in zip(masks, scores):
        labels, num_regions = ndimage.label(mask)
        regions = []
        for region_id in range(1, num_regions + 1):
            region_pixels = labels == region_id
            area = int(region_pixels.sum())
            if area == 0:
                continue
            regions.append((region_pixels, area))
        region_cache.append(regions)
        total_nominal += int((~mask).sum())

    if total_nominal == 0:
        return float("nan")

    pro_values = []
    fpr_values = []
    for threshold in thresholds:
        false_positive = 0
        region_overlaps = []
        for (mask, score_map), regions in zip(zip(masks, scores), region_cache):
            predicted = score_map >= threshold
            false_positive += int(np.logical_and(predicted, ~mask).sum())
            for region_pixels, area in regions:
                overlap = int(np.logical_and(predicted, region_pixels).sum())
                region_overlaps.append(overlap / area)
        if not region_overlaps:
            continue
        pro_values.append(float(np.mean(region_overlaps)))
        fpr_values.append(false_positive / total_nominal)

    if not pro_values:
        return float("nan")
    fpr_arr = np.asarray(fpr_values)
    pro_arr = np.asarray(pro_values)
    order = np.argsort(fpr_arr)
    fpr_arr = fpr_arr[order]
    pro_arr = pro_arr[order]
    mask_range = fpr_arr <= max_fpr
    if mask_range.sum() < 2:
        return float("nan")
    area = float(np.trapz(pro_arr[mask_range], fpr_arr[mask_range]))
    return area / max_fpr


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute threshold-based image-level classification metrics."""
    y_true = np.asarray(y_true).astype(np.int32)
    y_pred = np.asarray(y_pred).astype(np.int32)

    tp = int(np.logical_and(y_true == 1, y_pred == 1).sum())
    tn = int(np.logical_and(y_true == 0, y_pred == 0).sum())
    fp = int(np.logical_and(y_true == 0, y_pred == 1).sum())
    fn = int(np.logical_and(y_true == 1, y_pred == 0).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    accuracy = (tp + tn) / max(1, len(y_true))
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "specificity": float(specificity),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }
