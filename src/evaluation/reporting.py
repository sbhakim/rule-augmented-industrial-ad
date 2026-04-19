"""Aggregate and persist evaluation reports."""

from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np

from ..utils.io import ensure_dir, save_json


def aggregate_records(records: List[Dict]) -> Dict:
    """Aggregate per-sample records into category-level summaries."""
    by_category: Dict[str, List[Dict]] = defaultdict(list)
    for record in records:
        by_category[record["category"]].append(record)

    summary = {"categories": {}, "overall": {}, "defect_types": {}}
    all_scores = []
    all_region_counts = []
    for category, category_records in by_category.items():
        image_scores = [record["image_score"] for record in category_records]
        region_counts = [record["region_count"] for record in category_records]
        summary["categories"][category] = {
            "num_samples": len(category_records),
            "mean_image_score": float(np.mean(image_scores)),
            "std_image_score": float(np.std(image_scores)),
            "mean_region_count": float(np.mean(region_counts)),
            "positive_prediction_rate": float(np.mean([record["predicted_label"] for record in category_records])),
            "mean_largest_region_area_fraction": float(
                np.mean([record["largest_region_area_fraction"] for record in category_records])
            ),
            "mean_total_region_area_fraction": float(
                np.mean([record["total_region_area_fraction"] for record in category_records])
            ),
        }
        defect_breakdown: Dict[str, Dict[str, float]] = {}
        by_defect_type: Dict[str, List[Dict]] = defaultdict(list)
        for record in category_records:
            by_defect_type[record["defect_type"]].append(record)
        for defect_type, defect_records in by_defect_type.items():
            defect_breakdown[defect_type] = {
                "num_samples": len(defect_records),
                "mean_image_score": float(np.mean([record["image_score"] for record in defect_records])),
                "positive_prediction_rate": float(np.mean([record["predicted_label"] for record in defect_records])),
                "mean_region_count": float(np.mean([record["region_count"] for record in defect_records])),
            }
        summary["defect_types"][category] = defect_breakdown
        all_scores.extend(image_scores)
        all_region_counts.extend(region_counts)

    summary["overall"] = {
        "num_records": len(records),
        "mean_image_score": float(np.mean(all_scores)) if all_scores else 0.0,
        "std_image_score": float(np.std(all_scores)) if all_scores else 0.0,
        "mean_region_count": float(np.mean(all_region_counts)) if all_region_counts else 0.0,
    }
    return summary


def save_report(path: Path, summary: Dict) -> None:
    """Write one JSON report."""
    ensure_dir(path.parent)
    save_json(path, summary)
