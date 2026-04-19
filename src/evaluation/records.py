"""Per-sample record export utilities."""

import csv
from pathlib import Path
from typing import Dict, Iterable, List

from ..utils.io import ensure_dir, save_json_lines


CSV_FIELDS = [
    "category",
    "image_path",
    "label",
    "predicted_label",
    "defect_type",
    "image_score",
    "normalized_score",
    "score_threshold",
    "region_count",
    "largest_region_area_fraction",
    "total_region_area_fraction",
    "severity",
    "archetype",
    "confidence",
    "quality_flag",
    "spatial_relations",
    "explanation",
    "coverage",
]


def save_records_csv(path: Path, records: Iterable[Dict]) -> None:
    """Persist flat evaluation records as CSV."""
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            writer.writerow(record)


def save_records_jsonl(path: Path, records: List[Dict]) -> None:
    """Persist evaluation records as JSONL for downstream analysis."""
    save_json_lines(path, records)
