"""Dataset audit helpers for the selected MVTec subset."""

from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

from PIL import Image

from .indexer import MVTecSample


def _image_signature(path: Path) -> Dict[str, object]:
    """Read lightweight image metadata without loading the full pipeline."""
    with Image.open(path) as image:
        return {
            "size": [int(image.height), int(image.width)],
            "mode": image.mode,
        }


def summarize_split(samples: Iterable[MVTecSample]) -> Dict[str, object]:
    """Summarize one indexed split."""
    sample_list = list(samples)
    defect_counts = Counter(sample.defect_type for sample in sample_list)
    summary = {
        "num_samples": len(sample_list),
        "num_anomalous": int(sum(sample.label for sample in sample_list)),
        "defect_types": dict(sorted(defect_counts.items())),
    }
    if sample_list:
        summary["reference_image"] = _image_signature(sample_list[0].image_path)
    return summary


def audit_indexed_dataset(indexed: Dict[str, Dict[str, List[MVTecSample]]]) -> Dict[str, object]:
    """Produce a dataset audit summary over selected categories."""
    audit: Dict[str, object] = {"categories": {}, "overall": defaultdict(int)}

    for category, splits in indexed.items():
        category_summary = {}
        for split_name, split_samples in splits.items():
            split_summary = summarize_split(split_samples)
            category_summary[split_name] = split_summary
            audit["overall"]["num_samples"] += split_summary["num_samples"]
            audit["overall"]["num_anomalous"] += split_summary["num_anomalous"]
        audit["categories"][category] = category_summary

    audit["overall"] = dict(audit["overall"])
    audit["overall"]["num_categories"] = len(indexed)
    return audit
