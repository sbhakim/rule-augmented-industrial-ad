"""Cross-backend agreement diagnostic for the symbolic layer.

Two backends produce two independent JSONL exports for the same
test set. This module joins those exports on (category, image
path) and reports the fraction of shared images on which the two
backends assigned the same archetype or severity. Two granularities
are computed: across all shared images, and restricted to the
images on which both backends predicted positive (the "agreed
positives" view), which removes the cases where one backend
declared no salient region.

The diagnostic is meant for repository users who plug in a new
detector and want a quick read on whether their swap propagates
into report-level label changes.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def _load_records(path: Path) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _index_records(records: Iterable[Dict]) -> Dict[Tuple[str, str], Dict]:
    indexed: Dict[Tuple[str, str], Dict] = {}
    for record in records:
        key = (record["category"], record["image_path"])
        indexed[key] = record
    return indexed


def compare_runs(baseline_path: Path, feature_path: Path) -> Dict:
    """Join two record sets and tabulate per-category agreement.

    The two record files are not assumed to cover the same images;
    the comparison is done on the intersection keyed by (category,
    image path). Per-category and overall counts are returned in a
    single dictionary suitable for direct JSON serialization.
    """
    baseline_records = _load_records(baseline_path)
    feature_records = _load_records(feature_path)
    baseline_index = _index_records(baseline_records)
    feature_index = _index_records(feature_records)

    # Sorting the shared keys is purely for determinism: the
    # output ordering must not depend on the dict iteration order
    # of either input file.
    shared_keys = sorted(set(baseline_index).intersection(feature_index))
    per_category = defaultdict(lambda: {
        "samples": 0, "archetype_match": 0, "severity_match": 0,
        "both_positive": 0, "both_positive_archetype_match": 0, "both_positive_severity_match": 0,
    })

    for key in shared_keys:
        baseline = baseline_index[key]
        feature = feature_index[key]
        category = baseline["category"]
        per_category[category]["samples"] += 1
        if baseline.get("archetype") == feature.get("archetype"):
            per_category[category]["archetype_match"] += 1
        if baseline.get("severity") == feature.get("severity"):
            per_category[category]["severity_match"] += 1
        if int(baseline.get("predicted_label", 0)) == 1 and int(feature.get("predicted_label", 0)) == 1:
            per_category[category]["both_positive"] += 1
            if baseline.get("archetype") == feature.get("archetype"):
                per_category[category]["both_positive_archetype_match"] += 1
            if baseline.get("severity") == feature.get("severity"):
                per_category[category]["both_positive_severity_match"] += 1

    category_report: Dict[str, Dict] = {}
    for category, counts in per_category.items():
        samples = counts["samples"]
        bp = counts["both_positive"]
        category_report[category] = {
            "num_shared_samples": samples,
            "archetype_agreement": counts["archetype_match"] / samples if samples else float("nan"),
            "severity_agreement": counts["severity_match"] / samples if samples else float("nan"),
            "num_both_positive": bp,
            "archetype_agreement_on_agreed_positives": counts["both_positive_archetype_match"] / bp if bp else float("nan"),
            "severity_agreement_on_agreed_positives": counts["both_positive_severity_match"] / bp if bp else float("nan"),
        }

    total_samples = sum(c["samples"] for c in per_category.values())
    total_archetype = sum(c["archetype_match"] for c in per_category.values())
    total_severity = sum(c["severity_match"] for c in per_category.values())
    total_bp = sum(c["both_positive"] for c in per_category.values())
    total_bp_arch = sum(c["both_positive_archetype_match"] for c in per_category.values())
    total_bp_sev = sum(c["both_positive_severity_match"] for c in per_category.values())
    overall = {
        "num_shared_samples": total_samples,
        "archetype_agreement": total_archetype / total_samples if total_samples else float("nan"),
        "severity_agreement": total_severity / total_samples if total_samples else float("nan"),
        "num_both_positive": total_bp,
        "archetype_agreement_on_agreed_positives": total_bp_arch / total_bp if total_bp else float("nan"),
        "severity_agreement_on_agreed_positives": total_bp_sev / total_bp if total_bp else float("nan"),
    }

    return {
        "baseline_records": str(baseline_path),
        "feature_records": str(feature_path),
        "overall": overall,
        "per_category": category_report,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute cross-backend explanation consistency.")
    parser.add_argument("--baseline", type=Path, required=True, help="JSONL records from the statistics baseline.")
    parser.add_argument("--feature", type=Path, required=True, help="JSONL records from the feature backend.")
    parser.add_argument("--output", type=Path, required=True, help="Destination JSON path for the consistency report.")
    args = parser.parse_args()

    report = compare_runs(args.baseline, args.feature)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    print(f"Consistency report written to {args.output}")


if __name__ == "__main__":
    main()
