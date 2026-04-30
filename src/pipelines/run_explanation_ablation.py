"""Explanation-level ablation pipeline.

Re-runs the symbolic rule engine over the *already-computed* per-image
region statistics that the main evaluation pipeline exports to
``per_sample_records.jsonl`` (and to its detailed sibling JSONL produced
here). Instead of re-running the visual model, this script:

  1. Reads predicted region area-fraction statistics from existing JSONL
     records (``total_region_area_fraction``, ``largest_region_area_fraction``,
     ``region_count``, ``normalized_score``).
  2. Sweeps the severity thresholds ``severity_high_area_fraction`` (tau_h)
     and ``severity_medium_area_fraction`` (tau_m).
  3. Reports, per setting and per category, the *severity distribution*
     and the fraction of images whose severity label is *unchanged* with
     respect to the default setting (tau_h=0.05, tau_m=0.015).

This isolates the symbolic layer from detector variability: a
mask-cleanup ablation alone cannot move severity or archetype much
because cleanup mostly preserves region area.

Outputs
-------
``severity_stability.csv`` : per-(category, tau_h, tau_m) table with
    severity counts (none/low/medium/high) and unchanged-fraction.
``severity_stability_summary.json`` : overall stability and
    macro-averaged unchanged fractions across the sweep grid.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

DEFAULT_TAU_H = 0.05
DEFAULT_TAU_M = 0.015

# Sweep grid: tight relative to the defaults to keep the analysis on the
# same operating regime. The default settings are included.
TAU_H_GRID = (0.03, 0.05, 0.08)
TAU_M_GRID = (0.005, 0.015, 0.030)


def _load_records(path: Path) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _assign_severity(total_area_fraction: float, region_count: int, tau_h: float, tau_m: float) -> str:
    if region_count == 0:
        return "none"
    if total_area_fraction >= tau_h:
        return "high"
    if total_area_fraction >= tau_m:
        return "medium"
    return "low"


def sweep_severity(records: Iterable[Dict]) -> Dict:
    """Return per-(category, tau_h, tau_m) severity distributions and stability vs default."""
    records = list(records)

    # Cache default severity per record key.
    default_sev_by_key: Dict[Tuple[str, str], str] = {}
    for rec in records:
        key = (rec["category"], rec["image_path"])
        default_sev_by_key[key] = _assign_severity(
            float(rec.get("total_region_area_fraction", 0.0)),
            int(rec.get("region_count", 0)),
            DEFAULT_TAU_H,
            DEFAULT_TAU_M,
        )

    per_setting: Dict[Tuple[str, float, float], Dict] = {}
    overall_unchanged: Dict[Tuple[float, float], List[int]] = defaultdict(list)

    categories = sorted({rec["category"] for rec in records})

    for tau_h in TAU_H_GRID:
        for tau_m in TAU_M_GRID:
            if tau_m >= tau_h:
                continue  # invalid (medium >= high makes no sense)
            for category in categories:
                cat_records = [rec for rec in records if rec["category"] == category]
                sev_counter: Counter = Counter()
                unchanged = 0
                total = 0
                for rec in cat_records:
                    new_sev = _assign_severity(
                        float(rec.get("total_region_area_fraction", 0.0)),
                        int(rec.get("region_count", 0)),
                        tau_h,
                        tau_m,
                    )
                    sev_counter[new_sev] += 1
                    key = (rec["category"], rec["image_path"])
                    if new_sev == default_sev_by_key[key]:
                        unchanged += 1
                    total += 1
                    overall_unchanged[(tau_h, tau_m)].append(
                        1 if new_sev == default_sev_by_key[key] else 0
                    )
                per_setting[(category, tau_h, tau_m)] = {
                    "tau_h": tau_h,
                    "tau_m": tau_m,
                    "category": category,
                    "num_samples": total,
                    "none": int(sev_counter["none"]),
                    "low": int(sev_counter["low"]),
                    "medium": int(sev_counter["medium"]),
                    "high": int(sev_counter["high"]),
                    "unchanged_fraction": unchanged / total if total else float("nan"),
                }

    summary: Dict[str, Dict] = {
        "default_tau_h": DEFAULT_TAU_H,
        "default_tau_m": DEFAULT_TAU_M,
        "tau_h_grid": list(TAU_H_GRID),
        "tau_m_grid": list(TAU_M_GRID),
        "overall_unchanged_fraction": {
            f"tau_h={th}_tau_m={tm}": (sum(vals) / len(vals)) if vals else float("nan")
            for (th, tm), vals in sorted(overall_unchanged.items())
        },
    }
    return {"per_setting": per_setting, "summary": summary}


def save_severity_csv(path: Path, per_setting: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "category", "tau_h", "tau_m",
        "num_samples", "none", "low", "medium", "high",
        "unchanged_fraction",
    ]
    rows = sorted(
        per_setting.values(),
        key=lambda r: (r["category"], r["tau_h"], r["tau_m"]),
    )
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in fieldnames})


def save_summary_json(path: Path, summary: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Explanation-level ablation over severity thresholds.")
    parser.add_argument("--records", type=Path, required=True,
                        help="Path to per_sample_records.jsonl from a completed evaluation run.")
    parser.add_argument("--out-csv", type=Path, required=True,
                        help="Output CSV: per-(category, tau_h, tau_m) severity counts and stability.")
    parser.add_argument("--out-summary", type=Path, required=True,
                        help="Output JSON: overall macro stability per sweep setting.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = _load_records(args.records)
    result = sweep_severity(records)
    save_severity_csv(args.out_csv, result["per_setting"])
    save_summary_json(args.out_summary, result["summary"])
    print(f"Severity stability table: {args.out_csv}")
    print(f"Severity stability summary: {args.out_summary}")


if __name__ == "__main__":
    main()
