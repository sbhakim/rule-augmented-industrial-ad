"""Stand-alone ablation that varies only the symbolic layer.

The main evaluation pipeline exports a JSONL of per-sample records
that already contain the geometric summaries the rule engine
consumes (``total_region_area_fraction``, ``region_count``, and so
on). This script reuses those records, sweeps the severity area
thresholds across a small grid, and reports how often the severity
label changes relative to the default operating point.

The point of running an ablation in this form is to isolate the
symbolic layer's behaviour from any detector variability: no model
is reloaded, no image is re-read, and the geometric inputs are
fixed. A user adapting this script to study other rule constants
can copy the same pattern --- read once, mutate the symbolic
function, count how labels move.

Outputs
-------
``severity_stability.csv`` : per-(category, tau_h, tau_m) table
    with severity counts and an unchanged-fraction column.
``severity_stability_summary.json`` : aggregate unchanged-fraction
    per setting, plus the default thresholds used as the reference
    operating point.
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
    """Replica of the rule engine's severity decision.

    Held local to this module so the ablation can vary the
    thresholds freely without instantiating the full ``RuleEngine``.
    The decision logic must stay in lockstep with
    ``RuleEngine.analyze``; if the rule engine's severity branch
    ever changes shape, this function should change with it.
    """
    if region_count == 0:
        return "none"
    if total_area_fraction >= tau_h:
        return "high"
    if total_area_fraction >= tau_m:
        return "medium"
    return "low"


def sweep_severity(records: Iterable[Dict]) -> Dict:
    """Apply every grid setting to the record stream and tabulate stability.

    The reference labels are computed once at the default operating
    point, then each grid setting is compared against that
    reference image-by-image. The output covers per-category,
    per-setting severity counts so a downstream user can build
    distribution plots in addition to the headline stability
    fraction.
    """
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
