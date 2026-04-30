"""Top-level CLI entry point that fits and then evaluates.

This is the recommended way to run the framework end-to-end from
a single JSON config. It exists as a thin wrapper over the two
pipelines so users can invoke either stage independently from
their own scripts when they need finer control.
"""

import argparse
from pathlib import Path

from ..config.loader import load_experiment_config
from ..pipelines.evaluate import evaluate_categories
from ..pipelines.fit import fit_categories
from ..utils.io import save_json


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Run the MVTec rule-augmented anomaly pipeline.")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to a JSON experiment config. Defaults to configs/default_experiment.json.",
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=None,
        help="Optional explicit path for a duplicate evaluation summary JSON.",
    )
    return parser.parse_args()


def main() -> None:
    """Fit then evaluate the selected categories."""
    args = parse_args()
    config = load_experiment_config(args.config)
    fit_categories(config)
    summary = evaluate_categories(config)
    if args.summary_out is not None:
        save_json(args.summary_out, summary)


if __name__ == "__main__":
    main()
