"""Command-style entry point for dataset inspection and audit export."""

import argparse
from pathlib import Path

from ..config.loader import load_experiment_config
from ..data.audit import audit_indexed_dataset
from ..data.indexer import index_dataset
from ..utils.io import save_json


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Audit the selected MVTec dataset subset.")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to a JSON experiment config. Defaults to configs/default_experiment.json.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional explicit JSON output path for the audit summary.",
    )
    return parser.parse_args()


def main() -> None:
    """Inspect indexed samples and save the dataset audit summary."""
    args = parse_args()
    config = load_experiment_config(args.config)
    indexed = index_dataset(config.dataset.root, config.dataset.categories)
    audit_summary = audit_indexed_dataset(indexed)
    target = args.output or (config.outputs.reports_dir / "dataset_audit.json")
    save_json(target, audit_summary)


if __name__ == "__main__":
    main()
