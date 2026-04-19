"""Runtime helpers for reproducible experiment directories and metadata."""

from datetime import datetime
from pathlib import Path
from typing import Dict

from ..config.settings import ExperimentConfig
from .io import ensure_dir, save_json


def build_run_name(experiment_name: str) -> str:
    """Create a timestamped run name."""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{experiment_name}_{timestamp}"


def prepare_run_directories(config: ExperimentConfig) -> Dict[str, Path]:
    """Create the directory set for one experiment run."""
    run_root = ensure_dir(config.outputs.run_metadata_dir / build_run_name(config.runtime.experiment_name))
    paths = {
        "root": run_root,
        "config": run_root / "resolved_config.json",
        "metadata": run_root / "run_metadata.json",
        "records": ensure_dir(run_root / "records"),
        "reports": ensure_dir(run_root / "reports"),
        "plots": ensure_dir(run_root / "plots"),
    }
    return paths


def save_run_metadata(path: Path, payload: Dict) -> None:
    """Write run metadata as stable JSON."""
    save_json(path, payload)
