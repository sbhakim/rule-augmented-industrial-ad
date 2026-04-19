"""Load and save project-level experiment configuration files."""

import json
from pathlib import Path
from typing import Optional

from .settings import ExperimentConfig, default_experiment_config


def default_config_path() -> Path:
    """Return the canonical project config path."""
    return Path(__file__).resolve().parents[2] / "configs" / "default_experiment.json"


def load_experiment_config(path: Optional[Path] = None) -> ExperimentConfig:
    """Load an experiment configuration from JSON."""
    target = path or default_config_path()
    with open(target, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return ExperimentConfig.from_dict(payload)


def save_experiment_config(config: ExperimentConfig, path: Optional[Path] = None) -> Path:
    """Save an experiment configuration to JSON."""
    target = path or default_config_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as handle:
        json.dump(config.to_dict(), handle, indent=2, sort_keys=True)
    return target


def bootstrap_default_config(path: Optional[Path] = None) -> Path:
    """Write the default config if it does not exist yet."""
    target = path or default_config_path()
    if not target.exists():
        save_experiment_config(default_experiment_config(), target)
    return target
