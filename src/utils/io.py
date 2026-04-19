"""Lightweight IO and reproducibility helpers."""

import json
import random
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def ensure_dir(path: Path) -> Path:
    """Create a directory if it does not exist."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path: Path, payload: Any) -> None:
    """Write JSON with stable formatting."""
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def save_json_lines(path: Path, payloads: Iterable[Any]) -> None:
    """Write JSONL with one compact object per line."""
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as handle:
        for payload in payloads:
            handle.write(json.dumps(payload, sort_keys=True))
            handle.write("\n")


def set_global_seed(seed: int) -> None:
    """Seed Python and NumPy."""
    random.seed(seed)
    np.random.seed(seed)
