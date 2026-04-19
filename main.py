"""Top-level entry point for the IECON 2026 inspection project."""

import sys
from pathlib import Path


def _bootstrap_project_root() -> None:
    """Ensure the project root is importable when running main.py directly."""
    project_root = Path(__file__).resolve().parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))


def main() -> None:
    """Delegate to the config-driven experiment runner."""
    _bootstrap_project_root()
    from src.pipelines.run_experiment import main as run_experiment_main
    run_experiment_main()


if __name__ == "__main__":
    main()
