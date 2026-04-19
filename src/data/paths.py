"""Project path discovery."""

from pathlib import Path


def src_root() -> Path:
    return Path(__file__).resolve().parents[1]


def project_root() -> Path:
    return src_root().parent


def data_root() -> Path:
    return project_root() / "data" / "mvtec_ad"


def outputs_root() -> Path:
    return project_root() / "outputs"
