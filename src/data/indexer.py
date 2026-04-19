"""Index MVTec samples into structured records."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass(frozen=True)
class MVTecSample:
    """One indexed MVTec sample."""

    category: str
    split: str
    label: int
    defect_type: str
    image_path: Path
    mask_path: Optional[Path]


def _index_split(category_root: Path, category: str, split: str) -> List[MVTecSample]:
    samples: List[MVTecSample] = []
    split_root = category_root / split
    gt_root = category_root / "ground_truth"

    for defect_dir in sorted(p for p in split_root.iterdir() if p.is_dir()):
        defect_type = defect_dir.name
        label = 0 if defect_type == "good" else 1
        for image_path in sorted(defect_dir.glob("*.png")):
            if label == 0:
                mask_path = None
            else:
                mask_path = gt_root / defect_type / f"{image_path.stem}_mask.png"
            samples.append(
                MVTecSample(
                    category=category,
                    split=split,
                    label=label,
                    defect_type=defect_type,
                    image_path=image_path,
                    mask_path=mask_path,
                )
            )
    return samples


def index_category(root: Path, category: str) -> Dict[str, List[MVTecSample]]:
    """Index one category into train/test sample lists."""
    category_root = root / category
    return {
        "train": _index_split(category_root, category, "train"),
        "test": _index_split(category_root, category, "test"),
    }


def index_dataset(root: Path, categories: List[str]) -> Dict[str, Dict[str, List[MVTecSample]]]:
    """Index multiple categories."""
    return {category: index_category(root, category) for category in categories}
