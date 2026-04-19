"""Geometric features derived from binary anomaly regions."""

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from .connected_components import label_components


@dataclass
class RegionFeatures:
    """Summarized geometry for one anomaly region."""

    label: int
    area_px: int
    area_fraction: float
    bbox: tuple[int, int, int, int]
    aspect_ratio: float
    fill_ratio: float
    centroid_yx: tuple[float, float]
    touches_border: bool


def summarize_regions(binary_mask: np.ndarray, min_region_area_px: int, border_margin_px: int) -> List[RegionFeatures]:
    """Convert connected components into region feature records."""
    height, width = binary_mask.shape
    regions: List[RegionFeatures] = []

    for component in label_components(binary_mask):
        ys = component.coordinates[:, 0]
        xs = component.coordinates[:, 1]
        area_px = len(component.coordinates)
        if area_px < min_region_area_px:
            continue

        y_min, x_min, y_max, x_max = component.bbox
        bbox_height = y_max - y_min + 1
        bbox_width = x_max - x_min + 1
        bbox_area = max(1, bbox_height * bbox_width)
        aspect_ratio = max(bbox_height, bbox_width) / max(1, min(bbox_height, bbox_width))
        fill_ratio = area_px / bbox_area
        touches_border = (
            y_min <= border_margin_px
            or x_min <= border_margin_px
            or y_max >= (height - border_margin_px - 1)
            or x_max >= (width - border_margin_px - 1)
        )
        regions.append(
            RegionFeatures(
                label=component.label,
                area_px=area_px,
                area_fraction=area_px / float(height * width),
                bbox=component.bbox,
                aspect_ratio=float(aspect_ratio),
                fill_ratio=float(fill_ratio),
                centroid_yx=(float(ys.mean()), float(xs.mean())),
                touches_border=bool(touches_border),
            )
        )

    regions.sort(key=lambda region: region.area_px, reverse=True)
    return regions


def region_summary_dict(regions: List[RegionFeatures]) -> Dict[str, float]:
    """Compute coarse aggregate region statistics."""
    if not regions:
        return {"region_count": 0, "largest_area_fraction": 0.0, "total_area_fraction": 0.0}
    return {
        "region_count": float(len(regions)),
        "largest_area_fraction": float(regions[0].area_fraction),
        "total_area_fraction": float(sum(region.area_fraction for region in regions)),
    }
