"""Connected-component extraction for binary anomaly masks."""

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from scipy import ndimage


@dataclass
class Component:
    """One connected component."""

    label: int
    coordinates: np.ndarray
    bbox: Tuple[int, int, int, int]


def label_components(binary_mask: np.ndarray) -> List[Component]:
    """Extract 8-connected components from a binary mask using scipy."""
    mask = binary_mask.astype(bool)
    structure = ndimage.generate_binary_structure(2, 2)  # 8-connectivity
    labeled_array, num_features = ndimage.label(mask, structure=structure)

    components: List[Component] = []
    for i in range(1, num_features + 1):
        coords = np.argwhere(labeled_array == i)
        if len(coords) == 0:
            continue
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        components.append(
            Component(
                label=i,
                coordinates=coords.astype(np.int32),
                bbox=(int(y_min), int(x_min), int(y_max), int(x_max)),
            )
        )

    return components
