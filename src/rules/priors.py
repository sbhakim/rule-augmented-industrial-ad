"""Category-specific explanation priors."""

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class CategoryPrior:
    """Coarse structural prior for one category."""

    category: str
    object_family: str
    border_sensitive: bool
    prefers_elongated_reasoning: bool
    contamination_possible: bool


CATEGORY_PRIORS: Dict[str, CategoryPrior] = {
    "bottle": CategoryPrior("bottle", "rigid_container", False, False, True),
    "cable": CategoryPrior("cable", "flexible_thread_like_object", False, True, False),
    "capsule": CategoryPrior("capsule", "printed_object", False, False, False),
    "carpet": CategoryPrior("carpet", "texture", False, False, True),
    "grid": CategoryPrior("grid", "texture", False, True, True),
    "hazelnut": CategoryPrior("hazelnut", "natural_object", False, False, False),
    "leather": CategoryPrior("leather", "texture", False, False, False),
    "metal_nut": CategoryPrior("metal_nut", "rigid_object", True, False, False),
    "pill": CategoryPrior("pill", "printed_object", False, False, True),
    "screw": CategoryPrior("screw", "threaded_object", False, True, False),
    "tile": CategoryPrior("tile", "texture", False, False, True),
    "toothbrush": CategoryPrior("toothbrush", "bristled_object", False, True, False),
    "transistor": CategoryPrior("transistor", "printed_component", True, False, False),
    "wood": CategoryPrior("wood", "texture", False, False, False),
    "zipper": CategoryPrior("zipper", "bordered_structural_object", True, True, False),
}
