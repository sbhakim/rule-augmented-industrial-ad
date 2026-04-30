"""Per-category structural priors consumed by the rule engine.

Each category has a small set of qualitative traits (whether the
object family is texture or rigid, whether border defects are
physically meaningful, whether contamination-style defects occur,
and so on). The rule engine reads these flags to decide whether
to emit certain tags --- for example, the ``border_contact`` tag
is only meaningful for categories where the boundary is actually
the part being inspected.

Adding a new category means adding one entry to ``CATEGORY_PRIORS``
below; no other code change is required.
"""

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class CategoryPrior:
    """Coarse structural traits used to gate tag emission."""

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
