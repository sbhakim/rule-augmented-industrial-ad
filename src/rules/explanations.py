"""Human-readable explanation generation."""

from typing import List

from .engine import RuleResult


def compose_explanation(category: str, rule_result: RuleResult, region_count: int) -> str:
    """Turn rule outputs into a short explanation string."""
    if rule_result.archetype == "no_salient_region":
        quality = f" [{rule_result.quality_flag}]" if rule_result.quality_flag != "normal" else ""
        return f"{category}: no salient anomaly region was detected after thresholding.{quality}"

    fragments: List[str] = [f"{category}: {rule_result.archetype.replace('_', ' ')}"]
    fragments.append(f"severity={rule_result.severity}")
    fragments.append(f"regions={region_count}")
    fragments.append(f"confidence={rule_result.confidence:.2f}")
    if rule_result.tags:
        fragments.append("tags=" + ",".join(rule_result.tags))
    if rule_result.spatial_relations:
        fragments.append("spatial=" + ",".join(rule_result.spatial_relations))
    fragments.append(f"quality={rule_result.quality_flag}")
    return "; ".join(fragments)
