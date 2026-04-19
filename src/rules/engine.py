"""Rule engine that turns anomaly regions into interpretable reasoning tags."""

from dataclasses import dataclass, field
from typing import List

from ..features.region_features import RegionFeatures
from .priors import CATEGORY_PRIORS


@dataclass
class RuleResult:
    """Structured symbolic reasoning output."""

    severity: str
    archetype: str
    tags: List[str]
    confidence: float = 1.0
    spatial_relations: List[str] = field(default_factory=list)
    quality_flag: str = "normal"


class RuleEngine:
    """Symbolic layer over anomaly regions with confidence and spatial reasoning."""

    def __init__(
        self,
        severity_high_area_fraction: float,
        severity_medium_area_fraction: float,
        elongated_aspect_ratio: float,
        thin_fill_ratio: float,
        distributed_region_count: int,
    ):
        self.severity_high_area_fraction = severity_high_area_fraction
        self.severity_medium_area_fraction = severity_medium_area_fraction
        self.elongated_aspect_ratio = elongated_aspect_ratio
        self.thin_fill_ratio = thin_fill_ratio
        self.distributed_region_count = distributed_region_count

    def analyze(
        self,
        category: str,
        regions: List[RegionFeatures],
        normalized_score: float = 1.0,
    ) -> RuleResult:
        if not regions:
            return RuleResult(
                severity="none",
                archetype="no_salient_region",
                tags=["no_region"],
                confidence=1.0,
                quality_flag=self._quality_flag(normalized_score),
            )

        prior = CATEGORY_PRIORS[category]
        largest = regions[0]
        tags: List[str] = []

        # --- Severity ---
        total_area_fraction = sum(r.area_fraction for r in regions)
        if total_area_fraction >= self.severity_high_area_fraction:
            severity = "high"
        elif total_area_fraction >= self.severity_medium_area_fraction:
            severity = "medium"
        else:
            severity = "low"

        # --- Geometric tags ---
        if len(regions) >= self.distributed_region_count:
            tags.append("distributed")
        if largest.touches_border and prior.border_sensitive:
            tags.append("border_contact")
        if largest.aspect_ratio >= self.elongated_aspect_ratio:
            tags.append("elongated")
        if largest.fill_ratio <= self.thin_fill_ratio:
            tags.append("thin_structure")
        if largest.area_fraction >= self.severity_high_area_fraction:
            tags.append("large_region")
        if largest.fill_ratio >= 0.55 and largest.area_fraction <= self.severity_medium_area_fraction:
            tags.append("compact_localized")
        if prior.contamination_possible and largest.fill_ratio >= 0.45:
            tags.append("contamination_like")

        # --- Multi-region spatial reasoning ---
        spatial_relations = self._spatial_analysis(regions)

        # --- Confidence scoring ---
        confidence = self._compute_confidence(
            regions, total_area_fraction, normalized_score, severity
        )

        # --- Quality flag ---
        quality_flag = self._quality_flag(normalized_score)

        archetype = self._resolve_archetype(category, tags)
        return RuleResult(
            severity=severity,
            archetype=archetype,
            tags=tags,
            confidence=confidence,
            spatial_relations=spatial_relations,
            quality_flag=quality_flag,
        )

    def _compute_confidence(
        self,
        regions: List[RegionFeatures],
        total_area_fraction: float,
        normalized_score: float,
        severity: str,
    ) -> float:
        """Score how confident the symbolic explanation is.

        Confidence is high when:
          - the image score is well above threshold (normalized >> 1)
          - the anomalous area is substantial
          - the region geometry is unambiguous
        Confidence is low when the score is near-threshold or regions are tiny.
        """
        # Score factor: sigmoid-like ramp around threshold (normalized_score=1.0)
        score_conf = min(1.0, max(0.0, (normalized_score - 0.5) / 1.5))

        # Area factor: more anomalous area -> more trustworthy explanation
        area_conf = min(1.0, total_area_fraction / self.severity_medium_area_fraction)

        # Region clarity: single large region is clearer than many tiny ones
        if len(regions) == 1 and regions[0].area_fraction >= self.severity_medium_area_fraction:
            clarity = 1.0
        elif len(regions) <= 3:
            clarity = 0.7
        else:
            clarity = 0.5

        return round(score_conf * 0.5 + area_conf * 0.3 + clarity * 0.2, 3)

    def _quality_flag(self, normalized_score: float) -> str:
        """Flag the reliability of the anomaly map for this prediction."""
        if normalized_score >= 2.0:
            return "high_confidence"
        if normalized_score >= 1.0:
            return "near_threshold"
        if normalized_score >= 0.7:
            return "weak_signal"
        return "below_threshold"

    def _spatial_analysis(self, regions: List[RegionFeatures]) -> List[str]:
        """Analyze spatial relationships between multiple detected regions."""
        relations: List[str] = []
        if len(regions) < 2:
            return relations

        centroids = [r.centroid_yx for r in regions]

        # Check if regions are vertically or horizontally aligned
        ys = [c[0] for c in centroids]
        xs = [c[1] for c in centroids]
        y_spread = max(ys) - min(ys)
        x_spread = max(xs) - min(xs)

        if len(regions) >= 2:
            if y_spread > 0 and x_spread / max(y_spread, 1e-6) < 0.3:
                relations.append("vertically_aligned")
            elif x_spread > 0 and y_spread / max(x_spread, 1e-6) < 0.3:
                relations.append("horizontally_aligned")

        # Check if all regions touch the border
        border_count = sum(1 for r in regions if r.touches_border)
        if border_count == len(regions) and len(regions) >= 2:
            relations.append("all_border_regions")
        elif border_count > 0 and border_count < len(regions):
            relations.append("mixed_border_interior")

        # Check if regions are clustered vs scattered
        if len(regions) >= 3:
            mean_y = sum(ys) / len(ys)
            mean_x = sum(xs) / len(xs)
            avg_dist = sum(
                ((y - mean_y) ** 2 + (x - mean_x) ** 2) ** 0.5
                for y, x in zip(ys, xs)
            ) / len(regions)
            # Normalize by image diagonal (assume 256x256)
            norm_dist = avg_dist / 362.0
            if norm_dist < 0.15:
                relations.append("clustered")
            else:
                relations.append("scattered")

        return relations

    def _resolve_archetype(self, category: str, tags: List[str]) -> str:
        if "border_contact" in tags:
            return "border_localized_anomaly"
        if "elongated" in tags and "thin_structure" in tags:
            if category in {"cable", "grid", "screw", "toothbrush", "zipper"}:
                return "structural_thread_like_anomaly"
            return "elongated_surface_anomaly"
        if "large_region" in tags:
            return "large_contiguous_break_like_anomaly"
        if "distributed" in tags:
            return "distributed_multi_region_anomaly"
        if "contamination_like" in tags:
            return "compact_contamination_like_anomaly"
        return "localized_surface_anomaly"
