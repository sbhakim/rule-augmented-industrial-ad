"""Symbolic reasoning layer over post-processed anomaly regions.

This module is the interpretive half of the framework. The visual side
produces a binary mask and a list of connected-component descriptors;
this module turns that geometric summary into a small set of human-
readable labels: a severity bucket, a defect archetype, a confidence
score, an anomaly-map quality flag, and a list of spatial relations
between regions. The output is consumed by the reporting layer and
exported per sample.

The engine is deliberately rule-based rather than learned. All
thresholds and category priors are exposed as constructor arguments
or read from ``priors.py``, so a future user can swap heuristics, add
archetypes, or replace this module entirely without touching the
detector or the reporting code.
"""

from dataclasses import dataclass, field
from typing import List

from ..features.region_features import RegionFeatures
from .priors import CATEGORY_PRIORS


@dataclass
class RuleResult:
    """Container for the per-image symbolic output.

    Fields are intentionally flat so each can be serialized directly
    into the per-sample CSV/JSONL exports without further reshaping.
    ``tags`` holds the geometric descriptors the engine derived from
    region features; ``archetype`` is a single label resolved from
    those tags; ``spatial_relations`` is empty when only one region
    survives cleanup.
    """

    severity: str
    archetype: str
    tags: List[str]
    confidence: float = 1.0
    spatial_relations: List[str] = field(default_factory=list)
    quality_flag: str = "normal"


class RuleEngine:
    """Stateless symbolic layer over anomaly regions.

    The engine carries only a handful of scalar thresholds; it holds
    no per-image state. ``analyze`` is the single entry point and is
    safe to call from a tight evaluation loop.
    """

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
        """Map a list of region descriptors to a single ``RuleResult``.

        Callers pass the cleaned region set (largest first) together
        with the image-level score normalized by its threshold. When
        the region set is empty the engine returns a designated
        ``no_salient_region`` result so the per-sample record for a
        clean image still carries a valid quality flag.
        """
        if not regions:
            return RuleResult(
                severity="none",
                archetype="no_salient_region",
                tags=["no_region"],
                confidence=1.0,
                quality_flag=self._quality_flag(normalized_score),
            )

        prior = CATEGORY_PRIORS[category]
        # The pipeline guarantees regions are sorted by descending
        # area, so ``regions[0]`` is the dominant defect candidate.
        # All single-region geometric tags are derived from it.
        largest = regions[0]
        tags: List[str] = []

        # Severity is driven by total anomalous area fraction rather
        # than by the largest region alone, so a category dominated
        # by many small defects is not under-reported.
        total_area_fraction = sum(r.area_fraction for r in regions)
        if total_area_fraction >= self.severity_high_area_fraction:
            severity = "high"
        elif total_area_fraction >= self.severity_medium_area_fraction:
            severity = "medium"
        else:
            severity = "low"

        # Geometric tags. Each tag is independent and is consumed
        # later by ``_resolve_archetype``, which picks a single label
        # in priority order. New archetypes can be added by appending
        # a tag here and a branch in ``_resolve_archetype``.
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
        """Composite confidence in [0, 1] reported alongside each prediction.

        Three coarse signals are combined into a single triage value:
        how far the image score sits above its detection threshold,
        how much anomalous area was actually flagged, and how cleanly
        the flagged pixels partitioned into regions. The combination
        is hand-weighted, not learned, and is intended as a reviewer-
        facing summary rather than a calibrated probability.
        """
        # Score-margin contribution: a clipped linear ramp on the
        # normalized score. The lower anchor prevents near-threshold
        # predictions from being reported as confident even when the
        # spatial evidence looks clean; the upper saturation keeps
        # very high scores from drowning out the area and clarity
        # terms below.
        score_conf = min(1.0, max(0.0, (normalized_score - 0.5) / 1.5))

        # Area contribution: anomalous area normalized against the
        # medium-severity cutoff. Detections already at or above
        # medium severity contribute the full term; smaller
        # detections are discounted linearly so a single tiny
        # region cannot drive confidence on its own.
        area_conf = min(1.0, total_area_fraction / self.severity_medium_area_fraction)

        # Region-clarity contribution: a single dominant region is
        # treated as the cleanest evidence; a small handful of regions
        # is accepted at a reduced level; many small regions are read
        # as fragmented evidence and capped further.
        if len(regions) == 1 and regions[0].area_fraction >= self.severity_medium_area_fraction:
            clarity = 1.0
        elif len(regions) <= 3:
            clarity = 0.7
        else:
            clarity = 0.5

        return round(score_conf * 0.5 + area_conf * 0.3 + clarity * 0.2, 3)

    def _quality_flag(self, normalized_score: float) -> str:
        """Coarse reliability bucket for the underlying anomaly map.

        Distinct from ``confidence`` in that it does not look at
        region geometry at all. The flag is useful when an operator
        needs a single, fast signal for whether the upstream detector
        was committal on this image.
        """
        if normalized_score >= 2.0:
            return "high_confidence"
        if normalized_score >= 1.0:
            return "near_threshold"
        if normalized_score >= 0.7:
            return "weak_signal"
        return "below_threshold"

    def _spatial_analysis(self, regions: List[RegionFeatures]) -> List[str]:
        """Tag inter-region spatial patterns when more than one region survives.

        Returns a small list of qualitative descriptors (alignment,
        border behaviour, clustered vs. scattered). Single-region
        cases short-circuit because none of these descriptors carry
        meaning with one region.
        """
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
        """Collapse the active tag set into a single archetype label.

        The branches are checked in priority order: more structurally
        specific patterns (border, elongated/thin) take precedence
        over coarser ones (large, distributed). A small set of
        categories with thread-like geometry is routed to a
        dedicated archetype to keep the labels physically meaningful.
        """
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
