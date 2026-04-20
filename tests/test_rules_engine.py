from src.features.region_features import RegionFeatures
from src.rules.engine import RuleEngine


DEFAULT_PARAMS = dict(
    severity_high_area_fraction=0.05,
    severity_medium_area_fraction=0.015,
    elongated_aspect_ratio=3.0,
    thin_fill_ratio=0.15,
    distributed_region_count=3,
)


def _region(
    area_fraction=0.02,
    aspect_ratio=1.5,
    fill_ratio=0.6,
    centroid_yx=(128.0, 128.0),
    touches_border=False,
    label=1,
):
    return RegionFeatures(
        label=label,
        area_px=int(area_fraction * 65536),
        area_fraction=area_fraction,
        bbox=(0, 0, 10, 10),
        aspect_ratio=aspect_ratio,
        fill_ratio=fill_ratio,
        centroid_yx=centroid_yx,
        touches_border=touches_border,
    )


def _engine():
    return RuleEngine(**DEFAULT_PARAMS)


def test_no_regions_yields_no_salient_region_archetype():
    result = _engine().analyze("bottle", [], normalized_score=0.4)
    assert result.archetype == "no_salient_region"
    assert result.severity == "none"
    assert result.tags == ["no_region"]
    assert result.confidence == 1.0
    assert result.quality_flag == "below_threshold"


def test_severity_low_below_medium_fraction():
    result = _engine().analyze("hazelnut", [_region(area_fraction=0.005)])
    assert result.severity == "low"


def test_severity_medium_at_medium_threshold():
    result = _engine().analyze("hazelnut", [_region(area_fraction=0.02)])
    assert result.severity == "medium"


def test_severity_high_at_high_threshold():
    result = _engine().analyze("hazelnut", [_region(area_fraction=0.06, fill_ratio=0.2)])
    assert result.severity == "high"


def test_border_contact_only_fires_on_border_sensitive_category():
    engine = _engine()
    touching = _region(touches_border=True)
    assert "border_contact" not in engine.analyze("bottle", [touching]).tags
    assert "border_contact" in engine.analyze("metal_nut", [touching]).tags


def test_border_contact_archetype_takes_priority_over_other_tags():
    r = _region(touches_border=True, aspect_ratio=4.0, fill_ratio=0.1, area_fraction=0.06)
    result = _engine().analyze("zipper", [r])
    assert result.archetype == "border_localized_anomaly"


def test_elongated_thin_maps_to_thread_like_for_thread_categories():
    r = _region(aspect_ratio=4.0, fill_ratio=0.1)
    result = _engine().analyze("cable", [r])
    assert "elongated" in result.tags
    assert "thin_structure" in result.tags
    assert result.archetype == "structural_thread_like_anomaly"


def test_elongated_thin_maps_to_surface_for_non_thread_categories():
    r = _region(aspect_ratio=4.0, fill_ratio=0.1)
    result = _engine().analyze("hazelnut", [r])
    assert result.archetype == "elongated_surface_anomaly"


def test_large_region_archetype():
    r = _region(area_fraction=0.06, aspect_ratio=1.0, fill_ratio=0.3)
    result = _engine().analyze("hazelnut", [r])
    assert "large_region" in result.tags
    assert result.archetype == "large_contiguous_break_like_anomaly"


def test_distributed_archetype_when_three_or_more_regions():
    regions = [
        _region(area_fraction=0.005, centroid_yx=(50.0, 50.0), label=1),
        _region(area_fraction=0.005, centroid_yx=(128.0, 128.0), label=2),
        _region(area_fraction=0.005, centroid_yx=(200.0, 200.0), label=3),
    ]
    result = _engine().analyze("hazelnut", regions)
    assert "distributed" in result.tags
    assert result.archetype == "distributed_multi_region_anomaly"


def test_contamination_like_archetype_only_for_contamination_prior():
    r = _region(area_fraction=0.02, aspect_ratio=1.5, fill_ratio=0.5)
    result_bottle = _engine().analyze("bottle", [r])
    assert "contamination_like" in result_bottle.tags
    assert result_bottle.archetype == "compact_contamination_like_anomaly"

    result_hazelnut = _engine().analyze("hazelnut", [r])
    assert "contamination_like" not in result_hazelnut.tags


def test_default_archetype_is_localized_surface():
    r = _region(area_fraction=0.005, aspect_ratio=1.5, fill_ratio=0.3)
    result = _engine().analyze("hazelnut", [r])
    assert result.archetype == "localized_surface_anomaly"


def test_quality_flag_tier_boundaries():
    engine = _engine()
    assert engine.analyze("bottle", [], normalized_score=2.5).quality_flag == "high_confidence"
    assert engine.analyze("bottle", [], normalized_score=1.5).quality_flag == "near_threshold"
    assert engine.analyze("bottle", [], normalized_score=0.8).quality_flag == "weak_signal"
    assert engine.analyze("bottle", [], normalized_score=0.5).quality_flag == "below_threshold"


def test_vertically_aligned_relation():
    regions = [
        _region(centroid_yx=(50.0, 128.0), label=1),
        _region(centroid_yx=(200.0, 128.0), label=2),
    ]
    result = _engine().analyze("hazelnut", regions)
    assert "vertically_aligned" in result.spatial_relations


def test_horizontally_aligned_relation():
    regions = [
        _region(centroid_yx=(128.0, 50.0), label=1),
        _region(centroid_yx=(128.0, 200.0), label=2),
    ]
    result = _engine().analyze("hazelnut", regions)
    assert "horizontally_aligned" in result.spatial_relations


def test_all_border_regions_relation():
    regions = [
        _region(touches_border=True, centroid_yx=(50.0, 50.0), label=1),
        _region(touches_border=True, centroid_yx=(200.0, 200.0), label=2),
    ]
    result = _engine().analyze("hazelnut", regions)
    assert "all_border_regions" in result.spatial_relations


def test_mixed_border_interior_relation():
    regions = [
        _region(touches_border=True, centroid_yx=(50.0, 50.0), label=1),
        _region(touches_border=False, centroid_yx=(128.0, 128.0), label=2),
    ]
    result = _engine().analyze("hazelnut", regions)
    assert "mixed_border_interior" in result.spatial_relations


def test_clustered_vs_scattered_relation():
    clustered = [
        _region(centroid_yx=(120.0, 120.0), label=1),
        _region(centroid_yx=(130.0, 130.0), label=2),
        _region(centroid_yx=(125.0, 125.0), label=3),
    ]
    scattered = [
        _region(centroid_yx=(20.0, 20.0), label=1),
        _region(centroid_yx=(240.0, 240.0), label=2),
        _region(centroid_yx=(30.0, 200.0), label=3),
    ]
    assert "clustered" in _engine().analyze("hazelnut", clustered).spatial_relations
    assert "scattered" in _engine().analyze("hazelnut", scattered).spatial_relations


def test_identical_inputs_yield_identical_outputs():
    r = _region(area_fraction=0.03, aspect_ratio=2.0, fill_ratio=0.5)
    engine = _engine()
    a = engine.analyze("pill", [r], normalized_score=1.4)
    b = engine.analyze("pill", [r], normalized_score=1.4)
    assert a == b


def test_confidence_is_bounded_unit_interval():
    r = _region(area_fraction=0.05, aspect_ratio=2.0, fill_ratio=0.6)
    for score in (0.0, 0.5, 1.0, 2.0, 10.0):
        confidence = _engine().analyze("bottle", [r], normalized_score=score).confidence
        assert 0.0 <= confidence <= 1.0
