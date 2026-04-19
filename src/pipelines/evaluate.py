"""Full-dataset evaluation pipeline."""

from typing import Dict, List

import numpy as np

from ..config.loader import save_experiment_config
from ..config.settings import ExperimentConfig
from ..data.datasets import MVTecSubset
from ..data.indexer import index_dataset
from ..evaluation.records import save_records_csv, save_records_jsonl
from ..evaluation.metrics import auroc, classification_metrics, coverage_score, dice_score, iou_score, pro_score
from ..evaluation.reporting import aggregate_records, save_report
from ..features.postprocessing import cleanup_binary_mask
from ..features.region_features import summarize_regions
from ..models.factory import load_model
from ..rules.engine import RuleEngine
from ..rules.explanations import compose_explanation
from ..utils.io import ensure_dir, set_global_seed
from ..utils.runtime import prepare_run_directories, save_run_metadata


def evaluate_categories(config: ExperimentConfig) -> Dict:
    """Evaluate all categories and save a compact report."""
    set_global_seed(config.runtime.seed)
    ensure_dir(config.outputs.reports_dir)
    ensure_dir(config.outputs.records_dir)
    indexed = index_dataset(config.dataset.root, config.dataset.categories)
    run_paths = prepare_run_directories(config)
    save_experiment_config(config, run_paths["config"])
    rule_engine = RuleEngine(
        severity_high_area_fraction=config.rules.severity_high_area_fraction,
        severity_medium_area_fraction=config.rules.severity_medium_area_fraction,
        elongated_aspect_ratio=config.rules.elongated_aspect_ratio,
        thin_fill_ratio=config.rules.thin_fill_ratio,
        distributed_region_count=config.rules.distributed_region_count,
    )

    records: List[Dict] = []
    category_metrics: Dict[str, Dict] = {}
    category_scores: Dict[str, List[float]] = {}
    category_labels: Dict[str, List[int]] = {}
    category_thresholds: Dict[str, float] = {}

    for category in config.dataset.categories:
        model = load_model(category, config.model, config.outputs.models_dir)
        test_dataset = MVTecSubset(indexed[category]["test"], config.preprocessing.image_size, config.preprocessing.grayscale)

        image_labels = []
        predicted_labels = []
        image_scores = []
        pixel_labels = []
        pixel_scores = []
        per_image_masks = []
        per_image_score_maps = []
        dice_values = []
        iou_values = []
        coverage_values = []

        for loaded in test_dataset:
            prediction = model.predict(loaded.image)
            binary_mask = prediction.binary_mask
            if config.postprocessing.enabled:
                binary_mask = cleanup_binary_mask(
                    binary_mask,
                    opening_iterations=config.postprocessing.opening_iterations,
                    closing_iterations=config.postprocessing.closing_iterations,
                    min_component_area_px=config.postprocessing.min_component_area_px,
                )
            regions = summarize_regions(
                binary_mask,
                min_region_area_px=config.rules.min_region_area_px,
                border_margin_px=config.rules.border_margin_px,
            )
            predicted_label = int(prediction.score >= model.score_threshold)
            normalized_score = float(prediction.score / max(model.score_threshold, 1e-12))
            rule_result = rule_engine.analyze(category, regions, normalized_score)
            explanation = compose_explanation(category, rule_result, len(regions))
            largest_area_fraction = float(regions[0].area_fraction) if regions else 0.0
            total_area_fraction = float(sum(region.area_fraction for region in regions)) if regions else 0.0

            image_labels.append(loaded.sample.label)
            predicted_labels.append(predicted_label)
            image_scores.append(prediction.score)

            coverage = float("nan")
            if loaded.mask is not None:
                pixel_labels.append(loaded.mask.reshape(-1))
                pixel_scores.append(prediction.anomaly_map.reshape(-1))
                per_image_masks.append(loaded.mask.astype(bool))
                per_image_score_maps.append(prediction.anomaly_map)
                dice_values.append(dice_score(loaded.mask, binary_mask))
                iou_values.append(iou_score(loaded.mask, binary_mask))
                if loaded.mask.any():
                    coverage = coverage_score(loaded.mask, binary_mask)
                    coverage_values.append(coverage)

            records.append(
                {
                    "category": category,
                    "image_path": str(loaded.sample.image_path),
                    "label": int(loaded.sample.label),
                    "predicted_label": predicted_label,
                    "defect_type": loaded.sample.defect_type,
                    "image_score": float(prediction.score),
                    "normalized_score": normalized_score,
                    "score_threshold": float(model.score_threshold),
                    "region_count": len(regions),
                    "largest_region_area_fraction": largest_area_fraction,
                    "total_region_area_fraction": total_area_fraction,
                    "severity": rule_result.severity,
                    "archetype": rule_result.archetype,
                    "confidence": rule_result.confidence,
                    "quality_flag": rule_result.quality_flag,
                    "spatial_relations": ",".join(rule_result.spatial_relations) if rule_result.spatial_relations else "",
                    "explanation": explanation,
                    "coverage": coverage,
                }
            )

        pixel_auc = float("nan")
        if pixel_labels and pixel_scores:
            pixel_auc = auroc(np.concatenate(pixel_labels), np.concatenate(pixel_scores))

        pro = float("nan")
        if per_image_masks and any(mask.any() for mask in per_image_masks):
            pro = pro_score(per_image_masks, per_image_score_maps, max_fpr=0.3)

        category_metrics[category] = {
            "image_auroc": auroc(np.asarray(image_labels), np.asarray(image_scores)),
            "pixel_auroc": pixel_auc,
            "pro_score": pro,
            "mean_dice": float(np.mean(dice_values)) if dice_values else float("nan"),
            "mean_iou": float(np.mean(iou_values)) if iou_values else float("nan"),
            "mean_coverage": float(np.mean(coverage_values)) if coverage_values else float("nan"),
            "num_test_samples": len(image_labels),
        }
        category_metrics[category].update(
            classification_metrics(np.asarray(image_labels), np.asarray(predicted_labels))
        )
        category_scores[category] = image_scores
        category_labels[category] = image_labels
        category_thresholds[category] = float(model.score_threshold)

    summary = aggregate_records(records)
    summary["metrics"] = category_metrics
    save_report(config.outputs.reports_dir / "evaluation_summary.json", summary)
    save_report(run_paths["reports"] / "evaluation_summary.json", summary)

    if config.runtime.save_per_sample_records:
        save_records_csv(config.outputs.records_dir / "per_sample_records.csv", records)
        save_records_jsonl(config.outputs.records_dir / "per_sample_records.jsonl", records)
        save_records_csv(run_paths["records"] / "per_sample_records.csv", records)
        save_records_jsonl(run_paths["records"] / "per_sample_records.jsonl", records)

    save_run_metadata(
        run_paths["metadata"],
        {
            "categories": list(config.dataset.categories),
            "num_records": len(records),
            "postprocessing_enabled": config.postprocessing.enabled,
            "records_path": str(run_paths["records"] / "per_sample_records.csv"),
            "summary_path": str(run_paths["reports"] / "evaluation_summary.json"),
        },
    )
    return summary
