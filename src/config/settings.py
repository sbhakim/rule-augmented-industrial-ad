"""Schema for the JSON configuration consumed by the pipelines.

Each subsection of the config maps to a dataclass below. The whole
schema is mirrored both ways: ``ExperimentConfig.from_dict`` parses
a nested dict produced by the JSON loader, and ``to_dict`` writes
the same shape back so a fitted run can be re-described in full.
Keeping the schema explicit (rather than passing raw dicts) gives
type checks at the call site and a single place to add a new
parameter when extending the framework.
"""

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

from ..data.paths import data_root, outputs_root, project_root
from .constants import DEFAULT_CATEGORIES


@dataclass
class DatasetConfig:
    """Dataset and category selection settings."""

    root: Path = field(default_factory=data_root)
    categories: List[str] = field(default_factory=lambda: list(DEFAULT_CATEGORIES))


@dataclass
class PreprocessingConfig:
    """Image preprocessing settings."""

    image_size: Tuple[int, int] = (256, 256)
    grayscale: bool = True


@dataclass
class ModelConfig:
    """Anomaly model settings."""

    model_type: str = "normal_stats"
    eps: float = 1e-6
    pixel_quantile: float = 0.995
    score_quantile: float = 0.995
    topk_ratio: float = 0.02
    smoothing_sigma: float = 0.0
    # Feature-model-specific settings (used when model_type == "feature")
    backbone: str = "wide_resnet50_2"
    feature_layers: Tuple[int, ...] = (1, 2, 3)


@dataclass
class PostprocessingConfig:
    """Postprocessing controls for binary anomaly masks."""

    enabled: bool = True
    opening_iterations: int = 1
    closing_iterations: int = 1
    min_component_area_px: int = 32


@dataclass
class RuleConfig:
    """Threshold set consumed by the symbolic rule engine.

    These constants are deliberately surfaced at the config level
    rather than hard-coded inside the engine so a user studying the
    sensitivity of the symbolic layer can vary them from a single
    JSON file. Defaults are conservative; aggressive cleanup or
    aggressive severity bucketing should be done by overriding
    them rather than by patching the engine.
    """

    min_region_area_px: int = 32
    severity_high_area_fraction: float = 0.05
    severity_medium_area_fraction: float = 0.015
    elongated_aspect_ratio: float = 3.0
    thin_fill_ratio: float = 0.15
    distributed_region_count: int = 3
    border_margin_px: int = 4


@dataclass
class OutputConfig:
    """Output locations for models, reports, and plots."""

    root: Path = field(default_factory=outputs_root)
    models_dir: Path = field(default_factory=lambda: outputs_root() / "models")
    reports_dir: Path = field(default_factory=lambda: outputs_root() / "reports")
    plots_dir: Path = field(default_factory=lambda: outputs_root() / "plots")
    records_dir: Path = field(default_factory=lambda: outputs_root() / "records")
    run_metadata_dir: Path = field(default_factory=lambda: outputs_root() / "runs")


@dataclass
class RuntimeConfig:
    """General runtime and reproducibility settings."""

    experiment_name: str = "mvtec_rule_augmented_baseline"
    seed: int = 7
    save_per_sample_records: bool = True
    save_qualitative_examples: bool = True
    max_qualitative_examples_per_category: int = 12


@dataclass
class ExperimentConfig:
    """Full experiment configuration."""

    project_root: Path = field(default_factory=project_root)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    postprocessing: PostprocessingConfig = field(default_factory=PostprocessingConfig)
    rules: RuleConfig = field(default_factory=RuleConfig)
    outputs: OutputConfig = field(default_factory=OutputConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the full configuration into a JSON-serializable dictionary."""
        payload = asdict(self)
        payload["project_root"] = str(self.project_root)
        payload["dataset"]["root"] = str(self.dataset.root)
        payload["outputs"]["root"] = str(self.outputs.root)
        payload["outputs"]["models_dir"] = str(self.outputs.models_dir)
        payload["outputs"]["reports_dir"] = str(self.outputs.reports_dir)
        payload["outputs"]["plots_dir"] = str(self.outputs.plots_dir)
        payload["outputs"]["records_dir"] = str(self.outputs.records_dir)
        payload["outputs"]["run_metadata_dir"] = str(self.outputs.run_metadata_dir)
        return payload

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ExperimentConfig":
        """Construct an ExperimentConfig from a nested dictionary."""
        dataset_payload = payload.get("dataset", {})
        preprocessing_payload = payload.get("preprocessing", {})
        model_payload = payload.get("model", {})
        postprocessing_payload = payload.get("postprocessing", {})
        rules_payload = payload.get("rules", {})
        outputs_payload = payload.get("outputs", {})
        runtime_payload = payload.get("runtime", {})

        return cls(
            project_root=Path(payload.get("project_root", project_root())),
            dataset=DatasetConfig(
                root=Path(dataset_payload.get("root", data_root())),
                categories=list(dataset_payload.get("categories", DEFAULT_CATEGORIES)),
            ),
            preprocessing=PreprocessingConfig(
                image_size=tuple(preprocessing_payload.get("image_size", (256, 256))),
                grayscale=bool(preprocessing_payload.get("grayscale", True)),
            ),
            model=ModelConfig(
                model_type=str(model_payload.get("model_type", "normal_stats")),
                eps=float(model_payload.get("eps", 1e-6)),
                pixel_quantile=float(model_payload.get("pixel_quantile", 0.995)),
                score_quantile=float(model_payload.get("score_quantile", 0.995)),
                topk_ratio=float(model_payload.get("topk_ratio", 0.02)),
                smoothing_sigma=float(model_payload.get("smoothing_sigma", 0.0)),
                backbone=str(model_payload.get("backbone", "wide_resnet50_2")),
                feature_layers=tuple(model_payload.get("feature_layers", (1, 2, 3))),
            ),
            postprocessing=PostprocessingConfig(
                enabled=bool(postprocessing_payload.get("enabled", True)),
                opening_iterations=int(postprocessing_payload.get("opening_iterations", 1)),
                closing_iterations=int(postprocessing_payload.get("closing_iterations", 1)),
                min_component_area_px=int(postprocessing_payload.get("min_component_area_px", 32)),
            ),
            rules=RuleConfig(
                min_region_area_px=int(rules_payload.get("min_region_area_px", 32)),
                severity_high_area_fraction=float(rules_payload.get("severity_high_area_fraction", 0.05)),
                severity_medium_area_fraction=float(rules_payload.get("severity_medium_area_fraction", 0.015)),
                elongated_aspect_ratio=float(rules_payload.get("elongated_aspect_ratio", 3.0)),
                thin_fill_ratio=float(rules_payload.get("thin_fill_ratio", 0.15)),
                distributed_region_count=int(rules_payload.get("distributed_region_count", 3)),
                border_margin_px=int(rules_payload.get("border_margin_px", 4)),
            ),
            outputs=OutputConfig(
                root=Path(outputs_payload.get("root", outputs_root())),
                models_dir=Path(outputs_payload.get("models_dir", outputs_root() / "models")),
                reports_dir=Path(outputs_payload.get("reports_dir", outputs_root() / "reports")),
                plots_dir=Path(outputs_payload.get("plots_dir", outputs_root() / "plots")),
                records_dir=Path(outputs_payload.get("records_dir", outputs_root() / "records")),
                run_metadata_dir=Path(outputs_payload.get("run_metadata_dir", outputs_root() / "runs")),
            ),
            runtime=RuntimeConfig(
                experiment_name=str(runtime_payload.get("experiment_name", "mvtec_rule_augmented_baseline")),
                seed=int(runtime_payload.get("seed", 7)),
                save_per_sample_records=bool(runtime_payload.get("save_per_sample_records", True)),
                save_qualitative_examples=bool(runtime_payload.get("save_qualitative_examples", True)),
                max_qualitative_examples_per_category=int(runtime_payload.get("max_qualitative_examples_per_category", 12)),
            ),
        )


def default_experiment_config() -> ExperimentConfig:
    """Create the default experiment configuration."""
    return ExperimentConfig()
