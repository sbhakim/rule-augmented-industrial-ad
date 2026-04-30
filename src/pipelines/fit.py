"""Per-category fitting loop.

Walks the categories named in the config, instantiates a fresh
backend for each, fits it on that category's anomaly-free training
images, and persists the result. The fitted artifacts are written
to ``models_dir`` so that ``evaluate.py`` can later run on its own
without re-doing any training work.
"""

from typing import Dict

from ..config.settings import ExperimentConfig
from ..data.datasets import MVTecSubset
from ..data.indexer import index_dataset
from ..models.base import BaseAnomalyModel
from ..models.factory import create_model
from ..utils.io import ensure_dir, set_global_seed


def fit_categories(config: ExperimentConfig) -> Dict[str, BaseAnomalyModel]:
    """Fit and persist one model per category named in the config.

    Only label-0 (anomaly-free) training samples are kept; this
    enforces the one-class assumption at the data layer rather
    than relying on the backend to filter internally.
    """
    set_global_seed(config.runtime.seed)
    ensure_dir(config.outputs.models_dir)
    indexed = index_dataset(config.dataset.root, config.dataset.categories)
    models: Dict[str, BaseAnomalyModel] = {}

    for category in config.dataset.categories:
        train_samples = [sample for sample in indexed[category]["train"] if sample.label == 0]
        dataset = MVTecSubset(train_samples, config.preprocessing.image_size, config.preprocessing.grayscale)
        images = list(dataset.images())
        model = create_model(category, config.model)
        model.fit(images)
        model.save(config.outputs.models_dir)
        models[category] = model

    return models
