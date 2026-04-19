"""Dataset loading helpers built on top of the indexed records."""

from dataclasses import dataclass
from typing import Iterable, Iterator, List, Optional, Tuple

import numpy as np

from .indexer import MVTecSample
from .transforms import load_image_array, load_mask_array


@dataclass
class LoadedSample:
    """Materialized sample arrays ready for the pipeline."""

    sample: MVTecSample
    image: np.ndarray
    mask: Optional[np.ndarray]


class MVTecSubset:
    """Simple iterable wrapper around indexed samples."""

    def __init__(self, samples: List[MVTecSample], image_size: Tuple[int, int], grayscale: bool):
        self.samples = samples
        self.image_size = image_size
        self.grayscale = grayscale

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self) -> Iterator[LoadedSample]:
        for sample in self.samples:
            yield self.load(sample)

    def load(self, sample: MVTecSample) -> LoadedSample:
        image = load_image_array(sample.image_path, self.image_size, self.grayscale)
        mask = None
        if sample.mask_path is not None and sample.mask_path.exists():
            mask = load_mask_array(sample.mask_path, self.image_size)
        return LoadedSample(sample=sample, image=image, mask=mask)

    def images(self) -> Iterable[np.ndarray]:
        for loaded in self:
            yield loaded.image
