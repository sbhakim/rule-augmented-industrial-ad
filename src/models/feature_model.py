"""Per-category anomaly model based on pretrained CNN features.

This is one of the two interchangeable visual backends. A frozen
ImageNet backbone produces multi-scale feature maps; per-patch
Gaussian parameters are estimated from anomaly-free training images
after a random channel reduction; at test time the anomaly map is
the per-patch Mahalanobis distance to that Gaussian.

The model is fully self-contained: no other module needs to know
about the backbone, the channel reduction, or the covariance
inversion. The only outward-facing surface is the
``BaseAnomalyModel`` contract --- ``fit``, ``predict``, ``save``,
``load`` --- so a future user can drop in a different feature
extractor or scoring rule by subclassing the same interface.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import zoom
from torchvision import models, transforms

from ..features.anomaly_maps import smooth_anomaly_map, topk_score
from ..utils.io import ensure_dir, save_json
from .base import AnomalyPrediction, BaseAnomalyModel

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]

# Random dimension reduction target (following PaDiM paper)
_REDUCED_DIM = 100


@dataclass
class FeatureModelState:
    """JSON-serializable metadata that accompanies the saved weights.

    The numpy arrays themselves are written to a sibling ``.npz``;
    everything that is not a tensor (backbone choice, layer indices,
    thresholds learned during ``fit``, the patch grid shape) lives in
    this struct so a fitted model can be reloaded without re-running
    the feature extractor on training data.
    """

    category: str
    backbone: str
    feature_layers: Tuple[int, ...]
    pixel_quantile: float
    score_quantile: float
    smoothing_sigma: float
    image_size: Tuple[int, int]
    full_embedding_dim: int
    reduced_dim: int
    pixel_threshold: float
    score_threshold: float
    topk_ratio: float
    patch_h: int
    patch_w: int


class _FeatureExtractor(torch.nn.Module):
    """Hook-free wrapper that returns intermediate activations.

    The standard torchvision backbones expose only the final logits.
    This wrapper rebuilds the network as a stem plus a list of
    residual blocks so an arbitrary subset of intermediate stages
    can be returned without forward hooks. Adding a backbone
    requires only a new branch in the constructor.
    """

    def __init__(self, backbone_name: str, layers: Tuple[int, ...]):
        super().__init__()
        if backbone_name == "wide_resnet50_2":
            backbone = models.wide_resnet50_2(weights=models.Wide_ResNet50_2_Weights.IMAGENET1K_V1)
        elif backbone_name == "resnet18":
            backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        # Split the backbone: the stem (conv + bn + relu + maxpool)
        # is fused into a single ``prefix``, and each subsequent
        # residual stage is kept as its own block so we can stop at
        # the deepest requested layer and ignore the rest.
        self.layers = sorted(layers)
        children = list(backbone.children())
        self.prefix = torch.nn.Sequential(*children[:4])
        self.blocks = torch.nn.ModuleList()
        for i in range(1, max(self.layers) + 1):
            self.blocks.append(children[3 + i])

        # The backbone is used only as a fixed feature extractor;
        # gradients are never needed.
        for param in self.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        features = []
        out = self.prefix(x)
        for i, block in enumerate(self.blocks, start=1):
            out = block(out)
            if i in self.layers:
                features.append(out)
        return features


class CategoryFeatureModel(BaseAnomalyModel):
    """One Gaussian-per-patch model fitted from a single category.

    Each spatial location in the feature grid gets its own multivariate
    Gaussian estimated from the training images of one inspection
    category. Anomaly scoring is the Mahalanobis distance from a test
    embedding to the patch-specific Gaussian.

    Two implementation choices matter for users who plan to extend
    this class. First, a random subset of feature channels is kept
    before covariance estimation; this keeps the per-patch covariance
    matrices small enough to invert reliably. Second, the covariance
    estimation, inversion, and Mahalanobis evaluation are batched
    over patches with ``np.einsum`` --- avoiding explicit Python
    loops makes per-image scoring cheap enough that the bottleneck
    is the backbone forward pass, not the linear algebra.
    """

    def __init__(
        self,
        category: str,
        backbone: str = "wide_resnet50_2",
        feature_layers: Tuple[int, ...] = (1, 2, 3),
        pixel_quantile: float = 0.995,
        score_quantile: float = 0.995,
        topk_ratio: float = 0.02,
        smoothing_sigma: float = 4.0,
        image_size: Tuple[int, int] = (256, 256),
    ):
        self.category = category
        self.backbone_name = backbone
        self.feature_layers = feature_layers
        self.pixel_quantile = pixel_quantile
        self.score_quantile = score_quantile
        self.topk_ratio = topk_ratio
        self.smoothing_sigma = smoothing_sigma
        self.image_size = image_size

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._extractor: Optional[_FeatureExtractor] = None
        self._normalize = transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD)

        self.patch_h: Optional[int] = None
        self.patch_w: Optional[int] = None
        self.full_embedding_dim: Optional[int] = None
        self.reduced_dim: int = _REDUCED_DIM

        # Random projection indices for dimension reduction
        self.dim_indices: Optional[np.ndarray] = None

        # Per-patch Gaussian parameters (after dimension reduction)
        self.mean: Optional[np.ndarray] = None      # (reduced_dim, N_patches)
        self.cov_inv: Optional[np.ndarray] = None    # (N_patches, reduced_dim, reduced_dim)
        self.pixel_threshold: Optional[float] = None
        self.score_threshold: Optional[float] = None

    def _get_extractor(self) -> _FeatureExtractor:
        if self._extractor is None:
            self._extractor = _FeatureExtractor(self.backbone_name, self.feature_layers)
            self._extractor.eval()
            self._extractor.to(self._device)
        return self._extractor

    def _image_to_tensor(self, image: np.ndarray) -> torch.Tensor:
        """Convert a [0,1] numpy image to a normalized ImageNet tensor."""
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        tensor = torch.from_numpy(image.transpose(2, 0, 1)).float()
        tensor = self._normalize(tensor)
        return tensor.unsqueeze(0).to(self._device)

    def _extract_embedding(self, image: np.ndarray) -> np.ndarray:
        """Extract multi-scale patch embeddings -> (D_full, N_patches)."""
        extractor = self._get_extractor()
        tensor = self._image_to_tensor(image)
        features = extractor(tensor)

        target_h, target_w = features[0].shape[2], features[0].shape[3]
        aligned = []
        for feat in features:
            if feat.shape[2] != target_h or feat.shape[3] != target_w:
                feat = F.interpolate(feat, size=(target_h, target_w), mode="bilinear", align_corners=False)
            aligned.append(feat)

        combined = torch.cat(aligned, dim=1).squeeze(0)  # (D, H, W)
        D, H, W = combined.shape
        embedding = combined.reshape(D, H * W).cpu().numpy()

        if self.patch_h is None:
            self.patch_h = H
            self.patch_w = W
            self.full_embedding_dim = D

        return embedding

    def _reduce_dims(self, embedding: np.ndarray) -> np.ndarray:
        """Apply random dimension selection: (D_full, N) -> (reduced_dim, N)."""
        return embedding[self.dim_indices]

    def fit(self, images: List[np.ndarray]) -> "CategoryFeatureModel":
        """Fit per-patch Gaussians on a category's anomaly-free images.

        Five stages: collect embeddings, draw the random channel
        subset, compute the per-patch mean and inverse covariance,
        and finally derive image- and pixel-level thresholds from
        the same training images so that detection at deployment
        time uses category-specific operating points without
        peeking at any anomalous data.
        """
        # Stage 1: gather backbone embeddings for the full training
        # set. Stacking up front lets the rest of the routine be
        # written as pure NumPy without any per-image Python loops.
        embeddings = []
        for img in images:
            embeddings.append(self._extract_embedding(img))

        all_emb = np.stack(embeddings, axis=0)  # (num_images, D_full, N_patches)
        n_images, D_full, N = all_emb.shape

        # Stage 2: pick the channel subset. The seed is fixed so the
        # subset is reproducible across runs and saved models stay
        # comparable; the indices are sorted so downstream slicing
        # is contiguous in memory.
        rng = np.random.RandomState(42)
        self.reduced_dim = min(_REDUCED_DIM, D_full)
        self.dim_indices = rng.choice(D_full, self.reduced_dim, replace=False)
        self.dim_indices.sort()

        all_emb_reduced = all_emb[:, self.dim_indices, :]  # (num_images, reduced_dim, N)
        D = self.reduced_dim

        # Stage 3: per-patch mean across the training set.
        self.mean = all_emb_reduced.mean(axis=0)  # (D, N)

        # Stage 4: per-patch covariance, batched across all patches
        # at once. ``einsum`` does the (D x D) outer-product sum per
        # patch in one call, after which a small ridge regularizer
        # is added before inversion to keep numerically sparse
        # categories well-conditioned.
        centered = all_emb_reduced - self.mean[np.newaxis, :, :]
        centered_t = centered.transpose(2, 0, 1)  # (N, num_images, D)
        cov_batch = np.einsum("nji,njk->nik", centered_t, centered_t) / max(n_images - 1, 1)

        reg = 0.01 * np.eye(D, dtype=np.float32)
        cov_batch = cov_batch + reg[np.newaxis, :, :]
        # Inversion is done in float64 for numerical stability and
        # then downcast for storage and runtime evaluation.
        self.cov_inv = np.linalg.inv(cov_batch.astype(np.float64)).astype(np.float32)

        # Stage 5: derive thresholds from the just-fitted model.
        # Quantiles are taken over the same training images: pixel-
        # level over every distance value, image-level over the
        # top-k aggregated score per image. Both are stored on the
        # model so prediction is threshold-aware without re-reading
        # any training data.
        train_scores = []
        all_pixels = []
        for img in images:
            amap = self._compute_anomaly_map(img)
            all_pixels.append(amap.reshape(-1))
            train_scores.append(topk_score(amap, self.topk_ratio))

        self.pixel_threshold = float(np.quantile(np.concatenate(all_pixels), self.pixel_quantile))
        self.score_threshold = float(np.quantile(train_scores, self.score_quantile))

        return self

    def _compute_anomaly_map(self, image: np.ndarray) -> np.ndarray:
        """Pixel-resolution anomaly map for a single image.

        The Mahalanobis distance is computed at the patch grid
        resolution and then bilinearly upsampled to the input
        resolution. A Gaussian smoothing pass after upsampling
        suppresses block artifacts at patch boundaries; the
        smoothing kernel size is left as a constructor argument so
        users can disable it for ablations.
        """
        embedding = self._extract_embedding(image)
        embedding = self._reduce_dims(embedding)  # (D, N)
        diff = embedding - self.mean  # (D, N)

        # Two ``einsum`` passes evaluate the quadratic form
        # diff^T @ cov_inv @ diff for every patch in one shot.
        # The intermediate ``temp`` is (N, D); the final inner
        # product collapses it to one scalar per patch. The clip
        # to a non-negative domain protects ``sqrt`` from tiny
        # negative values that can appear from floating-point
        # rounding when a patch is almost on the training mean.
        diff_t = diff.T  # (N, D)
        temp = np.einsum("ni,nij->nj", diff_t, self.cov_inv)  # (N, D)
        distances = np.sqrt(np.clip(np.einsum("ni,ni->n", temp, diff_t), 0, None))  # (N,)

        distance_map = distances.reshape(self.patch_h, self.patch_w)
        scale_h = self.image_size[1] / self.patch_h
        scale_w = self.image_size[0] / self.patch_w
        anomaly_map = zoom(distance_map, (scale_h, scale_w), order=1).astype(np.float32)

        return smooth_anomaly_map(anomaly_map, self.smoothing_sigma)

    def predict(self, image: np.ndarray) -> AnomalyPrediction:
        if self.mean is None or self.cov_inv is None or self.pixel_threshold is None:
            raise RuntimeError("Model must be fitted before prediction.")
        anomaly_map = self._compute_anomaly_map(image)
        score = topk_score(anomaly_map, self.topk_ratio)
        binary_mask = (anomaly_map >= self.pixel_threshold).astype(np.uint8)
        return AnomalyPrediction(score=score, anomaly_map=anomaly_map, binary_mask=binary_mask)

    def save(self, output_dir: Path) -> None:
        if self.mean is None or self.cov_inv is None or self.pixel_threshold is None or self.score_threshold is None:
            raise RuntimeError("Cannot save an unfitted model.")

        ensure_dir(output_dir)
        np.savez_compressed(
            output_dir / f"{self.category}_feature_model.npz",
            mean=self.mean,
            cov_inv=self.cov_inv,
            dim_indices=self.dim_indices,
        )
        state = FeatureModelState(
            category=self.category,
            backbone=self.backbone_name,
            feature_layers=self.feature_layers,
            pixel_quantile=self.pixel_quantile,
            score_quantile=self.score_quantile,
            smoothing_sigma=self.smoothing_sigma,
            image_size=self.image_size,
            full_embedding_dim=self.full_embedding_dim,
            reduced_dim=self.reduced_dim,
            pixel_threshold=self.pixel_threshold,
            score_threshold=self.score_threshold,
            topk_ratio=self.topk_ratio,
            patch_h=self.patch_h,
            patch_w=self.patch_w,
        )
        save_json(output_dir / f"{self.category}_feature_model.json", asdict(state))

    @classmethod
    def load(cls, category: str, output_dir: Path) -> "CategoryFeatureModel":
        weights = np.load(output_dir / f"{category}_feature_model.npz")
        with open(output_dir / f"{category}_feature_model.json", "r", encoding="utf-8") as f:
            state = json.load(f)

        model = cls(
            category=category,
            backbone=state["backbone"],
            feature_layers=tuple(state["feature_layers"]),
            pixel_quantile=float(state.get("pixel_quantile", 0.995)),
            score_quantile=float(state.get("score_quantile", 0.995)),
            topk_ratio=float(state["topk_ratio"]),
            smoothing_sigma=float(state.get("smoothing_sigma", 4.0)),
            image_size=tuple(state.get("image_size", (256, 256))),
        )
        model.mean = weights["mean"]
        model.cov_inv = weights["cov_inv"]
        model.dim_indices = weights["dim_indices"]
        model.pixel_threshold = float(state["pixel_threshold"])
        model.score_threshold = float(state["score_threshold"])
        model.full_embedding_dim = int(state["full_embedding_dim"])
        model.reduced_dim = int(state["reduced_dim"])
        model.patch_h = int(state["patch_h"])
        model.patch_w = int(state["patch_w"])
        return model
