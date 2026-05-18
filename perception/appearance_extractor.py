"""
Appearance feature extractor for object re-identification.

Architecture
------------
AppearanceExtractor (ABC)
    └── NullAppearanceExtractor    — returns None (re-association disabled)
    └── DINOv2AppearanceExtractor  — Meta DINOv2 foundation features

Why DINOv2 over classical ReID
------------------------------
Classical person-ReID models (OSNet, FastReID) are trained on
human datasets — Market-1501, MSMT17 — and only work on people.
This pipeline tracks the full 80-class COCO set: people, vehicles,
furniture, animals. We need a backbone whose embeddings discriminate
*any* object instance, not just humans.

DINOv2 (Meta, 2023) is a self-supervised vision foundation model
that produces general-purpose image features. Cosine similarity in
its embedding space gives strong instance-discriminative signal
across arbitrary categories — exactly what we need for cross-class
ReID. CLIP would also work but is multi-modal overhead; DINOv2 is
vision-only and the smaller variants fit comfortably on a 12 GB GPU.

Variants:
    facebook/dinov2-small  — ViT-S/14, 22M params, 384-dim, ~90 MB
                             on disk, ~600 MB GPU with activations.
                             Default for this project.
    facebook/dinov2-base   — ViT-B/14, 86M params, 768-dim, ~340 MB
                             on disk, ~1.5 GB GPU. Sharper features,
                             slower.

Output convention
-----------------
All extractors return L2-normalised embeddings so cosine similarity
collapses to a dot product downstream:
    similarity(a, b) = a @ b   ∈ [-1, 1]

A returned None means "no embedding available" — caller should
fall back to spatial gating alone.

Reference
---------
Oquab et al. (2023) — DINOv2: Learning Robust Visual Features
                      without Supervision. arXiv:2304.07193
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import List, Optional, Sequence

import numpy as np

log = logging.getLogger(__name__)

# (x1, y1, x2, y2) integer pixel bbox.
Bbox = Sequence[float]


class AppearanceExtractor(ABC):
    """
    Abstract base for appearance embedding extractors.

    Implementations consume a list of (image, bbox) pairs and return
    one L2-normalised embedding per pair. Batching is left to the
    implementation — callers should provide all crops for a frame in
    one call to amortise model overhead.
    """

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Dimensionality of the output embedding. 0 for Null."""
        ...

    @abstractmethod
    def extract(
        self,
        image: np.ndarray,
        bboxes: List[Bbox],
    ) -> List[Optional[np.ndarray]]:
        """
        Compute embeddings for a list of crops from one image.

        Parameters
        ----------
        image  : (H, W, 3) uint8 BGR (OpenCV convention)
        bboxes : list of (x1, y1, x2, y2) in pixel coordinates;
                 fractional and out-of-bounds values are clamped.

        Returns
        -------
        list of length len(bboxes), each entry either:
          - (embedding_dim,) float32 L2-normalised embedding, or
          - None when the crop was empty / invalid.
        """
        ...

    def warmup(self) -> None:
        """Optional eager initialisation. Default is a no-op."""
        return


class NullAppearanceExtractor(AppearanceExtractor):
    """
    No-op extractor — always returns Nones.

    Used when re-association is disabled or when the heavy DINOv2
    weights aren't worth loading (CI, headless smoke tests). The
    WorldMap then falls back to spatial-gating-only re-association,
    which is still useful when the robot revisits a sparse scene.
    """

    @property
    def embedding_dim(self) -> int:
        return 0

    def extract(
        self,
        image: np.ndarray,
        bboxes: List[Bbox],
    ) -> List[Optional[np.ndarray]]:
        return [None] * len(bboxes)

    def __repr__(self) -> str:
        return "NullAppearanceExtractor()"


class DINOv2AppearanceExtractor(AppearanceExtractor):
    """
    DINOv2-backed appearance extractor.

    Loads the model lazily on first use so import is cheap. Crops
    are resized to the model's expected input (224×224 for ViT-S/14
    via HuggingFace's AutoImageProcessor), batched, and embedded.

    Performance on RTX 4070Ti (dinov2-small, batch of 10 crops):
        ~6 ms median per frame including H2D transfer.
        VRAM peak ~600 MB.
    """

    def __init__(
        self,
        model_name: str = "facebook/dinov2-small",
        device:     str = "cuda",
    ) -> None:
        self._model_name = model_name
        self._device = device
        # Lazy: model loaded on first extract() so import is free.
        self._model = None
        self._processor = None
        self._embedding_dim: Optional[int] = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        log.info("Loading appearance model %s on %s ...",
                 self._model_name, self._device)
        import torch
        from transformers import AutoImageProcessor, AutoModel
        self._processor = AutoImageProcessor.from_pretrained(self._model_name)
        self._model = AutoModel.from_pretrained(self._model_name).to(self._device)
        self._model.eval()
        self._embedding_dim = int(self._model.config.hidden_size)
        log.info(
            "Appearance model ready (dim=%d, device=%s)",
            self._embedding_dim, self._device,
        )

    def warmup(self) -> None:
        self._ensure_loaded()

    @property
    def embedding_dim(self) -> int:
        self._ensure_loaded()
        return self._embedding_dim or 0

    def extract(
        self,
        image: np.ndarray,
        bboxes: List[Bbox],
    ) -> List[Optional[np.ndarray]]:
        if not bboxes:
            return []
        self._ensure_loaded()

        import torch

        h, w = image.shape[:2]
        crops: List[np.ndarray] = []
        out_idx: List[int] = []
        results: List[Optional[np.ndarray]] = [None] * len(bboxes)

        for i, (x1, y1, x2, y2) in enumerate(bboxes):
            x1i = max(0, int(x1))
            y1i = max(0, int(y1))
            x2i = min(w, int(x2))
            y2i = min(h, int(y2))
            if x2i <= x1i or y2i <= y1i:
                continue
            crop = image[y1i:y2i, x1i:x2i]
            if crop.size == 0:
                continue
            # DINOv2's processor expects RGB; image is BGR (OpenCV).
            # `[..., ::-1]` returns a *view* with negative strides on the
            # last axis. transformers' image processor passes that through
            # to torch.from_numpy, which then raises
            # "tensors with negative strides are not currently supported"
            # — caught by the MOT17 benchmark run, not by unit tests
            # because the unit tests used contiguous arrays. ascontiguousarray
            # is the canonical fix and stays cheap (~50 µs per crop).
            crops.append(np.ascontiguousarray(crop[..., ::-1]))
            out_idx.append(i)

        if not crops:
            return results

        with torch.no_grad():
            inputs = self._processor(images=crops, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(self._device)
            outputs = self._model(pixel_values=pixel_values)
            # CLS token from last hidden state — DINOv2 convention.
            feats = outputs.last_hidden_state[:, 0, :]   # (N, D)
            feats = torch.nn.functional.normalize(feats, p=2, dim=1)
            feats_np = feats.cpu().numpy().astype(np.float32)

        for k, idx in enumerate(out_idx):
            results[idx] = feats_np[k]
        return results

    def __repr__(self) -> str:
        return (
            f"DINOv2AppearanceExtractor(model={self._model_name!r}, "
            f"device={self._device!r}, "
            f"loaded={self._model is not None})"
        )
