"""
Semantic segmentation for the robotics perception pipeline.

Provides a per-pixel class label map. Two downstream consumers:
  1. Drivable-surface masking — useful for planners (road, sidewalk,
     terrain in Cityscapes; floor / road / earth in ADE20K).
  2. Class priors that refine Phase 4 spatial-memory stability
     classification — a person *on* a road is more likely transient
     than a person *on* furniture, etc.

Architecture
------------
SemanticSegmenter (ABC)
    ├── NullSemanticSegmenter        — no-op, always returns None
    └── Mask2FormerSemanticSegmenter — HuggingFace Mask2Former
                                       (default: Swin-T on Cityscapes)

Backbone choice
---------------
Mask2Former (Cheng et al., CVPR 2022) — unified mask-prediction
architecture that beats per-task SOTA on Cityscapes (Swin-L: 84.5 mIoU)
and ADE20K. Swin-T variant is ~50M params / ~15 GFLOPs and fits
comfortably alongside YOLO + Depth Anything + DINOv2 on a 12 GB GPU.

Why not SegFormer or SAM2:
  - SegFormer is lighter but caps at ~83 mIoU on Cityscapes vs Mask2Former's
    ~84.5; the few-FPS difference isn't decisive on 4070Ti-class hardware.
  - SAM2 is *promptable* and class-agnostic — fantastic for interactive
    segmentation, wrong tool for a per-frame semantic prior.

Coordinate conventions
----------------------
The returned mask has the SAME shape as the source frame (H, W). Class
IDs are dataset-specific (Cityscapes 19-class, ADE20K 150-class); the
dataclass carries the id→name map so downstream code can resolve them
without hardcoding the dataset.

Upgrade path
------------
Replace Mask2Former with OneFormer (Jain et al., CVPR 2023) or an
open-vocabulary model (OpenScene, ConceptGraphs 2024) when use cases
demand classes outside the closed Cityscapes/ADE20K vocabularies.
Slots into the same SemanticSegmenter ABC.

References
----------
Cheng et al. — Mask2Former: Masked-attention Mask Transformer for
               Universal Image Segmentation. CVPR 2022.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Optional

import numpy as np

from perception.camera_interface import CameraFrame
from world_model.stability import StabilityClass

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class SemanticMask:
    """
    Per-pixel semantic-class label map for one frame.

    Attributes
    ----------
    mask          : (H, W) int32 — class ID per pixel.
    class_names   : {class_id: human-readable name}. Dataset-dependent.
    dataset       : 'cityscapes' | 'ade20k' | ... — used by helper
                    functions that pick class subsets (e.g. drivable).
    timestamp     : monotonic timestamp of the source frame.
    frame_idx     : frame index of the source frame.
    inference_ms  : inference latency for this frame, milliseconds.
    """
    mask:         np.ndarray
    class_names:  Dict[int, str]
    dataset:      str
    timestamp:    float
    frame_idx:    int
    inference_ms: float = 0.0

    def __post_init__(self) -> None:
        self.mask = np.asarray(self.mask, dtype=np.int32)
        assert self.mask.ndim == 2, f"mask must be 2-D, got {self.mask.shape}"

    @property
    def shape(self) -> tuple[int, int]:
        return self.mask.shape  # type: ignore[return-value]

    def class_at(self, u: int, v: int) -> Optional[str]:
        """Return class name at pixel (u col, v row), or None if out-of-bounds."""
        H, W = self.mask.shape
        if not (0 <= v < H and 0 <= u < W):
            return None
        return self.class_names.get(int(self.mask[v, u]))


# ---------------------------------------------------------------------------
# ABC
# ---------------------------------------------------------------------------

class SemanticSegmenter(ABC):
    """
    Abstract base class for semantic segmentation backends.

    Implementations return one SemanticMask per call, OR None when the
    backend is disabled / unavailable. None lets downstream consumers
    fall back to bbox-only reasoning without a code path change.
    """

    @abstractmethod
    def segment(self, frame: CameraFrame) -> Optional[SemanticMask]:
        """Per-pixel class IDs over the input image."""
        ...

    @abstractmethod
    def warmup(self) -> None:
        """Warmup inference — call once after construction."""
        ...

    @property
    @abstractmethod
    def mean_inference_ms(self) -> float:
        """Mean inference latency over the last ~30 frames, milliseconds."""
        ...


# ---------------------------------------------------------------------------
# Null backend
# ---------------------------------------------------------------------------

class NullSemanticSegmenter(SemanticSegmenter):
    """
    No-op segmenter. Always returns None.

    Used when semantic segmentation is disabled in config, when the
    backend fails to load, or in tests that don't want a GPU model
    in the loop. Downstream consumers (drivable masking, stability
    refinement) must already handle None and gracefully degrade.
    """

    def segment(self, frame: CameraFrame) -> Optional[SemanticMask]:
        return None

    def warmup(self) -> None:
        pass

    @property
    def mean_inference_ms(self) -> float:
        return 0.0


# ---------------------------------------------------------------------------
# Mask2Former (HuggingFace)
# ---------------------------------------------------------------------------

class Mask2FormerSemanticSegmenter(SemanticSegmenter):
    """
    Mask2Former semantic segmentation via HuggingFace transformers.

    Defaults to the Swin-T variant fine-tuned on Cityscapes (19 classes
    including road / sidewalk / terrain — i.e., everything the
    drivable-surface helper needs out of the box). Switch to an ADE20K
    checkpoint for general indoor scenes (150 classes).

    Install
    -------
    pip install transformers accelerate timm

    Lazy loading
    ------------
    The model is loaded in __init__ inside try/except so an env that
    lacks transformers / model weights still imports the module and
    constructs a degraded instance — `segment()` then returns None
    just like NullSemanticSegmenter. Mirrors DepthAnythingEstimator.
    """

    def __init__(
        self,
        device:     str = "cuda",
        model_name: str = "facebook/mask2former-swin-tiny-cityscapes-semantic",
        dataset:    str = "cityscapes",
    ) -> None:
        self._device     = device
        self._model_name = model_name
        self._dataset    = dataset
        self._model      = None
        self._processor  = None
        self._class_names: Dict[int, str] = {}
        self._ready      = False
        self._latencies: list[float] = []
        self._load_model()

    def _load_model(self) -> None:
        try:
            import torch
            from transformers import (
                Mask2FormerForUniversalSegmentation,
                Mask2FormerImageProcessor,
            )
            log.info(
                "Loading Mask2Former: %s on %s ...",
                self._model_name, self._device,
            )
            t0 = time.monotonic()
            self._processor = Mask2FormerImageProcessor.from_pretrained(
                self._model_name,
            )
            self._model = Mask2FormerForUniversalSegmentation.from_pretrained(
                self._model_name,
            ).to(self._device).eval()
            # Resolve the id→label dict the model ships with.
            id2label = getattr(self._model.config, "id2label", None) or {}
            self._class_names = {int(k): str(v) for k, v in id2label.items()}
            log.info(
                "Mask2Former loaded in %.1fs (%d classes)",
                time.monotonic() - t0, len(self._class_names),
            )
            self._ready = True
        except Exception as exc:
            log.warning(
                "Mask2Former load failed: %s\n"
                "  Install: pip install transformers accelerate timm\n"
                "  Semantic segmentation will be disabled (returns None).",
                exc,
            )
            self._ready = False

    def warmup(self) -> None:
        if not self._ready:
            return
        try:
            dummy = np.zeros((480, 640, 3), dtype=np.uint8)
            self._run_inference(dummy)
            log.info(
                "Mask2Former warmup complete. Latency: %.1f ms",
                self.mean_inference_ms,
            )
        except Exception as exc:
            log.warning("Mask2Former warmup failed: %s", exc)

    def segment(self, frame: CameraFrame) -> Optional[SemanticMask]:
        if not self._ready:
            return None
        try:
            mask = self._run_inference(frame.image)
        except Exception as exc:
            log.debug("Mask2Former inference error (suppressed): %s", exc)
            return None
        return SemanticMask(
            mask=mask,
            class_names=self._class_names,
            dataset=self._dataset,
            timestamp=frame.timestamp,
            frame_idx=frame.frame_idx,
            inference_ms=self._latencies[-1] if self._latencies else 0.0,
        )

    def _run_inference(self, image_bgr: np.ndarray) -> np.ndarray:
        """Run the model on a BGR image, return (H, W) int32 class mask."""
        import torch
        # transformers' processors expect RGB.
        rgb = image_bgr[:, :, ::-1].copy()
        inputs = self._processor(images=rgb, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        t0 = time.monotonic()
        with torch.no_grad():
            outputs = self._model(**inputs)
        self._latencies.append((time.monotonic() - t0) * 1000.0)

        # Post-process to a single semantic map at the original resolution.
        # Mask2Former's processor handles upsampling and argmax over classes.
        H, W = image_bgr.shape[:2]
        seg = self._processor.post_process_semantic_segmentation(
            outputs, target_sizes=[(H, W)],
        )[0]
        return np.asarray(seg.cpu().numpy(), dtype=np.int32)

    @property
    def mean_inference_ms(self) -> float:
        if not self._latencies:
            return 0.0
        return float(np.mean(self._latencies[-30:]))

    @property
    def is_ready(self) -> bool:
        return self._ready


# ---------------------------------------------------------------------------
# Downstream helpers
# ---------------------------------------------------------------------------

# Drivable surfaces by dataset. Names match Mask2Former's id→label
# vocabulary on each dataset. Anything else is treated as non-drivable.
_DRIVABLE_NAMES: Dict[str, FrozenSet[str]] = {
    "cityscapes": frozenset({"road", "sidewalk", "terrain"}),
    # ADE20K's "floor" covers indoor robots; "road"/"path"/"earth" outdoor.
    "ade20k":     frozenset({"floor", "road, route", "path", "earth, ground",
                             "sidewalk, pavement"}),
}


def drivable_mask(sm: SemanticMask) -> np.ndarray:
    """
    Boolean (H, W) mask: True where the pixel's class is drivable in the
    given dataset. Unknown datasets return an all-False mask.
    """
    allowed = _DRIVABLE_NAMES.get(sm.dataset, frozenset())
    if not allowed:
        return np.zeros(sm.mask.shape, dtype=bool)
    drivable_ids = {cid for cid, name in sm.class_names.items()
                    if name in allowed}
    if not drivable_ids:
        return np.zeros(sm.mask.shape, dtype=bool)
    out = np.zeros(sm.mask.shape, dtype=bool)
    for cid in drivable_ids:
        out |= (sm.mask == cid)
    return out


# Cityscapes-name → StabilityClass for the "surface" classes (i.e., the
# stuff an object might be *on*). DYNAMIC / SEMI_STATIC values are
# rough priors — the downstream stability rule combines this with the
# COCO-class prior of the object itself.
_CITYSCAPES_SURFACE_STABILITY: Dict[str, StabilityClass] = {
    "road":          StabilityClass.STATIC,
    "sidewalk":      StabilityClass.STATIC,
    "building":      StabilityClass.STATIC,
    "wall":          StabilityClass.STATIC,
    "fence":         StabilityClass.STATIC,
    "pole":          StabilityClass.STATIC,
    "traffic light": StabilityClass.STATIC,
    "traffic sign":  StabilityClass.STATIC,
    "vegetation":    StabilityClass.STATIC,
    "terrain":       StabilityClass.STATIC,
    "sky":           StabilityClass.STATIC,
    "person":        StabilityClass.DYNAMIC,
    "rider":         StabilityClass.DYNAMIC,
    "car":           StabilityClass.SEMI_STATIC,
    "truck":         StabilityClass.SEMI_STATIC,
    "bus":           StabilityClass.SEMI_STATIC,
    "train":         StabilityClass.SEMI_STATIC,
    "motorcycle":    StabilityClass.SEMI_STATIC,
    "bicycle":       StabilityClass.SEMI_STATIC,
}


def semantic_class_stability(class_name: str, dataset: str) -> StabilityClass:
    """
    Stability prior for a SEMANTIC-segmentation class.

    Mirrors world_model.stability.stability_for_class but on the
    segmenter's vocabulary instead of COCO. Unknown classes default to
    SEMI_STATIC, same conservative middle ground.

    Used by callers that want to refine an object's stability based on
    *what surface it sits on* — e.g. a chair detected on "road" is
    more likely DYNAMIC than a chair detected on "floor".
    """
    if dataset == "cityscapes":
        return _CITYSCAPES_SURFACE_STABILITY.get(
            class_name, StabilityClass.SEMI_STATIC,
        )
    # Other datasets fall back to SEMI_STATIC pending a hand-built table.
    return StabilityClass.SEMI_STATIC
