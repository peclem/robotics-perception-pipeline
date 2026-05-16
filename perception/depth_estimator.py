"""
Monocular depth estimation for the robotics perception pipeline.

Provides metric depth at each detection centroid using Depth Anything V2.
Output feeds into ObjectState.position_3d for metric-space planning.

Architecture
------------
DepthEstimator (ABC)
    ├── DepthAnythingEstimator  — Depth Anything V2, ~10ms/frame on RTX 4070Ti
    └── NullDepthEstimator      — no-op fallback when model unavailable

Coordinate conventions
-----------------------
Pixel frame:   (u, v) — column, row in image
Camera frame:  (X, Y, Z) — right, down, forward in metres
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    Z = metric depth from Depth Anything V2

Depth quality
-------------
Depth Anything V2 Metric Indoor produces absolute metric depth.
Accuracy is approximately +-10% at typical indoor ranges (1-5m).
Sufficient for obstacle avoidance and scene graph metric coordinates.
Not sufficient for precise manipulation.

Upgrade path
------------
Phase 5: Replace DepthAnythingEstimator with StereoDepthEstimator
         using the Meta glasses stereo pair. Metric accuracy improves
         to +-1-2% from stereo triangulation. Same interface — zero
         changes to downstream modules.

Reference
---------
Yang et al. — Depth Anything V2 (2024). arXiv:2406.09414
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from perception.camera_interface import CameraFrame, CameraIntrinsics
from perception.detector import Detection

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class DepthEstimate:
    """
    Metric depth and 3D position for one detection centroid.

    Attributes
    ----------
    detection_idx : index into the detections list this corresponds to
    depth_m       : estimated metric depth in metres (Z in camera frame)
                    0.0 if estimation failed or model unavailable
    position_3d   : (3,) float64 array [X, Y, Z] in camera frame (metres)
                    None if depth estimation failed or intrinsics unavailable
    depth_map     : (H, W) float32 depth map — None if not available
    """
    detection_idx: int
    depth_m:       float
    position_3d:   Optional[np.ndarray] = None
    depth_map:     Optional[np.ndarray] = None


# ---------------------------------------------------------------------------
# ABC
# ---------------------------------------------------------------------------

class DepthEstimator(ABC):
    """
    Abstract base class for monocular depth estimators.

    All implementations return one DepthEstimate per Detection,
    in the same order as the input detections list.
    """

    @abstractmethod
    def estimate(
        self,
        frame:      CameraFrame,
        detections: List[Detection],
    ) -> List[DepthEstimate]:
        """Estimate metric depth for each detection centroid."""
        ...

    @abstractmethod
    def warmup(self) -> None:
        """Warmup inference — call once after construction."""
        ...

    @property
    @abstractmethod
    def mean_inference_ms(self) -> float:
        """Mean inference latency in milliseconds."""
        ...


# ---------------------------------------------------------------------------
# Pixel -> 3D projection
# ---------------------------------------------------------------------------

def project_to_3d(
    u:     float,
    v:     float,
    depth: float,
    intr:  CameraIntrinsics,
) -> np.ndarray:
    """
    Project pixel (u, v) at metric depth Z to camera-frame 3D.

    Parameters
    ----------
    u     : pixel column
    v     : pixel row
    depth : metric depth in metres
    intr  : calibrated camera intrinsics

    Returns
    -------
    (3,) float64 [X, Y, Z] in camera frame
    """
    X = (u - intr.cx) * depth / intr.fx
    Y = (v - intr.cy) * depth / intr.fy
    return np.array([X, Y, depth], dtype=np.float64)


# ---------------------------------------------------------------------------
# Depth Anything V2
# ---------------------------------------------------------------------------

class DepthAnythingEstimator(DepthEstimator):
    """
    Metric monocular depth using Depth Anything V2.

    Depth Anything V2 (Yang et al. 2024) is trained on 595K labelled
    images plus 62M unlabelled images via self-supervised distillation.
    The Metric Indoor variant produces absolute depth in metres without
    scale ambiguity.

    Model variants
    --------------
    "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"  ~10ms/frame
    "depth-anything/Depth-Anything-V2-Metric-Indoor-Base-hf"   ~20ms/frame
    "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf"  ~45ms/frame

    Install
    -------
    pip install transformers accelerate

    Usage
    -----
    estimator = DepthAnythingEstimator(device="cuda")
    estimator.warmup()
    estimates = estimator.estimate(frame, detections)
    for est in estimates:
        print(f"depth={est.depth_m:.2f}m  3D={est.position_3d}")
    """

    def __init__(
        self,
        device:     str = "cuda",
        model_name: str = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf",
    ) -> None:
        self._device     = device
        self._model_name = model_name
        self._pipe       = None
        self._ready      = False
        self._latencies: list[float] = []
        self._load_model()

    def _load_model(self) -> None:
        try:
            from transformers import pipeline as hf_pipeline
            log.info(
                "Loading Depth Anything V2: %s on %s ...",
                self._model_name, self._device,
            )
            t0        = time.monotonic()
            device_id = 0 if "cuda" in self._device else -1

            self._pipe = hf_pipeline(
                task="depth-estimation",
                model=self._model_name,
                device=device_id,
            )
            log.info(
                "Depth Anything V2 loaded in %.1fs",
                time.monotonic() - t0,
            )
            self._ready = True

        except Exception as e:
            log.warning(
                "Depth Anything V2 load failed: %s\n"
                "  Install: pip install transformers accelerate\n"
                "  Depth module will return zero depth (NullDepthEstimator).",
                e,
            )
            self._ready = False

    def warmup(self) -> None:
        """Run one dummy inference to initialise CUDA kernels."""
        if not self._ready:
            return
        try:
            from PIL import Image as PILImage
            dummy = PILImage.fromarray(
                np.zeros((480, 640, 3), dtype=np.uint8)
            )
            self._pipe(dummy)
            log.info(
                "Depth Anything V2 warmup complete. "
                "Latency: %.1f ms",
                self.mean_inference_ms,
            )
        except Exception as e:
            log.warning("Depth Anything V2 warmup failed: %s", e)

    def estimate(
        self,
        frame:      CameraFrame,
        detections: List[Detection],
    ) -> List[DepthEstimate]:
        """
        Estimate metric depth for each detection centroid.

        Runs one full-frame inference, then samples a 5x5 median patch
        at each detection centroid. Median is more robust than
        single-pixel sampling at object boundaries.
        """
        if not detections:
            return []

        depth_map = self._run_inference(frame)
        results   = []

        for i, det in enumerate(detections):
            x1, y1, x2, y2 = det.bbox_xyxy
            u = (x1 + x2) / 2.0
            v = (y1 + y2) / 2.0

            depth_m = self._sample_depth(depth_map, u, v, frame)

            pos_3d = None
            if frame.intrinsics is not None and depth_m > 0.0:
                pos_3d = project_to_3d(u, v, depth_m, frame.intrinsics)

            results.append(DepthEstimate(
                detection_idx=i,
                depth_m=depth_m,
                position_3d=pos_3d,
                depth_map=depth_map,
            ))

        return results

    def _run_inference(
        self, frame: CameraFrame
    ) -> Optional[np.ndarray]:
        """Run Depth Anything V2. Returns (H, W) float32 depth in metres."""
        if not self._ready or self._pipe is None:
            return None
        try:
            from PIL import Image as PILImage
            rgb     = frame.image[:, :, ::-1].copy()
            pil_img = PILImage.fromarray(rgb.astype(np.uint8))

            t0     = time.monotonic()
            result = self._pipe(pil_img)
            self._latencies.append((time.monotonic() - t0) * 1000)

            return np.array(result["depth"], dtype=np.float32)

        except Exception as e:
            log.debug("Depth inference error (suppressed): %s", e)
            return None

    def _sample_depth(
        self,
        depth_map: Optional[np.ndarray],
        u:         float,
        v:         float,
        frame:     CameraFrame,
    ) -> float:
        """Sample 5x5 median patch centred on (u, v)."""
        if depth_map is None:
            return 0.0

        H, W = depth_map.shape[:2]
        col  = int(np.clip(u, 0, W - 1))
        row  = int(np.clip(v, 0, H - 1))
        r0   = max(0, row - 2)
        r1   = min(H, row + 3)
        c0   = max(0, col - 2)
        c1   = min(W, col + 3)
        patch = depth_map[r0:r1, c0:c1]

        return float(np.median(patch)) if patch.size > 0 else 0.0

    @property
    def mean_inference_ms(self) -> float:
        if not self._latencies:
            return 0.0
        return float(np.mean(self._latencies[-30:]))

    @property
    def is_ready(self) -> bool:
        return self._ready


# ---------------------------------------------------------------------------
# Null estimator — fallback
# ---------------------------------------------------------------------------

class NullDepthEstimator(DepthEstimator):
    """
    No-op depth estimator. Returns zero depth for all detections.
    Used when depth is disabled in config or model failed to load.
    Ensures the pipeline runs without depth support with no code changes.
    """

    def estimate(
        self,
        frame:      CameraFrame,
        detections: List[Detection],
    ) -> List[DepthEstimate]:
        return [
            DepthEstimate(detection_idx=i, depth_m=0.0, position_3d=None)
            for i in range(len(detections))
        ]

    def warmup(self) -> None:
        pass

    @property
    def mean_inference_ms(self) -> float:
        return 0.0
