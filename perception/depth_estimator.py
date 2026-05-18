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

import cv2
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


# ---------------------------------------------------------------------------
# Stereo backend (classical SGBM)
# ---------------------------------------------------------------------------

class StereoSGBMDepthEstimator(DepthEstimator):
    """
    Classical stereo depth from cv2.StereoSGBM (Semi-Global Block Matching).

    Why classical SGBM and not a neural stereo network
    --------------------------------------------------
    Neural stereo (IGEV, FoundationStereo, CREStereo) wins on
    benchmarks but adds a 1-3 GB GPU model, a CUDA build dependency,
    and bigger per-frame latency. SGBM is well-engineered, runs on
    CPU at ~30 ms for 640×480, and is genuinely the right choice
    for many production robots that don't have GPU headroom. A
    neural backend can slot in later under the same DepthEstimator
    ABC without touching downstream code.

    Inputs
    ------
    frame.image       : left-eye BGR image
    frame.right_image : right-eye BGR image (must be present)
    frame.intrinsics.baseline_m : stereo baseline in metres (must be > 0)
    frame.intrinsics.fx         : focal length in pixels

    Output
    ------
    Per-detection depth derived from the disparity at the detection
    centroid. depth = fx * baseline / disparity. Invalid pixels
    (disparity ≤ 0) yield depth_m = 0.0 and position_3d = None.

    Reference
    ---------
    Hirschmüller (2008) — Stereo Processing by Semi-Global Matching
                          and Mutual Information. IEEE TPAMI 30(2).
    """

    def __init__(
        self,
        min_disparity:   int = 0,
        num_disparities: int = 96,   # must be divisible by 16
        block_size:      int = 7,    # odd, typical range 3-11
        device:          str = "cpu",
    ) -> None:
        # device unused — SGBM is CPU-only. Kept on the signature for
        # API symmetry with DepthAnythingEstimator.
        self._sgbm = cv2.StereoSGBM_create(
            minDisparity=min_disparity,
            numDisparities=num_disparities,
            blockSize=block_size,
            P1=8  * 3 * block_size * block_size,
            P2=32 * 3 * block_size * block_size,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=32,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        )
        self._inference_times_ms: list[float] = []

    def warmup(self) -> None:
        # SGBM has no state — first call's cost is dominated by buffer
        # allocation. Do a dummy compute so the first real call isn't
        # an outlier in latency stats.
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        gl = cv2.cvtColor(dummy, cv2.COLOR_BGR2GRAY)
        self._sgbm.compute(gl, gl)

    def estimate(
        self,
        frame:      CameraFrame,
        detections: List[Detection],
    ) -> List[DepthEstimate]:
        if frame.right_image is None:
            log.warning(
                "StereoSGBMDepthEstimator: frame.right_image is None "
                "— stereo depth unavailable, returning zeros."
            )
            return [DepthEstimate(detection_idx=i, depth_m=0.0)
                    for i in range(len(detections))]
        if frame.intrinsics.baseline_m <= 0:
            log.warning(
                "StereoSGBMDepthEstimator: intrinsics.baseline_m = %.3f m "
                "(must be > 0). Returning zeros.",
                frame.intrinsics.baseline_m,
            )
            return [DepthEstimate(detection_idx=i, depth_m=0.0)
                    for i in range(len(detections))]

        t0 = time.monotonic()
        gl = cv2.cvtColor(frame.image,       cv2.COLOR_BGR2GRAY)
        gr = cv2.cvtColor(frame.right_image, cv2.COLOR_BGR2GRAY)
        # SGBM output is int16, fixed-point with 4 fractional bits.
        disparity = self._sgbm.compute(gl, gr).astype(np.float32) / 16.0
        self._inference_times_ms.append(
            (time.monotonic() - t0) * 1000.0
        )

        fx = frame.intrinsics.fx
        baseline = frame.intrinsics.baseline_m
        results: List[DepthEstimate] = []
        H, W = disparity.shape
        for i, det in enumerate(detections):
            x1, y1, x2, y2 = det.bbox_xyxy
            u = int(np.clip((x1 + x2) / 2.0, 0, W - 1))
            v = int(np.clip((y1 + y2) / 2.0, 0, H - 1))
            d_px = float(disparity[v, u])
            if d_px <= 0.5:
                # SGBM uses -1 for invalid; treat very small disparities
                # as far/unknown to avoid divide-by-zero blow-ups.
                results.append(DepthEstimate(
                    detection_idx=i, depth_m=0.0, position_3d=None,
                ))
                continue
            depth_m = fx * baseline / d_px
            pos_3d = project_to_3d(float(u), float(v), depth_m,
                                   frame.intrinsics)
            results.append(DepthEstimate(
                detection_idx=i, depth_m=depth_m, position_3d=pos_3d,
            ))
        return results

    @property
    def mean_inference_ms(self) -> float:
        return float(np.mean(self._inference_times_ms[-30:]))  if \
            self._inference_times_ms else 0.0

    def __repr__(self) -> str:
        return (
            f"StereoSGBMDepthEstimator("
            f"numDisp={self._sgbm.getNumDisparities()}, "
            f"blockSize={self._sgbm.getBlockSize()})"
        )


# ---------------------------------------------------------------------------
# Stereo backend (neural — RAFT-Stereo, Princeton-VL)
# ---------------------------------------------------------------------------

class RAFTStereoDepthEstimator(DepthEstimator):
    """
    Neural stereo via RAFT-Stereo (Lipson, Teed, Deng — 3DV 2021).

    Why RAFT-Stereo over IGEV / FoundationStereo / CREStereo
    --------------------------------------------------------
    - **Pure PyTorch.** No custom CUDA extensions. The DPVO experience
      ([[dpvo-build-patches]]) made it clear that every new
      CUDA-extension dependency is a tax on CUDA/PyTorch version
      churn. RAFT-Stereo builds itself out of standard torch ops.
    - **Open weights, MIT license, well-maintained until 2023.**
    - **Same accuracy tier as IGEV in published Middlebury / ETH3D /
      KITTI** for indoor scenes; IGEV pulls ahead on hard outdoor
      cases.
    - **Familiar idiom**: same lazy-load + graceful-fallback pattern
      already used by DepthAnythingEstimator.

    When a real stereo camera arrives and accuracy on hard cases
    becomes the bottleneck, FoundationStereo (2024, zero-shot
    generalist) or IGEV-Stereo can slot in as siblings under the same
    DepthEstimator ABC.

    Install
    -------
    See `README.md` — `RAFT-Stereo setup`. Briefly:
        cd third_party
        git clone https://github.com/princeton-vl/RAFT-Stereo.git
        cd RAFT-Stereo && bash download_models.sh

    The default `raftstereo-middlebury.pth` checkpoint is ~30 MB and
    is the strongest generalist for indoor scenes. Use `-eth3d.pth`
    for outdoor low-texture, `-realtime.pth` if you need <10 ms.

    Inputs / outputs are the same contract as StereoSGBMDepthEstimator:
    `frame.right_image` must be present, `frame.intrinsics.baseline_m`
    must be > 0; depth = fx × baseline / disparity per detection
    centroid; downstream sees one DepthEstimate per detection.
    """

    def __init__(
        self,
        repo_dir:   str = "third_party/RAFT-Stereo",
        checkpoint: str = "third_party/RAFT-Stereo/models/raftstereo-middlebury.pth",
        device:     str = "cuda",
        iters:      int = 16,   # RAFT-Stereo iterations; 16 is the published default
    ) -> None:
        self._repo_dir   = repo_dir
        self._checkpoint = checkpoint
        self._device     = device
        self._iters      = int(iters)
        self._model      = None
        self._padder_cls = None
        self._ready      = False
        self._latencies: list[float] = []
        self._load_model()

    def _load_model(self) -> None:
        """
        Lazy import + checkpoint load. Every branch that can fail is
        wrapped — a missing repo, missing weights, or upstream API
        drift downgrades the estimator to a no-op rather than crashing
        the pipeline. Same discipline as DepthAnythingEstimator.
        """
        import os
        import sys

        if not os.path.isdir(self._repo_dir):
            log.warning(
                "RAFTStereoDepthEstimator: repo not found at %r. "
                "Clone with `git clone https://github.com/princeton-vl/"
                "RAFT-Stereo.git %s`. Estimator will return zeros.",
                self._repo_dir, self._repo_dir,
            )
            self._ready = False
            return
        if not os.path.isfile(self._checkpoint):
            log.warning(
                "RAFTStereoDepthEstimator: checkpoint not found at %r. "
                "Run `bash %s/download_models.sh`. Estimator will return zeros.",
                self._checkpoint, self._repo_dir,
            )
            self._ready = False
            return

        try:
            import torch
            sys.path.insert(0, os.path.abspath(self._repo_dir))
            sys.path.insert(0, os.path.abspath(os.path.join(self._repo_dir, "core")))
            from argparse import Namespace
            from raft_stereo import RAFTStereo
            from utils.utils import InputPadder

            # RAFT-Stereo defaults from the README + middlebury config.
            args = Namespace(
                hidden_dims=[128, 128, 128],
                corr_implementation="reg",
                shared_backbone=False,
                corr_levels=4,
                corr_radius=4,
                n_downsample=2,
                slow_fast_gru=False,
                n_gru_layers=3,
                mixed_precision=False,
                valid_iters=self._iters,
            )
            t0 = time.monotonic()
            model = torch.nn.DataParallel(RAFTStereo(args), device_ids=[0])
            state = torch.load(self._checkpoint, map_location=self._device)
            model.load_state_dict(state)
            self._model = model.module.to(self._device).eval()
            self._padder_cls = InputPadder
            log.info(
                "RAFT-Stereo loaded from %s in %.1fs (device=%s, iters=%d)",
                self._checkpoint, time.monotonic() - t0,
                self._device, self._iters,
            )
            self._ready = True

        except Exception as exc:
            log.warning(
                "RAFT-Stereo load failed: %s\n"
                "  Common causes: PyTorch / RAFT-Stereo API drift, missing "
                "core/ subdir, mismatched state_dict keys. "
                "Falling back to zero depth.",
                exc,
            )
            self._ready = False

    def warmup(self) -> None:
        if not self._ready:
            return
        try:
            import torch
            dummy = np.zeros((480, 640, 3), dtype=np.uint8)
            self._infer_disparity(dummy, dummy)
            log.info(
                "RAFT-Stereo warmup complete. Latency: %.1f ms",
                self.mean_inference_ms,
            )
        except Exception as exc:
            log.warning("RAFT-Stereo warmup failed: %s", exc)
            self._ready = False

    def estimate(
        self,
        frame:      CameraFrame,
        detections: List[Detection],
    ) -> List[DepthEstimate]:
        if not self._ready:
            return [DepthEstimate(detection_idx=i, depth_m=0.0)
                    for i in range(len(detections))]
        if frame.right_image is None:
            log.warning(
                "RAFTStereoDepthEstimator: frame.right_image is None "
                "— stereo depth unavailable, returning zeros."
            )
            return [DepthEstimate(detection_idx=i, depth_m=0.0)
                    for i in range(len(detections))]
        if frame.intrinsics.baseline_m <= 0:
            log.warning(
                "RAFTStereoDepthEstimator: intrinsics.baseline_m = %.3f m "
                "(must be > 0). Returning zeros.",
                frame.intrinsics.baseline_m,
            )
            return [DepthEstimate(detection_idx=i, depth_m=0.0)
                    for i in range(len(detections))]

        try:
            disparity = self._infer_disparity(frame.image, frame.right_image)
        except Exception as exc:
            log.warning(
                "RAFT-Stereo inference failed (%s); returning zeros.", exc,
            )
            return [DepthEstimate(detection_idx=i, depth_m=0.0)
                    for i in range(len(detections))]

        fx = frame.intrinsics.fx
        baseline = frame.intrinsics.baseline_m
        H, W = disparity.shape
        results: List[DepthEstimate] = []
        for i, det in enumerate(detections):
            x1, y1, x2, y2 = det.bbox_xyxy
            u = int(np.clip((x1 + x2) / 2.0, 0, W - 1))
            v = int(np.clip((y1 + y2) / 2.0, 0, H - 1))
            d_px = float(disparity[v, u])
            if d_px <= 0.5:
                results.append(DepthEstimate(
                    detection_idx=i, depth_m=0.0, position_3d=None,
                ))
                continue
            depth_m = fx * baseline / d_px
            pos_3d = project_to_3d(float(u), float(v), depth_m,
                                   frame.intrinsics)
            results.append(DepthEstimate(
                detection_idx=i, depth_m=depth_m, position_3d=pos_3d,
            ))
        return results

    def _infer_disparity(
        self, left_bgr: np.ndarray, right_bgr: np.ndarray,
    ) -> np.ndarray:
        """
        Run RAFT-Stereo on one stereo pair and return (H, W) disparity
        in pixels. Disparity output convention: positive values where
        the left view sees the object at higher u than the right view
        (standard stereo). NaN / negative outputs are returned as-is —
        the per-detection sampling treats <0.5 as invalid.
        """
        import torch
        # transformers + most stereo nets expect RGB.
        left_rgb  = left_bgr[:,  :, ::-1].copy()
        right_rgb = right_bgr[:, :, ::-1].copy()
        # → (B=1, C=3, H, W) float32 tensors on device.
        def _to_tensor(img):
            t = torch.from_numpy(img.astype(np.float32)).permute(2, 0, 1)
            return t.unsqueeze(0).to(self._device)
        l_t = _to_tensor(left_rgb)
        r_t = _to_tensor(right_rgb)
        # RAFT-Stereo wants the input padded to a multiple of 32.
        padder = self._padder_cls(l_t.shape, divis_by=32)
        l_t, r_t = padder.pad(l_t, r_t)

        t0 = time.monotonic()
        with torch.no_grad():
            # RAFT-Stereo returns (flow_predictions, flow_up) when
            # test_mode=True; flow_up is the up-sampled final disparity
            # at full resolution, shape (B, 1, H, W). RAFT outputs
            # NEGATIVE disparity (left flow direction); flip sign.
            _, flow_up = self._model(
                l_t, r_t, iters=self._iters, test_mode=True,
            )
        self._latencies.append((time.monotonic() - t0) * 1000.0)

        # Un-pad and convert to numpy.
        flow_up = padder.unpad(flow_up).squeeze().cpu().numpy()
        disparity = -flow_up   # RAFT-Stereo convention
        return disparity.astype(np.float32)

    @property
    def mean_inference_ms(self) -> float:
        return float(np.mean(self._latencies[-30:])) if self._latencies else 0.0

    @property
    def is_ready(self) -> bool:
        return self._ready

    def __repr__(self) -> str:
        return (
            f"RAFTStereoDepthEstimator(device={self._device!r}, "
            f"iters={self._iters}, ready={self._ready})"
        )
