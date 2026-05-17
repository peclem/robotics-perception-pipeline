"""
DPVO-backed ego-pose estimator.

Wraps Princeton-VL's DPVO (Deep Patch Visual Odometry, Teed et al. 2023)
as a drop-in PoseEstimator. Monocular only — scale is not metric until
anchored against the depth estimator (see scale_anchor.py, task #15).

Why DPVO over alternatives
--------------------------
On this hardware (RTX 4070Ti, torch 2.11+cu130):
  - 16.9 ms median per frame at 640×480, real video
  - 20.5 ms median at 1280×720
  - 359–775 MB VRAM
  - MIT license; PyTorch-native
Picked over DROID-SLAM (VRAM-bound at this config) and ORB-SLAM3
(classical, not deep-SOTA). Benchmark methodology in
scripts/benchmark_dpvo_latency.py.

Rate decoupling (the "cheat")
-----------------------------
DPVO runs at a stride relative to the camera frame rate. With
stride=2, pose updates at 15 Hz while the rest of the pipeline runs
at 30 Hz. This matches standard robotics convention — different
sensors/estimators run at independent rates and the transform tree
handles staleness via timestamps. Between DPVO updates, callers see
the most recent CameraPose; the SceneGraph applies it to the latest
position_3d, which is acceptable because 33 ms of pose staleness is
well below planner update rates (Nav2: 5–10 Hz).

Bootstrap
---------
DPVO needs ~8 keyframes (≈ 30–50 input frames) before is_initialized
becomes True. estimate() returns None during this period; the
SceneGraph then leaves position_world unset, falling back to
position_3d (camera-frame metric) — which is still useful for
relative obstacle reasoning.

Coordinate convention
---------------------
DPVO stores poses as world←camera SE(3); get_pose(t).inv() yields
camera-in-world (the conventional CameraPose form). Stored as
[x, y, z, qx, qy, qz, qw] — quaternion in scalar-last format.

Reference
---------
Teed, Lipson & Deng — DPVO (NeurIPS 2023). arXiv:2208.04726
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np

from perception.camera_interface import CameraFrame
from perception.pose_estimator import CameraPose, PoseEstimator

log = logging.getLogger(__name__)

# DPVO lives in third_party/ — add to path on first use, not at module
# import, so that the project still imports cleanly without DPVO present.
_DPVO_PATH = Path(__file__).resolve().parent.parent / "third_party" / "DPVO"


def _ensure_dpvo_on_path() -> None:
    p = str(_DPVO_PATH)
    if p not in sys.path:
        sys.path.insert(0, p)


def _quat_to_rotation(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Scalar-last quaternion → 3×3 rotation matrix."""
    n = np.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
    if n == 0.0:
        return np.eye(3, dtype=np.float64)
    qx, qy, qz, qw = qx/n, qy/n, qz/n, qw/n
    return np.array([
        [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw),     1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw),     1 - 2*(qx*qx + qy*qy)],
    ], dtype=np.float64)


class DPVOPoseEstimator(PoseEstimator):
    """
    DPVO PoseEstimator backend.

    Parameters
    ----------
    config : full pipeline config dict. Reads `pose_estimator` section:
        type             : must be 'dpvo' (sanity check)
        checkpoint       : path to dpvo.pth (default: third_party/DPVO/models/dpvo.pth)
        stride           : run DPVO every Nth input frame (default 2)
        patches_per_frame: DPVO's main accuracy/speed knob (default 96)
    """

    def __init__(self, config: dict) -> None:
        pe = config.get("pose_estimator", {})

        self._stride: int = int(pe.get("stride", 2))
        if self._stride < 1:
            raise ValueError(f"pose_estimator.stride must be >= 1, got {self._stride}")

        ckpt_default = _DPVO_PATH / "models" / "dpvo.pth"
        self._checkpoint: str = str(pe.get("checkpoint", ckpt_default))
        if not Path(self._checkpoint).exists():
            raise FileNotFoundError(
                f"DPVO checkpoint not found: {self._checkpoint}\n"
                "Run: cd third_party/DPVO && wget "
                "https://www.dropbox.com/s/nap0u8zslspdwm4/models.zip?dl=1 "
                "-O models.zip && unzip -o models.zip -d models/"
            )

        _ensure_dpvo_on_path()
        from dpvo.config import cfg as _dpvo_cfg
        from dpvo.dpvo import DPVO as _DPVO

        self._DPVO = _DPVO
        self._cfg = _dpvo_cfg.clone()
        self._cfg.PATCHES_PER_FRAME = int(pe.get("patches_per_frame", 96))

        # Lazy slam construction — needs H, W from first frame.
        self._slam = None
        self._counter: int = 0
        self._fed: int = 0
        self._latest: Optional[CameraPose] = None

        log.info(
            "DPVOPoseEstimator: stride=%d patches=%d checkpoint=%s",
            self._stride, self._cfg.PATCHES_PER_FRAME, self._checkpoint,
        )

    def estimate(self, frame: CameraFrame) -> Optional[CameraPose]:
        import torch   # Local import — DPVO already requires torch

        # Lazy SLAM init on first frame (needs H, W).
        if self._slam is None:
            h, w = frame.image.shape[:2]
            self._slam = self._DPVO(
                self._cfg, self._checkpoint,
                ht=h, wd=w, viz=False,
            )
            log.info("DPVO initialised for %dx%d frames", w, h)

        # Stride: only feed every Nth frame to DPVO. The pipeline still
        # runs at full rate; pose updates are at 30/stride Hz.
        if self._counter % self._stride == 0:
            img = torch.from_numpy(frame.image).permute(2, 0, 1).cuda()
            K = np.array([
                frame.intrinsics.fx, frame.intrinsics.fy,
                frame.intrinsics.cx, frame.intrinsics.cy,
            ], dtype=np.float32)
            K_t = torch.from_numpy(K).cuda()

            with torch.no_grad():
                self._slam(self._counter, img, K_t)
            self._fed += 1

            # DPVO needs ~8 keyframes before producing valid poses.
            # Note: slam.get_pose(t) only works after slam.terminate() has
            # populated self.traj. For live use, read the latest keyframe
            # pose directly from the pose graph buffer and invert via
            # lietorch (DPVO stores world←camera; we want camera-in-world).
            if self._slam.is_initialized and self._slam.n > 0:
                try:
                    # DPVO bundles its own lietorch; not a top-level package.
                    from dpvo.lietorch import SE3
                    raw = self._slam.pg.poses_[self._slam.n - 1].unsqueeze(0)
                    inv = SE3(raw).inv()
                    data = inv.data.cpu().numpy().reshape(-1)
                    # [x, y, z, qx, qy, qz, qw]
                    self._latest = CameraPose(
                        R=_quat_to_rotation(data[3], data[4], data[5], data[6]),
                        t=data[:3].astype(np.float64),
                        timestamp=frame.timestamp,
                        frame_idx=frame.frame_idx,
                        confidence=1.0,
                        source="dpvo",
                    )
                except Exception as exc:
                    log.warning(
                        "DPVO pose read at frame %d failed: %s — returning last known",
                        self._counter, exc,
                    )

        self._counter += 1
        return self._latest

    def reset(self) -> None:
        """Drop SLAM state. Call between sequences."""
        self._slam = None
        self._counter = 0
        self._fed = 0
        self._latest = None

    @property
    def is_initialised(self) -> bool:
        return (
            self._slam is not None
            and getattr(self._slam, "is_initialized", False)
            and self._latest is not None
        )

    def __repr__(self) -> str:
        return (
            f"DPVOPoseEstimator(stride={self._stride}, "
            f"patches={self._cfg.PATCHES_PER_FRAME}, "
            f"init={self.is_initialised}, fed={self._fed})"
        )
