"""
CODa (UT Campus Object Dataset) replay as a CameraInterface backend.

CODa (Zhang et al., T-RO 2024) is a ground-robot perception dataset:
a Clearpath Husky driving campus sidewalks, indoor and outdoor, with
hardware-synchronised stereo cameras + LiDAR and globally consistent
6-DoF poses from LiDAR SLAM (LeGO-LOAM). It is the outdoor / sidewalk
counterpart to TUM RGB-D for this pipeline's perception validation —
the wheeled-robot motion profile the stack will actually see.

This backend replays one sequence's left rectified camera (cam0) as a
monocular source, exposing the ground-truth pose via pose_gt() for
offline trajectory evaluation. depth_gt() returns None: CODa has no
dense depth (outdoor active depth fails in sunlight) — only LiDAR,
whose sparse-projection depth is deferred.

Directory layout expected (extracted CODa_vslam sequence):

    <sequence_dir>/
        2d_rect/cam0/<SEQ>/2d_rect_cam0_<SEQ>_<N>.jpg   rectified frames
        poses/dense_global/<SEQ>.txt   'ts x y z qw qx qy qz' per frame
        calibrations/<SEQ>/calib_cam0_intrinsics.yaml

Frame / pose association
------------------------
cam0 is hardware-synchronised to the LiDAR and the dense_global pose
file has one line per synchronised frame, so association is by integer
frame index N — no nearest-timestamp matching (unlike TUMDatasetCamera).

Intrinsics
----------
The replayed frames are rectified (2d_rect), so the calibration's
projection_matrix — not the raw camera_matrix — gives the effective
pinhole intrinsics, and distortion is zero.

Timestamps
----------
CODa stores absolute Unix timestamps in the pose file. As with
TUMDatasetCamera, CameraFrame.timestamp is emitted as a monotonic
offset (ts - first_ts) so it starts at 0.0 and preserves the true
inter-frame dt; the absolute stamp is exposed via current_timestamp.
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import yaml

from perception.camera_interface import (
    CameraFrame,
    CameraInterface,
    CameraIntrinsics,
)
from perception.tum_dataset_camera import _quat_to_rot

# A 2d_rect cam0 frame file: 2d_rect_cam0_<SEQ>_<FRAME>.jpg
_FRAME_RE = re.compile(r"_(\d+)\.jpg$", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Parsing helpers (module-level — reused by the offline evaluation script)
# ---------------------------------------------------------------------------

def read_coda_trajectory(path: str | Path) -> List[Tuple[float, np.ndarray]]:
    """
    Parse a CODa dense_global pose file into (timestamp, 4x4 pose) pairs.

    Each line is 'ts x y z qw qx qy qz' (note: w-first quaternion, unlike
    TUM's w-last). Each pose is the homogeneous world-from-camera transform
    T_world_cam. Line index == frame index. Returned in file order.
    """
    poses: List[Tuple[float, np.ndarray]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            ts = float(parts[0])
            x, y, z, qw, qx, qy, qz = (float(v) for v in parts[1:8])
            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = _quat_to_rot(qx, qy, qz, qw)
            T[:3, 3] = (x, y, z)
            poses.append((ts, T))
    return poses


def coda_intrinsics_from_yaml(calib_path: str | Path) -> CameraIntrinsics:
    """
    Build CameraIntrinsics from a CODa calib_cam*_intrinsics.yaml.

    The frames are rectified, so the projection_matrix gives the effective
    pinhole parameters and distortion is zero. Falls back to camera_matrix
    if no projection_matrix is present.
    """
    with open(calib_path) as f:
        calib = yaml.safe_load(f)

    width = int(calib["image_width"])
    height = int(calib["image_height"])

    proj = calib.get("projection_matrix")
    if proj is not None:
        d = proj["data"]              # 3x4 row-major: fx 0 cx 0 / 0 fy cy 0 / ...
        fx, cx, fy, cy = d[0], d[2], d[5], d[6]
    else:
        d = calib["camera_matrix"]["data"]   # 3x3: fx 0 cx / 0 fy cy / 0 0 1
        fx, cx, fy, cy = d[0], d[2], d[4], d[5]

    return CameraIntrinsics(
        fx=float(fx), fy=float(fy), cx=float(cx), cy=float(cy),
        width=width, height=height,
        dist_coeffs=np.zeros(5, dtype=np.float64),  # rectified → undistorted
    )


def _detect_sequence_id(seq_dir: Path) -> str:
    """Find the single sequence-id subdirectory under 2d_rect/cam0/."""
    cam0 = seq_dir / "2d_rect" / "cam0"
    if not cam0.is_dir():
        raise FileNotFoundError(
            f"{cam0} missing — not an extracted CODa sequence directory."
        )
    subdirs = sorted(p.name for p in cam0.iterdir() if p.is_dir())
    if not subdirs:
        raise RuntimeError(f"No sequence-id subdirectory under {cam0}.")
    if len(subdirs) > 1:
        warnings.warn(
            f"Multiple sequence ids under {cam0} ({subdirs}) — using {subdirs[0]!r}.",
            stacklevel=2,
        )
    return subdirs[0]


# ---------------------------------------------------------------------------
# Backend: CODa sequence replay
# ---------------------------------------------------------------------------

class CODaDatasetCamera(CameraInterface):
    """
    Replay one CODa sequence's left rectified camera as a camera source.

    Deterministic, like VideoFileCamera — the same directory always yields
    the same frame sequence. Colour frames drive the pipeline; the
    index-associated ground-truth pose is held aside for the offline
    evaluation script via pose_gt(). depth_gt() always returns None.

    Parameters
    ----------
    sequence_dir : path to an extracted CODa sequence directory.
    max_frames   : stop after this many frames (None = whole sequence).
    """

    def __init__(
        self,
        config: dict,
        sequence_dir: str | Path,
        *,
        max_frames: Optional[int] = None,
    ):
        super().__init__(config)
        self._dir = Path(sequence_dir)
        self._max_frames = max_frames

        self._seq_id: Optional[str] = None
        self._intrinsics: Optional[CameraIntrinsics] = None
        self._frame_paths: List[Path] = []
        self._frame_ids: List[int] = []
        self._traj: List[Tuple[float, np.ndarray]] = []
        self._t0: float = 0.0

        # Ground truth for the most recently returned frame.
        self._cur_tum_ts: Optional[float] = None
        self._cur_pose_gt: Optional[np.ndarray] = None

    # -- lifecycle ----------------------------------------------------------

    def open(self) -> None:
        if not self._dir.is_dir():
            raise FileNotFoundError(f"CODa sequence directory not found: {self._dir}")

        self._seq_id = _detect_sequence_id(self._dir)
        frame_dir = self._dir / "2d_rect" / "cam0" / self._seq_id

        # Sort frame files by their integer frame index.
        indexed: List[Tuple[int, Path]] = []
        for p in frame_dir.iterdir():
            m = _FRAME_RE.search(p.name)
            if m:
                indexed.append((int(m.group(1)), p))
        indexed.sort(key=lambda r: r[0])
        if not indexed:
            raise RuntimeError(f"CODa sequence {frame_dir} contains no cam0 frames.")
        self._frame_ids = [i for i, _ in indexed]
        self._frame_paths = [p for _, p in indexed]
        if self._max_frames is not None:
            self._frame_ids = self._frame_ids[: self._max_frames]
            self._frame_paths = self._frame_paths[: self._max_frames]

        pose_txt = self._dir / "poses" / "dense_global" / f"{self._seq_id}.txt"
        if pose_txt.exists():
            self._traj = read_coda_trajectory(pose_txt)
            self._t0 = self._traj[self._frame_ids[0]][0] if self._traj else 0.0
        else:
            warnings.warn(
                f"{pose_txt} missing — pose_gt() will return None.",
                stacklevel=2,
            )

        calib = (self._dir / "calibrations" / self._seq_id
                 / "calib_cam0_intrinsics.yaml")
        if not calib.exists():
            raise FileNotFoundError(f"CODa cam0 calibration not found: {calib}")
        self._intrinsics = coda_intrinsics_from_yaml(calib)

        self._is_open = True

    def get_frame(self) -> Optional[CameraFrame]:
        if not self._is_open:
            return None
        if self._frame_idx >= len(self._frame_paths):
            return None  # sequence exhausted — caller stops the loop

        path = self._frame_paths[self._frame_idx]
        frame_id = self._frame_ids[self._frame_idx]
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            warnings.warn(f"CODaDatasetCamera: unreadable frame {path}.", stacklevel=2)
            return None

        # Pose is associated by integer frame index — cam0 is hardware-synced.
        if 0 <= frame_id < len(self._traj):
            ts, pose = self._traj[frame_id]
        else:
            ts, pose = None, None
        self._cur_tum_ts = ts
        self._cur_pose_gt = pose

        frame = CameraFrame(
            image=image,
            timestamp=(ts - self._t0) if ts is not None else float(self._frame_idx),
            frame_idx=self._frame_idx,
            intrinsics=self._intrinsics,
            source_id=f"coda:{self._dir.name}/{self._seq_id}",
        )
        self._frame_idx += 1
        return frame

    def release(self) -> None:
        self._is_open = False

    # -- ground-truth accessors (evaluation only — not pipeline inputs) -----

    def depth_gt(self) -> Optional[np.ndarray]:
        """
        Always None — CODa has no dense depth ground truth. Outdoor depth
        would have to be projected from the sparse LiDAR point cloud, which
        is deferred.
        """
        return None

    def pose_gt(self) -> Optional[np.ndarray]:
        """
        4x4 world-from-camera ground-truth pose for the frame last returned
        by get_frame(), or None when the pose file is missing / short.
        """
        return self._cur_pose_gt

    @property
    def current_timestamp(self) -> Optional[float]:
        """Absolute CODa timestamp of the last returned frame (None before any)."""
        return self._cur_tum_ts

    @property
    def intrinsics(self) -> Optional[CameraIntrinsics]:
        return self._intrinsics

    @property
    def total_frames(self) -> int:
        return len(self._frame_paths)

    @property
    def sequence_id(self) -> Optional[str]:
        """The CODa sequence id detected under 2d_rect/cam0/."""
        return self._seq_id
