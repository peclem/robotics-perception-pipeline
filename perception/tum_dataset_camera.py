"""
TUM RGB-D dataset replay as a CameraInterface backend.

The TUM RGB-D benchmark (Sturm et al., IROS 2012) is the canonical indoor
RGB-D evaluation set: synchronised colour + depth frames plus a 6-DoF
ground-truth trajectory from an external motion-capture system. This
backend replays a sequence directory as a CameraInterface so the live
pipeline runs on it unchanged, and exposes the per-frame depth and pose
ground truth (depth_gt() / pose_gt()) for offline accuracy evaluation —
mirroring SyntheticCamera.get_ground_truth_bboxes().

Directory layout expected (standard TUM extraction):

    <sequence_dir>/
        rgb.txt            'timestamp  rgb/<ts>.png'   index
        depth.txt          'timestamp  depth/<ts>.png' index
        groundtruth.txt    'timestamp  tx ty tz qx qy qz qw'
        rgb/   depth/       the PNG frames

Timestamps
----------
TUM stores absolute Unix timestamps. CameraFrame.timestamp must be a
monotonic offset (see CameraFrame docstring), so frames are emitted with
timestamp = tum_timestamp - first_tum_timestamp. This starts at 0.0,
increases monotonically, and — unlike VideoFileCamera's wall-clock
stamps — preserves the dataset's true inter-frame dt, which the SLAM /
Kalman prediction step depends on. The original absolute timestamp is
kept for ground-truth association and exposed via current_timestamp.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from perception.camera_interface import (
    CameraFrame,
    CameraInterface,
    CameraIntrinsics,
)

# ---------------------------------------------------------------------------
# TUM constants
# ---------------------------------------------------------------------------

# Per-Freiburg-camera factory calibration, OpenCV convention. Published at
# https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats
# A TUM sequence directory name contains 'freiburg1', 'freiburg2' or
# 'freiburg3' (also abbreviated fr1/fr2/fr3); intrinsics are picked from it.
_TUM_INTRINSICS: dict[str, dict] = {
    "freiburg1": dict(
        fx=517.306408, fy=516.469215, cx=318.643040, cy=255.313989,
        dist=[0.262383, -0.953104, -0.005358, 0.002628, 1.163314],
    ),
    "freiburg2": dict(
        fx=520.908620, fy=521.007327, cx=325.141442, cy=249.701764,
        dist=[0.231222, -0.784899, -0.003257, -0.000105, 0.917205],
    ),
    "freiburg3": dict(
        fx=535.4, fy=539.2, cx=320.1, cy=247.6,
        dist=[0.0, 0.0, 0.0, 0.0, 0.0],
    ),
}

# TUM depth PNGs are 16-bit; metric depth = pixel_value / depth_factor.
TUM_DEPTH_FACTOR: float = 5000.0


# ---------------------------------------------------------------------------
# Parsing helpers (module-level — reused by the offline evaluation script)
# ---------------------------------------------------------------------------

def _read_index(path: Path) -> List[Tuple[float, str]]:
    """Parse a TUM 'timestamp  relative/path' index file (rgb.txt, depth.txt)."""
    rows: List[Tuple[float, str]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            rows.append((float(parts[0]), parts[1]))
    rows.sort(key=lambda r: r[0])
    return rows


def read_tum_trajectory(path: str | Path) -> List[Tuple[float, np.ndarray]]:
    """
    Parse a TUM groundtruth.txt into (timestamp, 4x4 pose) pairs.

    Each pose is the homogeneous world-from-camera transform T_world_cam
    built from the line 'timestamp tx ty tz qx qy qz qw'. Returned sorted
    by timestamp.
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
            tx, ty, tz, qx, qy, qz, qw = (float(x) for x in parts[1:8])
            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = _quat_to_rot(qx, qy, qz, qw)
            T[:3, 3] = (tx, ty, tz)
            poses.append((ts, T))
    poses.sort(key=lambda r: r[0])
    return poses


def _quat_to_rot(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Unit-normalised quaternion (x, y, z, w) -> 3x3 rotation matrix."""
    n = np.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    return np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=np.float64,
    )


def _nearest(query: float, stamps: np.ndarray, max_diff: float) -> Optional[int]:
    """Index of the entry in sorted `stamps` closest to `query`, or None."""
    if stamps.size == 0:
        return None
    pos = int(np.searchsorted(stamps, query))
    best, best_d = None, max_diff
    for cand in (pos - 1, pos):
        if 0 <= cand < stamps.size:
            d = abs(stamps[cand] - query)
            if d <= best_d:
                best, best_d = cand, d
    return best


def tum_intrinsics_for(sequence_dir: str | Path) -> CameraIntrinsics:
    """
    Pick the TUM factory calibration matching the sequence directory name.

    Falls back to the freiburg1 calibration with a warning if the name
    contains no recognisable freiburg tag.
    """
    name = Path(sequence_dir).name.lower()
    key = None
    for tag, alias in (("freiburg1", "fr1"), ("freiburg2", "fr2"),
                       ("freiburg3", "fr3")):
        if tag in name or alias in name:
            key = tag
            break
    if key is None:
        warnings.warn(
            f"TUM sequence {name!r} has no freiburg1/2/3 tag — using the "
            "freiburg1 calibration. Depth-to-metric and pose error may be "
            "biased; rename the directory or pass intrinsics explicitly.",
            stacklevel=2,
        )
        key = "freiburg1"
    p = _TUM_INTRINSICS[key]
    return CameraIntrinsics(
        fx=p["fx"], fy=p["fy"], cx=p["cx"], cy=p["cy"],
        width=640, height=480,
        dist_coeffs=np.array(p["dist"], dtype=np.float64),
    )


# ---------------------------------------------------------------------------
# Backend: TUM RGB-D dataset replay
# ---------------------------------------------------------------------------

class TUMDatasetCamera(CameraInterface):
    """
    Replay a TUM RGB-D sequence directory as a camera source.

    Deterministic — the same directory always yields the same frame
    sequence, like VideoFileCamera. Colour frames drive the pipeline;
    the time-associated depth map and ground-truth pose are held aside
    for the offline evaluation script via depth_gt() / pose_gt().

    Parameters
    ----------
    sequence_dir   : path to an extracted TUM sequence directory.
    depth_factor   : 16-bit-PNG -> metres divisor (TUM standard: 5000).
    max_frames     : stop after this many RGB frames (None = whole sequence).
    assoc_max_diff : max timestamp gap (s) for RGB<->depth and RGB<->pose
                     association; unmatched frames yield None ground truth.
    """

    def __init__(
        self,
        config: dict,
        sequence_dir: str | Path,
        *,
        depth_factor: float = TUM_DEPTH_FACTOR,
        max_frames: Optional[int] = None,
        assoc_max_diff: float = 0.02,
    ):
        super().__init__(config)
        self._dir = Path(sequence_dir)
        self._depth_factor = float(depth_factor)
        self._max_frames = max_frames
        self._assoc_max_diff = float(assoc_max_diff)

        self._intrinsics: Optional[CameraIntrinsics] = None
        self._rgb: List[Tuple[float, str]] = []
        self._depth_stamps = np.empty(0, dtype=np.float64)
        self._depth_paths: List[str] = []
        self._gt_stamps = np.empty(0, dtype=np.float64)
        self._gt_poses: List[np.ndarray] = []
        self._t0: float = 0.0

        # Ground truth for the most recently returned frame.
        self._cur_tum_ts: Optional[float] = None
        self._cur_depth_path: Optional[Path] = None
        self._cur_pose_gt: Optional[np.ndarray] = None

    # -- lifecycle ----------------------------------------------------------

    def open(self) -> None:
        if not self._dir.is_dir():
            raise FileNotFoundError(f"TUM sequence directory not found: {self._dir}")
        rgb_txt = self._dir / "rgb.txt"
        if not rgb_txt.exists():
            raise FileNotFoundError(
                f"{rgb_txt} missing — not a TUM RGB-D sequence directory."
            )

        self._rgb = _read_index(rgb_txt)
        if self._max_frames is not None:
            self._rgb = self._rgb[: self._max_frames]
        if not self._rgb:
            raise RuntimeError(f"TUM sequence {self._dir} contains no RGB frames.")
        self._t0 = self._rgb[0][0]

        depth_txt = self._dir / "depth.txt"
        if depth_txt.exists():
            depth = _read_index(depth_txt)
            self._depth_stamps = np.array([d[0] for d in depth], dtype=np.float64)
            self._depth_paths = [d[1] for d in depth]
        else:
            warnings.warn(
                f"{depth_txt} missing — depth_gt() will return None.",
                stacklevel=2,
            )

        gt_txt = self._dir / "groundtruth.txt"
        if gt_txt.exists():
            traj = read_tum_trajectory(gt_txt)
            self._gt_stamps = np.array([t[0] for t in traj], dtype=np.float64)
            self._gt_poses = [t[1] for t in traj]
        else:
            warnings.warn(
                f"{gt_txt} missing — pose_gt() will return None.",
                stacklevel=2,
            )

        self._intrinsics = tum_intrinsics_for(self._dir)
        self._is_open = True

    def get_frame(self) -> Optional[CameraFrame]:
        if not self._is_open:
            return None
        if self._frame_idx >= len(self._rgb):
            return None  # sequence exhausted — caller stops the loop

        tum_ts, rel_path = self._rgb[self._frame_idx]
        image = cv2.imread(str(self._dir / rel_path), cv2.IMREAD_COLOR)
        if image is None:
            warnings.warn(
                f"TUMDatasetCamera: unreadable RGB frame {rel_path}.",
                stacklevel=2,
            )
            return None

        # Associate depth + ground-truth pose by nearest timestamp.
        self._cur_tum_ts = tum_ts
        d_idx = _nearest(tum_ts, self._depth_stamps, self._assoc_max_diff)
        self._cur_depth_path = (
            self._dir / self._depth_paths[d_idx] if d_idx is not None else None
        )
        g_idx = _nearest(tum_ts, self._gt_stamps, self._assoc_max_diff)
        self._cur_pose_gt = self._gt_poses[g_idx] if g_idx is not None else None

        frame = CameraFrame(
            image=image,
            timestamp=tum_ts - self._t0,  # monotonic offset, preserves true dt
            frame_idx=self._frame_idx,
            intrinsics=self._intrinsics,
            source_id=f"tum:{self._dir.name}",
        )
        self._frame_idx += 1
        return frame

    def release(self) -> None:
        self._is_open = False

    # -- ground-truth accessors (evaluation only — not pipeline inputs) -----

    def depth_gt(self) -> Optional[np.ndarray]:
        """
        Metric depth map (float64, metres) time-associated with the frame
        last returned by get_frame(). Invalid/missing readings are NaN.
        Returns None when no depth frame matched within assoc_max_diff.
        """
        if self._cur_depth_path is None:
            return None
        raw = cv2.imread(str(self._cur_depth_path), cv2.IMREAD_UNCHANGED)
        if raw is None:
            return None
        depth = raw.astype(np.float64) / self._depth_factor
        depth[raw == 0] = np.nan  # 0 == no return in the TUM convention
        return depth

    def pose_gt(self) -> Optional[np.ndarray]:
        """
        4x4 world-from-camera ground-truth pose for the frame last returned
        by get_frame(), or None when no pose matched within assoc_max_diff.
        """
        return self._cur_pose_gt

    @property
    def current_timestamp(self) -> Optional[float]:
        """Absolute TUM timestamp of the last returned frame (None before any)."""
        return self._cur_tum_ts

    @property
    def intrinsics(self) -> Optional[CameraIntrinsics]:
        return self._intrinsics

    @property
    def total_frames(self) -> int:
        return len(self._rgb)
