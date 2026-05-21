"""
CODa IMU replay as an IMUInterface backend.

Feeds a CODa sequence's VectorNav inertial stream to the VIO fuser.
CODa records raw IMU at ~20 Hz in poses/imu/<SEQ>.txt, one line:

    ts  ax ay az  gx gy gz  qx qy qz qw

— timestamp, body-frame acceleration (m/s², gravity included), body-
frame angular velocity (rad/s), and the sensor's onboard AHRS
quaternion (unused here; pre-integration needs only accel + gyro).

Replay vs live timing
---------------------
SyntheticIMU and hardware backends are live sources: get_samples_since()
caps at wall-clock now. A dataset replay has no wall clock — frames are
processed as fast as the CPU allows — so this backend is paced by the
*dataset* clock instead. It loads the camera frame grid (the
poses/dense_global timestamps, which CODaDatasetCamera also replays)
and get_samples_since(t) returns exactly the IMU samples in the window
(t, t_next_frame]. Both this backend and CODaDatasetCamera derive the
same t0 from the same dense_global file, so their monotonic-offset
timelines coincide — the VIO handshake stays synchronised.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import List, Tuple

import numpy as np

from perception.imu_interface import IMUInterface, IMUSample
from perception.coda_dataset_camera import read_coda_trajectory


def _read_coda_imu(path: str | Path) -> List[Tuple[float, np.ndarray, np.ndarray]]:
    """Parse a CODa poses/imu/<SEQ>.txt into (ts, accel, gyro) rows."""
    rows: List[Tuple[float, np.ndarray, np.ndarray]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 7:
                continue
            ts = float(parts[0])
            accel = np.array([float(v) for v in parts[1:4]], dtype=np.float64)
            gyro = np.array([float(v) for v in parts[4:7]], dtype=np.float64)
            rows.append((ts, accel, gyro))
    rows.sort(key=lambda r: r[0])
    return rows


class CODaIMU(IMUInterface):
    """
    Replay a CODa sequence's IMU stream behind the IMUInterface ABC.

    Parameters
    ----------
    sequence_dir  : path to an extracted CODa sequence directory
                    (the same one CODaDatasetCamera replays).
    sigma_gyro_n  : gyro noise density (rad/s/√Hz) — read by the VIO
    sigma_accel_n : accel noise density (m/s²/√Hz)   pre-integrator.
    """

    def __init__(
        self,
        sequence_dir: str | Path,
        *,
        sigma_gyro_n: float = 1.7e-4,
        sigma_accel_n: float = 2.0e-3,
    ) -> None:
        self._dir = Path(sequence_dir)
        self.sigma_gyro_n = float(sigma_gyro_n)
        self.sigma_accel_n = float(sigma_accel_n)

        self._is_open = False
        self._samples: List[IMUSample] = []
        self._sample_ts = np.empty(0, dtype=np.float64)
        self._frame_ts = np.empty(0, dtype=np.float64)
        self._rate_hz = 0.0

    # -- lifecycle ----------------------------------------------------------

    def open(self) -> None:
        if not self._dir.is_dir():
            raise FileNotFoundError(f"CODa sequence directory not found: {self._dir}")

        imu_dir = self._dir / "poses" / "imu"
        if not imu_dir.is_dir():
            raise FileNotFoundError(
                f"{imu_dir} missing — sequence has no IMU stream."
            )
        imu_txts = sorted(imu_dir.glob("*.txt"))
        if not imu_txts:
            raise FileNotFoundError(f"No IMU file under {imu_dir}.")
        seq_id = imu_txts[0].stem

        # Camera frame grid + shared t0 from dense_global (one-to-one with
        # the frames CODaDatasetCamera replays).
        pose_txt = self._dir / "poses" / "dense_global" / f"{seq_id}.txt"
        if not pose_txt.exists():
            raise FileNotFoundError(
                f"{pose_txt} missing — cannot align the IMU to the frame grid."
            )
        traj = read_coda_trajectory(pose_txt)
        if not traj:
            raise RuntimeError(f"{pose_txt} contains no poses.")
        t0 = traj[0][0]
        self._frame_ts = np.array([ts - t0 for ts, _ in traj], dtype=np.float64)

        rows = _read_coda_imu(imu_txts[0])
        if not rows:
            warnings.warn(f"{imu_txts[0]} contains no IMU samples.", stacklevel=2)
        self._samples = [
            IMUSample(accel=a, gyro=g, timestamp=ts - t0) for ts, a, g in rows
        ]
        self._sample_ts = np.array([s.timestamp for s in self._samples],
                                   dtype=np.float64)
        if self._sample_ts.size > 1:
            span = self._sample_ts[-1] - self._sample_ts[0]
            self._rate_hz = (self._sample_ts.size - 1) / span if span > 0 else 0.0

        self._is_open = True

    def release(self) -> None:
        self._is_open = False

    # -- IMUInterface -------------------------------------------------------

    def get_samples_since(self, t: float) -> List[IMUSample]:
        """
        IMU samples in the inter-frame window after `t`.

        `t` is the caller's previous visual-frame timestamp; the window
        runs from that frame to the next one on the camera grid. `t` is
        snapped to the nearest grid frame first — the caller passes a
        frame timestamp, but a YAML/float round-trip can leave it a hair
        off-grid, and an un-snapped bound collapses the window. Returns
        [] before open() or once the sequence is exhausted.
        """
        if not self._is_open or self._sample_ts.size == 0:
            return []
        n = self._frame_ts.size
        # Snap t to the nearest camera frame.
        pos = int(np.searchsorted(self._frame_ts, t))
        cands = [c for c in (pos - 1, pos) if 0 <= c < n]
        k = min(cands, key=lambda c: abs(self._frame_ts[c] - t)) if cands else 0
        lo_t = self._frame_ts[k]
        hi_t = self._frame_ts[k + 1] if k + 1 < n else np.inf

        lo = int(np.searchsorted(self._sample_ts, lo_t, side="right"))
        hi = int(np.searchsorted(self._sample_ts, hi_t, side="right"))
        return self._samples[lo:hi]

    @property
    def rate_hz(self) -> float:
        return self._rate_hz

    @property
    def total_samples(self) -> int:
        return len(self._samples)

    def __repr__(self) -> str:
        return (
            f"CODaIMU(dir={self._dir.name}, samples={len(self._samples)}, "
            f"rate={self._rate_hz:.1f}Hz)"
        )
