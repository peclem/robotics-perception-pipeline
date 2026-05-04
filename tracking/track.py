"""
Track — single-object track integrating a Kalman Filter.

State machine
-------------
TENTATIVE  Born from an unmatched detection. Not yet reliable.
           Lives until it accumulates min_hits updates (→ CONFIRMED)
           or misses one frame (→ REMOVED).

CONFIRMED  Reliable track. Visible to downstream consumers (world model).
           Misses one frame → LOST.

LOST       Not detected recently. KF keeps predicting.
           Re-found → CONFIRMED. Exceeds max_age misses → REMOVED.

REMOVED    Dead. Filtered out at end of each tracker update cycle.

Design notes
------------
- Track owns exactly one KalmanFilter instance.
- State transitions are NOT implemented here — the ByteTracker
  owns all transition logic. Track is a pure data + KF container.
- Track IDs are globally unique and monotonically increasing within
  a session. reset_id_counter() must be called between sessions
  (e.g. between test runs) to prevent ID leakage.
- history stores KFSnapshot objects for the world model (Step 11).

Upgrade path
------------
Step 11: world_model/scene_graph.py reads track.history for trajectory
         smoothing and occupancy queries.
Step 10: Replace KalmanFilter with ExtendedKalmanFilter — no other
         change required, the interface is identical.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Optional

import numpy as np

from perception.detector import Detection
from state_estimation.kalman_filter import KalmanFilter, KFSnapshot


class TrackState(IntEnum):
    TENTATIVE = 1
    CONFIRMED = 2
    LOST      = 3
    REMOVED   = 4


class Track:
    """
    Single-object track with an embedded Kalman Filter.

    Attributes
    ----------
    track_id   : globally unique, monotonically increasing integer
    state      : TrackState — managed by ByteTracker, not Track itself
    kf         : KalmanFilter — predict/update driven by ByteTracker
    n_hits     : total successful detection-track associations
    n_misses   : consecutive frames without a matched detection
    age        : total frames since birth (hits + misses)
    score      : confidence of the most recent matched detection
    class_id   : COCO class from the most recent matched detection
    class_name : human-readable class label
    history    : list of KFSnapshot — one per update, for world model
    """

    _id_counter: int = 0

    @classmethod
    def _next_id(cls) -> int:
        cls._id_counter += 1
        return cls._id_counter

    @classmethod
    def reset_id_counter(cls) -> None:
        """
        Reset global ID counter to zero.
        Call between pipeline sessions and between test runs.
        """
        cls._id_counter = 0

    def __init__(self, detection: Detection, config: dict) -> None:
        self.track_id: int       = self._next_id()
        self.state: TrackState   = TrackState.TENTATIVE
        self.kf: KalmanFilter    = KalmanFilter.from_detection(detection, config)
        self.n_hits: int         = 1
        self.n_misses: int       = 0
        self.age: int            = 1
        self.score: float        = detection.confidence
        self.class_id: int       = detection.class_id
        self.class_name: str     = detection.class_name
        self.history: list[KFSnapshot] = []

    # ------------------------------------------------------------------
    # Core operations — called by ByteTracker each frame
    # ------------------------------------------------------------------

    def predict(self, dt: float) -> None:
        """
        Advance the Kalman Filter prediction by dt seconds.
        Called once per frame before association, for every active track.
        """
        self.kf.predict(dt=dt)
        self.age += 1

    def update(self, detection: Detection, timestamp: float) -> None:
        """
        Update the KF with a matched detection.
        State transition (e.g. LOST → CONFIRMED) is handled by ByteTracker.

        Parameters
        ----------
        detection : matched Detection from this frame
        timestamp : monotonic timestamp of the current frame
        """
        self.kf.update(
            observation=detection.bbox_xywh.astype(np.float64),
            confidence=detection.confidence,
        )
        self.n_hits  += 1
        self.n_misses = 0
        self.score    = detection.confidence
        self.class_id = detection.class_id
        self.class_name = detection.class_name

        # Snapshot for world model history
        snap = self.kf.snapshot(timestamp=timestamp, frame_idx=self.age)
        self.history.append(snap)

    # ------------------------------------------------------------------
    # State accessors
    # ------------------------------------------------------------------

    @property
    def bbox_xyxy(self) -> np.ndarray:
        """Current KF estimate as [x1, y1, x2, y2]."""
        return self.kf.bbox_xyxy

    @property
    def bbox_xywh(self) -> np.ndarray:
        """Current KF estimate as [cx, cy, w, h]."""
        return self.kf.bbox_xywh

    @property
    def position(self) -> np.ndarray:
        """Estimated center [cx, cy]."""
        return self.kf.position

    @property
    def velocity(self) -> np.ndarray:
        """Estimated velocity [vx, vy, vw, vh]."""
        return self.kf.velocity

    @property
    def covariance(self) -> np.ndarray:
        """8×8 state covariance matrix."""
        return self.kf.covariance

    @property
    def position_std(self) -> np.ndarray:
        """
        Standard deviation of position estimate [std_cx, std_cy].
        A direct measure of localisation uncertainty — used by the
        world model to set occupancy grid confidence weights.
        """
        return np.sqrt(self.kf.position_variance)

    @property
    def nis(self) -> float:
        """Last NIS value from the KF. NaN before first update."""
        return self.kf.nis()

    @property
    def is_confirmed(self) -> bool:
        return self.state == TrackState.CONFIRMED

    @property
    def is_tentative(self) -> bool:
        return self.state == TrackState.TENTATIVE

    @property
    def is_lost(self) -> bool:
        return self.state == TrackState.LOST

    @property
    def is_removed(self) -> bool:
        return self.state == TrackState.REMOVED

    def __repr__(self) -> str:
        cx, cy = self.position
        return (
            f"Track(id={self.track_id} {self.state.name} "
            f"{self.class_name!r} "
            f"cx={cx:.0f} cy={cy:.0f} "
            f"hits={self.n_hits} misses={self.n_misses})"
        )
