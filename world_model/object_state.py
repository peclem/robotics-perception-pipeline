"""
ObjectState — probabilistic state of a single tracked object.

Stores the current KF/EKF estimate plus a bounded history buffer
of KFSnapshot objects for trajectory analysis.

Robotics context
----------------
This is the per-object node in the scene graph. In a full SLAM system
this would be replaced by a landmark in the factor graph. Here it is
a lightweight probabilistic state container that any downstream module
(planner, occupancy grid, visualizer) can query.

Upgrade path
------------
Step 13: feed ObjectState.position + ObjectState.covariance into
         the MOT17 benchmark evaluator.
Phase 3: replace history buffer with a Gaussian Process trajectory
         model for smooth long-horizon prediction.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional

import numpy as np

from state_estimation.kalman_filter import KFSnapshot
from world_model.stability import StabilityClass, stability_for_class
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from perception.pose_estimator import CameraPose


@dataclass
class ObjectState:
    """
    Probabilistic state of a single tracked object in the world model.

    Parameters
    ----------
    track_id    : unique track ID from ByteTracker
    class_id    : COCO class integer
    class_name  : human-readable class label
    position    : (2,) float64 array — estimated centre [cx, cy] in pixels
    covariance  : (N, N) float64 — KF/EKF covariance matrix (8×8 or 9×9)
    velocity    : (4,) float64 — [vx, vy, vw, vh] in pixels/s
    score       : detector confidence of last matched detection
    last_seen   : monotonic timestamp of last update
    n_updates   : total KF update count
    is_lost     : True when track is in LOST state (predicting, not updated)
    max_history : maximum KFSnapshot entries to keep
    history     : bounded deque of KFSnapshot — one per KF update
    """
    track_id:   int
    class_id:   int
    class_name: str
    position:   np.ndarray          # (2,) [cx, cy]
    covariance: np.ndarray          # (N, N)
    velocity:   np.ndarray          # (4,) [vx, vy, vw, vh]
    score:      float
    last_seen:  float               # time.monotonic()
    n_updates:  int
    is_lost:    bool                = False
    position_3d: Optional[np.ndarray] = None   # (3,) [X, Y, Z] in metres, camera frame; None if unavailable
    position_world: Optional[np.ndarray] = None  # (3,) [X, Y, Z] in metres, world (map) frame; None if no ego-pose
    stability:  StabilityClass      = StabilityClass.SEMI_STATIC
    # Frame counters used by SceneGraph's motion-based stability override.
    # Reset whenever the relevant condition lapses.
    _moving_frames:     int         = 0
    _stationary_frames: int         = 0
    max_history: int                = 30
    history:    Deque[KFSnapshot]   = field(
        default_factory=lambda: deque(maxlen=30)
    )

    def __post_init__(self):
        self.position   = np.asarray(self.position,   dtype=np.float64)
        self.velocity   = np.asarray(self.velocity,   dtype=np.float64)
        self.covariance = np.asarray(self.covariance, dtype=np.float64)
        # Reinitialise history with correct maxlen
        if not isinstance(self.history, deque) or \
                self.history.maxlen != self.max_history:
            existing = list(self.history)
            self.history = deque(existing, maxlen=self.max_history)
        if self.position_3d is not None:
            self.position_3d = np.asarray(self.position_3d, dtype=np.float64)
        if self.position_world is not None:
            self.position_world = np.asarray(self.position_world, dtype=np.float64)
    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------

    @property
    def position_std(self) -> np.ndarray:
        """
        Standard deviation of position estimate [std_cx, std_cy].
        Extracted from the diagonal of the 2×2 position submatrix of P.
        Small std = well-localised. Large std = uncertain (occluded or new).
        """
        return np.sqrt(np.maximum(np.diag(self.covariance)[:2], 0.0))

    @property
    def position_uncertainty_area(self) -> float:
        """
        Area of the 2σ covariance ellipse (pixels²).
        Proportional to det(P[:2,:2]). Used to rank objects by certainty.
        """
        P2 = self.covariance[:2, :2]
        det = P2[0, 0] * P2[1, 1] - P2[0, 1] * P2[1, 0]
        return float(max(det, 0.0)) * (2.0 ** 2) * np.pi

    @property
    def speed(self) -> float:
        """Estimated speed in pixels/second: ||[vx, vy]||."""
        return float(np.linalg.norm(self.velocity[:2]))

    @property
    def trajectory(self) -> np.ndarray:
        """
        (N, 2) array of historical centre positions from the history buffer.
        First row = oldest, last row = most recent.
        Returns empty (0, 2) array if history is empty.
        """
        if not self.history:
            return np.zeros((0, 2), dtype=np.float64)
        return np.array(
            [snap.state[:2] for snap in self.history],
            dtype=np.float64,
        )

    @property
    def has_world_position(self) -> bool:
        """True if metric world-frame position is available."""
        return self.position_world is not None

    @property
    def has_metric_depth(self) -> bool:
        """True if metric 3D position is available (camera calibrated + depth estimated)."""
        return self.position_3d is not None

    @property
    def depth_m(self) -> float:
        """Metric depth in metres. 0.0 if unavailable."""
        if self.position_3d is None:
            return 0.0
        return float(self.position_3d[2])

    def add_snapshot(self, snap: KFSnapshot) -> None:
        """Append a KFSnapshot to the bounded history buffer."""
        self.history.append(snap)

    def distance_to(self, point: np.ndarray) -> float:
        """
        Euclidean distance from object centre to a query point.

        Parameters
        ----------
        point : (2,) array [x, y] in the same pixel coordinate frame.
        """
        return float(np.linalg.norm(self.position - np.asarray(point)))

    def mahalanobis_to(self, point: np.ndarray) -> float:
        """
        Mahalanobis distance from object centre to a query point,
        using the 2×2 position covariance submatrix.

        This is the uncertainty-aware distance — an object with a large
        covariance ellipse is "closer" in Mahalanobis terms than a
        well-localised object at the same Euclidean distance.

        Returns Euclidean distance if covariance is singular.
        """
        p    = np.asarray(point, dtype=np.float64)
        diff = self.position - p
        P2   = self.covariance[:2, :2]
        try:
            return float(diff @ np.linalg.solve(P2, diff))
        except np.linalg.LinAlgError:
            return self.distance_to(point)

    def __repr__(self) -> str:
        cx, cy = self.position
        return (
            f"ObjectState(id={self.track_id} '{self.class_name}' "
            f"cx={cx:.0f} cy={cy:.0f} "
            f"std=[{self.position_std[0]:.1f},{self.position_std[1]:.1f}] "
            f"lost={self.is_lost} updates={self.n_updates})"
        )
