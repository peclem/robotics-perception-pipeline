"""
Camera ego-pose estimation for the robotics perception pipeline.

Provides the camera's SE(3) pose in the world frame, used to transform
object positions from camera frame to world frame in the SceneGraph.

Architecture
------------
PoseEstimator (ABC)
    └── NullPoseEstimator   — returns None pose (camera-frame only mode)

Integration points
------------------
1. SceneGraph.update(pose=estimator.estimate(frame))
       ObjectState.position_world set when pose available.

2. Future backends — drop-in replacements, zero other changes:
       ORBSLAMPoseEstimator    — ORB-SLAM3 via socket/ROS2 topic
       VINSMonoPoseEstimator   — VINS-Mono monocular VIO
       ROS2PoseEstimator       — subscribes to /odom or /tf

Coordinate convention
---------------------
World frame: right-handed, Z-up (ROS2 REP-103 convention)
Camera frame: right-handed, Z-forward (OpenCV convention)

CameraPose.R : (3,3) rotation matrix  — world ← camera
CameraPose.t : (3,)  translation      — camera origin in world frame

Transform from camera to world:
    p_world = R @ p_camera + t

Reference
---------
Campos et al. — ORB-SLAM3 (2021). arXiv:2007.11898
Qin et al.    — VINS-Mono (2018). arXiv:1708.03852
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np

from perception.camera_interface import CameraFrame

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CameraPose
# ---------------------------------------------------------------------------

@dataclass
class CameraPose:
    """
    SE(3) camera pose in the world frame.

    Attributes
    ----------
    R           : (3, 3) rotation matrix — world ← camera
    t           : (3,)   translation — camera origin in world frame (metres)
    timestamp   : monotonic timestamp of this pose estimate
    frame_idx   : frame index this pose corresponds to
    confidence  : pose confidence in [0, 1]. 1.0 = fully tracked,
                  lower values indicate initialisation or lost tracking.
                  Used by SceneGraph to decide whether to update position_world.
    source      : string identifier of the estimator that produced this pose
    """
    R:          np.ndarray       # (3, 3) float64
    t:          np.ndarray       # (3,)   float64
    timestamp:  float
    frame_idx:  int
    confidence: float = 1.0
    source:     str   = "unknown"

    def __post_init__(self):
        self.R = np.asarray(self.R, dtype=np.float64)
        self.t = np.asarray(self.t, dtype=np.float64)
        assert self.R.shape == (3, 3), f"R must be (3,3), got {self.R.shape}"
        assert self.t.shape == (3,),   f"t must be (3,),  got {self.t.shape}"

    def transform_point(self, p_camera: np.ndarray) -> np.ndarray:
        """
        Transform a point from camera frame to world frame.

        Parameters
        ----------
        p_camera : (3,) point in camera frame [X, Y, Z] metres

        Returns
        -------
        (3,) point in world frame [X, Y, Z] metres
        """
        p = np.asarray(p_camera, dtype=np.float64)
        return self.R @ p + self.t

    def transform_points(self, pts_camera: np.ndarray) -> np.ndarray:
        """
        Transform (N, 3) points from camera frame to world frame.

        Parameters
        ----------
        pts_camera : (N, 3) points in camera frame

        Returns
        -------
        (N, 3) points in world frame
        """
        pts = np.asarray(pts_camera, dtype=np.float64)
        return (self.R @ pts.T).T + self.t

    @classmethod
    def identity(cls, timestamp: float = 0.0, frame_idx: int = 0) -> "CameraPose":
        """Identity pose — camera at world origin, no rotation."""
        return cls(
            R=np.eye(3),
            t=np.zeros(3),
            timestamp=timestamp,
            frame_idx=frame_idx,
            confidence=1.0,
            source="identity",
        )

    def __repr__(self) -> str:
        return (
            f"CameraPose(t=[{self.t[0]:.2f},{self.t[1]:.2f},{self.t[2]:.2f}]m "
            f"conf={self.confidence:.2f} src={self.source})"
        )


# ---------------------------------------------------------------------------
# ABC
# ---------------------------------------------------------------------------

class PoseEstimator(ABC):
    """
    Abstract base class for camera ego-pose estimators.

    All implementations return Optional[CameraPose] — None when
    the estimator has not yet initialised or has lost tracking.

    The SceneGraph treats a None pose as camera-frame-only mode:
    ObjectState.position_world is not updated, position (pixels)
    remains the only available coordinate.
    """

    @abstractmethod
    def estimate(self, frame: CameraFrame) -> Optional[CameraPose]:
        """
        Estimate camera pose for the current frame.

        Returns None if pose is unavailable (not yet initialised,
        lost tracking, or estimator disabled).
        """
        ...

    @abstractmethod
    def reset(self) -> None:
        """Reset estimator state between sequences."""
        ...

    @property
    @abstractmethod
    def is_initialised(self) -> bool:
        """True once the estimator has produced its first valid pose."""
        ...


# ---------------------------------------------------------------------------
# Null estimator — camera-frame-only mode
# ---------------------------------------------------------------------------

class NullPoseEstimator(PoseEstimator):
    """
    No-op pose estimator. Always returns None.

    Used when no ego-pose source is available. The pipeline runs
    in camera-frame-only mode — ObjectState.position_world is None,
    ObjectState.position (pixel coordinates) is the only output.

    Replace with a real estimator when ego-pose becomes available:
        ORBSLAMPoseEstimator, StereoVIOPoseEstimator, ROS2PoseEstimator.
    """

    def estimate(self, frame: CameraFrame) -> Optional[CameraPose]:
        return None

    def reset(self) -> None:
        pass

    @property
    def is_initialised(self) -> bool:
        return False

    def __repr__(self) -> str:
        return "NullPoseEstimator(camera-frame-only mode)"
