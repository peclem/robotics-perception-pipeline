"""
VIOPoseEstimator — live orchestration of the visual-inertial EKF.

Wraps an inner `PoseEstimator` (visual: DPVO, ORB-SLAM3-with-IMU, etc.)
and an `IMUInterface`, and threads them through the `VisualInertialEKF`
fuser shipped in state_estimation/visual_inertial_ekf.py.

Why this lives behind the PoseEstimator ABC
-------------------------------------------
The whole downstream stack (TransformTree, SceneGraph, ROS2 pose
publishers, occupancy builders) already consumes `PoseEstimator`.
Wrapping VIO in the same interface means swapping the visual-only
backend for the fused backend is a one-line config change, with zero
downstream churn. Same trick we used for the depth backends.

Per-frame flow
--------------
On each `estimate(frame)`:
  1. Pull IMU samples since the previous frame's timestamp.
  2. Pre-integrate them via `IMUPreintegrator.integrate(...)`.
  3. EKF `predict()` consumes the pre-integration.
  4. Call the inner visual estimator. If it returns a pose, EKF
     `update()` consumes it as a 6-DOF measurement.
  5. Return the EKF's current pose, tagged source='vio'.

What this v1 does NOT do
------------------------
- **No producer thread**: IMU samples are pulled synchronously on the
  visual-frame callback. With SyntheticIMU (deterministic catch-up)
  or a hardware backend that buffers, this is fine; with a true
  push-source IMU the user would add a background queue and the
  wrapper would consume from it instead.
- **No initialisation from rest**: caller can pass an initial state;
  default is identity at the origin (good enough for relative-motion
  tasks; a real deployment would solve gravity + initial velocity
  from a few seconds of stationary IMU).
- **No outlier rejection** on the visual update — easy to add as a
  chi-square gate once real data shows the failure mode.
- **No camera-IMU extrinsic**: assumed identity (camera == body) in
  v1. A non-trivial extrinsic should transform the visual pose
  before update, via the existing TransformTree.

References
----------
Sola (2017), Forster et al. (2017) — see visual_inertial_ekf.py for
the full math context.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from perception.camera_interface import CameraFrame
from perception.imu_interface import IMUInterface, NullIMU
from perception.pose_estimator import (
    CameraPose, NullPoseEstimator, PoseEstimator,
)
from state_estimation.imu_preintegration import IMUPreintegrator
from state_estimation.visual_inertial_ekf import (
    VIOConfig, VIONominalState, VisualInertialEKF,
)

log = logging.getLogger(__name__)


class VIOPoseEstimator(PoseEstimator):
    """
    Loosely-coupled VIO orchestration behind the PoseEstimator ABC.

    Parameters
    ----------
    visual_estimator
        Inner PoseEstimator producing per-frame visual poses (DPVO,
        ORB-SLAM3, etc.). May be NullPoseEstimator — in that case the
        EKF degenerates to IMU-only dead-reckoning and the output
        drifts. Not a useful configuration in practice but the
        wrapper handles it for completeness.
    imu
        IMUInterface backend. NullIMU disables the predict step; the
        EKF then propagates with empty pre-integration windows
        (covariance grows at the bias-random-walk rate but the
        nominal state stays put).
    ekf_cfg
        VIOConfig for the underlying VisualInertialEKF.
    initial_state
        Optional `VIONominalState` — caller's starting estimate.
        Default = identity at origin, zero velocity, zero biases.
    """

    def __init__(
        self,
        visual_estimator: PoseEstimator,
        imu:              IMUInterface,
        ekf_cfg:          Optional[VIOConfig] = None,
        initial_state:    Optional[VIONominalState] = None,
    ) -> None:
        self._visual = visual_estimator
        self._imu = imu
        self._preint = IMUPreintegrator(
            sigma_gyro_n=getattr(imu, "sigma_gyro_n",  1.7e-4),
            sigma_accel_n=getattr(imu, "sigma_accel_n", 2.0e-3),
        )
        self._ekf = VisualInertialEKF(
            ekf_cfg or VIOConfig(),
            initial_state=initial_state,
        )
        self._last_visual_ts: Optional[float] = None
        # True once any predict or update has incorporated information
        # into the EKF state. Before that, the EKF holds only its
        # initial guess and returning it would mislead consumers into
        # thinking VIO has a pose lock when it doesn't.
        self._has_information: bool = False

    # ------------------------------------------------------------------
    # PoseEstimator ABC
    # ------------------------------------------------------------------

    def estimate(self, frame: CameraFrame) -> Optional[CameraPose]:
        """
        Run one VIO step. Returns the fused pose, or None when neither
        the IMU nor the visual estimator has produced enough information
        for a meaningful estimate yet.
        """
        # 1. Pull IMU samples since the previous visual frame.
        if self._last_visual_ts is not None:
            samples = self._imu.get_samples_since(self._last_visual_ts)
            if len(samples) >= 2:
                # 2. Pre-integrate.
                preint = self._preint.integrate(
                    samples,
                    b_g=self._ekf.state.b_g,
                    b_a=self._ekf.state.b_a,
                )
                if preint.dt > 0.0:
                    # 3. EKF predict.
                    self._ekf.predict(preint)
                    self._has_information = True

        # 4. Visual measurement.
        visual_pose = self._visual.estimate(frame)
        if visual_pose is not None:
            self._ekf.update(visual_pose)
            self._has_information = True

        self._last_visual_ts = frame.timestamp

        # 5. Surface the fused state once we've actually incorporated
        # information. Before then the EKF only holds its initial guess
        # — returning it would falsely imply VIO has a pose lock.
        if not self._has_information:
            return None

        out = self._ekf.current_pose(frame_idx=frame.frame_idx)
        out.timestamp = frame.timestamp
        out.source = "vio"
        return out

    def reset(self) -> None:
        """Reset inner visual estimator + EKF state; IMU cursor preserved."""
        self._visual.reset()
        cfg = self._ekf._cfg   # reuse the config we were built with
        self._ekf = VisualInertialEKF(cfg)
        self._last_visual_ts = None
        self._has_information = False

    @property
    def is_initialised(self) -> bool:
        return self._has_information

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def ekf(self) -> VisualInertialEKF:
        """Direct access to the inner EKF (state + covariance, biases)."""
        return self._ekf

    @property
    def visual(self) -> PoseEstimator:
        return self._visual

    @property
    def imu(self) -> IMUInterface:
        return self._imu

    def __repr__(self) -> str:
        return (
            f"VIOPoseEstimator(visual={type(self._visual).__name__}, "
            f"imu={type(self._imu).__name__}, "
            f"init={self.is_initialised})"
        )
