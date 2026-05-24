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
  3b. If the IMU window is at rest, a zero-velocity update (ZUPT)
     pins the body velocity to ~0 — bounds drift through stops.
  4. Call the inner visual estimator. If it returns a pose, it is
     re-expressed in the IMU body frame and EKF `update()` consumes
     it as a 6-DOF measurement.
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
from state_estimation.imu_init import is_stationary
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
        cam_imu_extrinsic: Optional[tuple] = None,
    ) -> None:
        self._visual = visual_estimator
        self._imu = imu

        # Camera→IMU extrinsic (R_imu_cam, p_imu_cam): x_imu = R·x_cam + p.
        # None / identity ⇒ camera == IMU body (the v1 assumption).
        if cam_imu_extrinsic is not None:
            R_ic, p_ic = cam_imu_extrinsic
            self._R_imu_cam = np.asarray(R_ic, dtype=np.float64).reshape(3, 3)
            self._p_imu_cam = np.asarray(p_ic, dtype=np.float64).reshape(3)
            self._has_extrinsic = True
        else:
            self._R_imu_cam = np.eye(3, dtype=np.float64)
            self._p_imu_cam = np.zeros(3, dtype=np.float64)
            self._has_extrinsic = False
        self._preint = IMUPreintegrator(
            sigma_gyro_n=getattr(imu, "sigma_gyro_n",  1.7e-4),
            sigma_accel_n=getattr(imu, "sigma_accel_n", 2.0e-3),
        )
        self._initial_state = initial_state
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

                # 3b. Zero-velocity update when the IMU window is at
                # rest — bounds velocity + accel-bias drift through stops.
                if is_stationary(samples):
                    self._ekf.update_zero_velocity()
                    self._has_information = True

        # 4. Visual measurement — transformed from the camera frame into
        # the IMU body frame the EKF tracks (no-op if extrinsic is identity).
        visual_pose = self._visual.estimate(frame)
        if visual_pose is not None:
            self._ekf.update(self._to_body_frame(visual_pose))
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

    def _to_body_frame(self, visual_pose: CameraPose) -> CameraPose:
        """
        Re-express a camera-frame visual pose as the IMU-body pose the
        EKF observes. The visual estimator tracks the camera; the EKF
        tracks the IMU; they are rigidly offset by the extrinsic.

        With R_imu_cam, p_imu_cam (x_imu = R·x_cam + p):
            R_w_i  = R_w_c · R_imu_camᵀ
            p_w_i  = p_w_c − R_w_i · p_imu_cam
        The visual position is in monocular (unscaled) units, while the
        lever arm p_imu_cam is metric — so it is divided by the EKF's
        current scale estimate to land in the same units as the
        measurement. Identity extrinsic ⇒ returns the pose unchanged.
        """
        if not self._has_extrinsic:
            return visual_pose
        R_w_c = np.asarray(visual_pose.R, dtype=np.float64).reshape(3, 3)
        p_w_c = np.asarray(visual_pose.t, dtype=np.float64).reshape(3)
        R_w_i = R_w_c @ self._R_imu_cam.T
        scale = self._ekf.state.scale
        p_w_i = p_w_c - (R_w_i @ self._p_imu_cam) / scale
        return CameraPose(
            R=R_w_i, t=p_w_i,
            timestamp=visual_pose.timestamp, frame_idx=visual_pose.frame_idx,
            confidence=visual_pose.confidence, source=visual_pose.source,
        )

    def reset(self) -> None:
        """Reset inner visual estimator + EKF state; IMU cursor preserved."""
        self._visual.reset()
        cfg = self._ekf._cfg   # reuse the config we were built with
        self._ekf = VisualInertialEKF(cfg, initial_state=self._initial_state)
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
