"""
PoseEstimator + IMU factory.

Single source of truth for constructing the perception stack's pose
backend, used by both `launch.py` (standalone Python pipeline) and the
ROS2 `pose_node` adapter. Extracted so that adding a new visual
backend, IMU backend, or fusion strategy is a one-file change and
both entry points pick it up automatically.

Public API
----------
build_imu(cfg)                          → IMUInterface
build_visual_pose_estimator(cfg, raw)   → PoseEstimator (pre-fusion)
build_pose_estimator(cfg, raw)          → PoseEstimator (post-fusion;
                                          wraps in VIO when enabled)

Why three functions instead of one
----------------------------------
- build_imu is used standalone for tests + diagnostics.
- build_visual_pose_estimator is used by tests that want the bare
  visual backend without the EKF layer.
- build_pose_estimator is the one consumers should typically call —
  it returns the fully-configured estimator, which is a
  VIOPoseEstimator when `cfg.vio.enabled` and the visual backend
  otherwise.
"""

from __future__ import annotations

import logging

from perception.imu_interface import IMUInterface, NullIMU, SyntheticIMU
from perception.pose_estimator import NullPoseEstimator, PoseEstimator
from perception.vio_pose_estimator import VIOPoseEstimator
from state_estimation.visual_inertial_ekf import VIOConfig as EKFConfig

log = logging.getLogger(__name__)


def build_imu(cfg) -> IMUInterface:
    """Construct an IMUInterface from a typed IMUConfig (cfg.imu)."""
    ic = cfg.imu
    if ic.type == "null":
        return NullIMU()
    if ic.type == "synthetic":
        imu = SyntheticIMU(
            motion=ic.synthetic_motion,
            rate_hz=ic.rate_hz,
            noise_std_accel=ic.synthetic_noise_std_accel,
            noise_std_gyro=ic.synthetic_noise_std_gyro,
            seed=ic.synthetic_seed,
        )
        imu.open()
        return imu
    raise ValueError(
        f"Unknown imu.type={ic.type!r}. Supported: 'null', 'synthetic'."
    )


def build_visual_pose_estimator(cfg, raw: dict) -> PoseEstimator:
    """The pre-fusion visual backend keyed on cfg.pose_estimator.type."""
    pe_type = cfg.pose_estimator.type
    if pe_type == "null":
        return NullPoseEstimator()
    if pe_type == "dpvo":
        # Imported lazily — DPVO is a heavy CUDA-extension dep that
        # most callers (config validators, tests, the ROS2 dry-run
        # path) don't need to drag in unless they're actually
        # selecting it.
        from perception.dpvo_pose_estimator import DPVOPoseEstimator
        est = DPVOPoseEstimator(raw)
        log.info("Visual PoseEstimator: %r", est)
        return est
    if pe_type == "dpv_slam":
        # DPV-SLAM = DPVO + loop closure. Same heavy dep, lazy-imported.
        from perception.dpv_slam_pose_estimator import DPVSLAMPoseEstimator
        est = DPVSLAMPoseEstimator(raw)
        log.info("Visual PoseEstimator: %r", est)
        return est
    raise ValueError(
        f"Unknown pose_estimator.type={pe_type!r}. "
        "Supported: 'null', 'dpvo', 'dpv_slam'."
    )


def build_pose_estimator(cfg, raw: dict) -> PoseEstimator:
    """
    Fully-configured PoseEstimator. When `cfg.vio.enabled=true`, the
    visual estimator is wrapped in a VIOPoseEstimator that fuses it
    with the configured IMU via the loosely-coupled error-state EKF.
    Downstream consumers see the fused pose without changes — the ABC
    is the same.
    """
    visual = build_visual_pose_estimator(cfg, raw)
    if not cfg.vio.enabled:
        return visual

    imu = build_imu(cfg)
    v = cfg.vio
    ekf_cfg = EKFConfig(
        init_position_std_m=      v.init_position_std_m,
        init_velocity_std_mps=    v.init_velocity_std_mps,
        init_orientation_std_rad= v.init_orientation_std_rad,
        init_bias_gyro_std=       v.init_bias_gyro_std,
        init_bias_accel_std=      v.init_bias_accel_std,
        bias_gyro_random_walk=    v.bias_gyro_random_walk,
        bias_accel_random_walk=   v.bias_accel_random_walk,
        visual_position_std_m=    v.visual_position_std_m,
        visual_orientation_std_rad= v.visual_orientation_std_rad,
        gravity_w=                tuple(v.gravity_w),
    )
    log.info(
        "Wrapping %s in VIOPoseEstimator with IMU %r",
        type(visual).__name__, type(imu).__name__,
    )
    return VIOPoseEstimator(
        visual_estimator=visual, imu=imu, ekf_cfg=ekf_cfg,
    )
