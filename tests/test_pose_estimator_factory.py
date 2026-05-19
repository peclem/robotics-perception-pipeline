"""
Unit tests for perception.pose_estimator_factory.

The factory backs both launch.py (standalone Python) and the ROS2
pose_node, so the contract verified here is the one both consumers
depend on. DPVO is excluded — needs CUDA + the third-party build —
and is covered by integration tests separately.
"""

from __future__ import annotations

import pytest

from perception.config_loader import (
    IMUConfig, PipelineConfig, VIOConfig as VIOCfgLoader,
)
from perception.imu_interface import NullIMU, SyntheticIMU
from perception.pose_estimator import NullPoseEstimator
from perception.pose_estimator_factory import (
    build_imu, build_pose_estimator, build_visual_pose_estimator,
)
from perception.vio_pose_estimator import VIOPoseEstimator


def _cfg(vio_enabled: bool = False, imu_type: str = "null",
         pe_type: str = "null") -> PipelineConfig:
    cfg = PipelineConfig()
    cfg.imu = IMUConfig(type=imu_type)
    cfg.vio = VIOCfgLoader(enabled=vio_enabled)
    cfg.pose_estimator.type = pe_type
    return cfg


class TestBuildIMU:

    def test_null(self):
        cfg = _cfg(imu_type="null")
        imu = build_imu(cfg)
        assert isinstance(imu, NullIMU)

    def test_synthetic(self):
        cfg = _cfg(imu_type="synthetic")
        imu = build_imu(cfg)
        assert isinstance(imu, SyntheticIMU)

    def test_unknown_type_raises(self):
        cfg = _cfg()
        cfg.imu = IMUConfig(type="not-a-backend")
        with pytest.raises(ValueError, match="Unknown imu.type"):
            build_imu(cfg)


class TestBuildVisualPoseEstimator:

    def test_null(self):
        cfg = _cfg(pe_type="null")
        est = build_visual_pose_estimator(cfg, raw=cfg.as_dict())
        assert isinstance(est, NullPoseEstimator)

    def test_unknown_type_raises(self):
        cfg = _cfg(pe_type="not-a-backend")
        with pytest.raises(ValueError, match="Unknown pose_estimator.type"):
            build_visual_pose_estimator(cfg, raw=cfg.as_dict())


class TestBuildPoseEstimator:

    def test_vio_disabled_returns_visual_directly(self):
        cfg = _cfg(vio_enabled=False, pe_type="null")
        est = build_pose_estimator(cfg, raw=cfg.as_dict())
        assert isinstance(est, NullPoseEstimator)

    def test_vio_enabled_wraps_visual_with_null_imu(self):
        cfg = _cfg(vio_enabled=True, imu_type="null", pe_type="null")
        est = build_pose_estimator(cfg, raw=cfg.as_dict())
        assert isinstance(est, VIOPoseEstimator)
        assert isinstance(est.visual, NullPoseEstimator)
        assert isinstance(est.imu, NullIMU)

    def test_vio_enabled_wraps_with_synthetic_imu(self):
        cfg = _cfg(vio_enabled=True, imu_type="synthetic", pe_type="null")
        est = build_pose_estimator(cfg, raw=cfg.as_dict())
        assert isinstance(est, VIOPoseEstimator)
        assert isinstance(est.imu, SyntheticIMU)

    def test_vio_ekf_config_carries_through(self):
        # EKFConfig fields populated from cfg.vio — sanity-check that
        # at least one non-default value round-trips correctly.
        cfg = _cfg(vio_enabled=True, imu_type="null", pe_type="null")
        cfg.vio.init_position_std_m = 0.42
        est = build_pose_estimator(cfg, raw=cfg.as_dict())
        # Underlying EKF picks up the init std → covariance block is
        # 3 * (0.42**2) ≈ 0.5292.
        assert isinstance(est, VIOPoseEstimator)
        import numpy as np
        np.testing.assert_allclose(
            np.trace(est.ekf.covariance[0:3, 0:3]),
            3 * (0.42 ** 2),
            atol=1e-9,
        )
