"""
Unit tests for VIOPoseEstimator — the live-orchestration wrapper that
runs the visual estimator + IMU pre-integrator + error-state EKF
behind the PoseEstimator ABC.

TestContract            : satisfies PoseEstimator ABC end-to-end
TestVisualPassthrough   : with NullIMU + visual updates only, fused
                          pose tracks the visual measurements
TestIMUPredictBetween   : IMU predict-step fires between visual frames
TestReset               : resets clear inner visual estimator + EKF
TestFactoryWiring       : launch.Pipeline._build_pose_estimator
                          wraps in VIOPoseEstimator when vio.enabled
"""

from __future__ import annotations

import time

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from perception.camera_interface import CameraFrame, CameraIntrinsics
from perception.imu_interface import NullIMU, SyntheticIMU, IMUSample
from perception.pose_estimator import (
    CameraPose, NullPoseEstimator, PoseEstimator,
)
from perception.vio_pose_estimator import VIOPoseEstimator
from state_estimation.visual_inertial_ekf import VIOConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _frame(t: float, idx: int = 0) -> CameraFrame:
    return CameraFrame(
        image=np.zeros((32, 32, 3), dtype=np.uint8),
        timestamp=t, frame_idx=idx,
        intrinsics=CameraIntrinsics(
            fx=500.0, fy=500.0, cx=16.0, cy=16.0,
            width=32, height=32, dist_coeffs=np.zeros(5),
        ),
        source_id="test",
    )


class _FixedVisual(PoseEstimator):
    """
    Deterministic visual stand-in. Returns the same pose every call
    (or None if `pose` is None). Lets us inject a known measurement
    sequence without bringing up DPVO.
    """
    def __init__(self, pose: CameraPose | None) -> None:
        self._pose = pose
        self._n = 0
    def estimate(self, frame):
        self._n += 1
        return self._pose
    def reset(self) -> None:
        self._n = 0
    @property
    def is_initialised(self) -> bool:
        return self._n > 0


class _SequenceVisual(PoseEstimator):
    """Returns the next CameraPose in a queued sequence per call."""
    def __init__(self, poses: list[CameraPose | None]) -> None:
        self._poses = list(poses)
        self._i = 0
    def estimate(self, frame):
        if self._i >= len(self._poses):
            return None
        p = self._poses[self._i]
        self._i += 1
        return p
    def reset(self) -> None:
        self._i = 0
    @property
    def is_initialised(self) -> bool:
        return self._i > 0


def _zero_gravity_cfg() -> VIOConfig:
    return VIOConfig(gravity_w=(0.0, 0.0, 0.0))


# ---------------------------------------------------------------------------
# PoseEstimator-ABC contract
# ---------------------------------------------------------------------------

class TestContract:

    def test_returns_none_before_first_visual_or_imu(self):
        vio = VIOPoseEstimator(
            visual_estimator=NullPoseEstimator(), imu=NullIMU(),
            ekf_cfg=_zero_gravity_cfg(),
        )
        out = vio.estimate(_frame(t=0.0))
        # Nothing has produced information yet → None.
        assert out is None
        assert not vio.is_initialised

    def test_returns_pose_after_first_visual(self):
        first_pose = CameraPose(
            R=np.eye(3), t=np.array([0.1, 0.0, 0.0]),
            timestamp=0.0, frame_idx=0, source="test",
        )
        vio = VIOPoseEstimator(
            visual_estimator=_FixedVisual(first_pose),
            imu=NullIMU(),
            ekf_cfg=_zero_gravity_cfg(),
        )
        out = vio.estimate(_frame(t=0.0))
        assert out is not None
        assert out.source == "vio"
        assert vio.is_initialised

    def test_source_label_and_timestamp_set(self):
        pose = CameraPose(
            R=np.eye(3), t=np.zeros(3),
            timestamp=0.0, frame_idx=0, source="test",
        )
        vio = VIOPoseEstimator(
            visual_estimator=_FixedVisual(pose), imu=NullIMU(),
            ekf_cfg=_zero_gravity_cfg(),
        )
        out = vio.estimate(_frame(t=7.5, idx=42))
        assert out.timestamp == 7.5
        assert out.frame_idx == 42
        assert out.source == "vio"


# ---------------------------------------------------------------------------
# Visual-only pass-through (NullIMU + tight visual noise)
# ---------------------------------------------------------------------------

class TestVisualPassthrough:

    def test_fused_pose_tracks_tight_visual_measurement(self):
        # Visual says we move steadily along +X. NullIMU → no IMU
        # contribution. EKF should converge on the visual measurement.
        target = np.array([1.0, 0.0, 0.0])
        cfg = VIOConfig(
            gravity_w=(0.0, 0.0, 0.0),
            visual_position_std_m=1e-3,
            visual_orientation_std_rad=1e-3,
        )
        vio = VIOPoseEstimator(
            visual_estimator=_FixedVisual(CameraPose(
                R=np.eye(3), t=target,
                timestamp=0.0, frame_idx=0, source="test",
            )),
            imu=NullIMU(),
            ekf_cfg=cfg,
        )
        # Run several frames so the EKF converges through repeated updates.
        for k in range(10):
            vio.estimate(_frame(t=float(k), idx=k))
        out_t = vio.ekf.state.p_w_i
        # Tight visual noise → state should land very close to the
        # measurement.
        assert np.allclose(out_t, target, atol=0.05), out_t


# ---------------------------------------------------------------------------
# IMU predict step actually runs between visual frames
# ---------------------------------------------------------------------------

class TestIMUPredictBetween:

    def test_synthetic_imu_drives_preintegration(self):
        """
        SyntheticIMU + 'stationary_with_gravity' + zero visual → state
        stays at rest because pre-integration's Δv ≈ R·(0,0,g)·dt
        cancels world-frame gravity at the fusion step.
        """
        imu = SyntheticIMU(motion="stationary_with_gravity",
                           rate_hz=200.0, seed=0)
        imu.open()
        # No visual measurements — pure IMU dead-reckon.
        vio = VIOPoseEstimator(
            visual_estimator=NullPoseEstimator(),
            imu=imu,
            ekf_cfg=VIOConfig(),  # default gravity
        )
        # First frame seeds the timestamp; subsequent frames trigger
        # actual pre-integration over the elapsed window.
        vio.estimate(_frame(t=0.0))
        time.sleep(0.05)   # let synthetic IMU's wall-clock cursor advance
        out = vio.estimate(_frame(t=0.05, idx=1))
        # With default gravity + stationary IMU, the predicted state
        # should hover near the origin (within a few cm thanks to the
        # exact cancellation; numerical noise might add a small offset).
        assert out is not None
        assert np.linalg.norm(out.t) < 0.5

    def test_preint_skipped_on_first_call(self):
        """No prior timestamp → no IMU samples to integrate → no crash."""
        imu = SyntheticIMU(motion="stationary", rate_hz=200.0)
        imu.open()
        vio = VIOPoseEstimator(
            visual_estimator=NullPoseEstimator(), imu=imu,
            ekf_cfg=_zero_gravity_cfg(),
        )
        # Should not raise; returns None because no visual + no IMU history.
        assert vio.estimate(_frame(t=0.0)) is None


# ---------------------------------------------------------------------------
# Reset semantics
# ---------------------------------------------------------------------------

class TestReset:

    def test_reset_clears_initialisation(self):
        pose = CameraPose(
            R=np.eye(3), t=np.zeros(3),
            timestamp=0.0, frame_idx=0, source="test",
        )
        vio = VIOPoseEstimator(
            visual_estimator=_FixedVisual(pose), imu=NullIMU(),
            ekf_cfg=_zero_gravity_cfg(),
        )
        vio.estimate(_frame(t=0.0))
        assert vio.is_initialised
        vio.reset()
        assert not vio.is_initialised
        # And the inner visual estimator was reset too.
        assert not vio.visual.is_initialised


# ---------------------------------------------------------------------------
# launch.Pipeline factory wiring
# ---------------------------------------------------------------------------

class TestFactoryWiring:

    def _pipe(self, vio_enabled: bool, imu_type: str = "null"):
        from perception.config_loader import (
            PipelineConfig, IMUConfig, VIOConfig as VIOCfgLoader,
        )
        from launch import Pipeline
        cfg = PipelineConfig()
        cfg.imu = IMUConfig(type=imu_type)
        cfg.vio = VIOCfgLoader(enabled=vio_enabled)
        return Pipeline(cfg=cfg, source="synthetic", input_path=None)

    def test_vio_disabled_returns_plain_visual(self):
        pipe = self._pipe(vio_enabled=False)
        est = pipe._build_pose_estimator(raw=cfg_dict_for_pipe(pipe))
        assert isinstance(est, NullPoseEstimator)

    def test_vio_enabled_wraps_in_vio(self):
        pipe = self._pipe(vio_enabled=True, imu_type="null")
        est = pipe._build_pose_estimator(raw=cfg_dict_for_pipe(pipe))
        assert isinstance(est, VIOPoseEstimator)
        assert isinstance(est.imu, NullIMU)
        assert isinstance(est.visual, NullPoseEstimator)

    def test_imu_synthetic_factory(self):
        pipe = self._pipe(vio_enabled=True, imu_type="synthetic")
        est = pipe._build_pose_estimator(raw=cfg_dict_for_pipe(pipe))
        assert isinstance(est, VIOPoseEstimator)
        assert isinstance(est.imu, SyntheticIMU)


def cfg_dict_for_pipe(pipe):
    """Helper: the raw dict the factory expects, derived from PipelineConfig."""
    return pipe._cfg.as_dict()


# ---------------------------------------------------------------------------
# Camera→IMU extrinsic transform
# ---------------------------------------------------------------------------

class TestCameraImuExtrinsic:
    """VIOPoseEstimator._to_body_frame — re-express the camera pose as
    the IMU-body pose the EKF tracks, given the rigid extrinsic."""

    @staticmethod
    def _vio(extrinsic):
        return VIOPoseEstimator(
            visual_estimator=NullPoseEstimator(), imu=NullIMU(),
            cam_imu_extrinsic=extrinsic,
        )

    @staticmethod
    def _pose(R_mat, t):
        return CameraPose(R=R_mat, t=np.asarray(t, float),
                          timestamp=0.0, frame_idx=0, source="test")

    def test_no_extrinsic_is_passthrough(self):
        vio = self._vio(None)
        out = vio._to_body_frame(self._pose(np.eye(3), [5.0, 1.0, 2.0]))
        np.testing.assert_allclose(out.t, [5.0, 1.0, 2.0])
        np.testing.assert_allclose(out.R, np.eye(3))

    def test_translation_lever_arm(self):
        # Camera 0.2 m ahead of the IMU in x ⇒ body sits 0.2 m behind.
        vio = self._vio((np.eye(3), np.array([0.2, 0.0, 0.0])))
        out = vio._to_body_frame(self._pose(np.eye(3), [5.0, 0.0, 0.0]))
        np.testing.assert_allclose(out.t, [4.8, 0.0, 0.0], atol=1e-9)

    def test_lever_arm_rotates_with_camera(self):
        # 90° about z: the +x lever arm now points along +y.
        Rz = R.from_rotvec([0, 0, np.pi / 2]).as_matrix()
        vio = self._vio((np.eye(3), np.array([0.2, 0.0, 0.0])))
        out = vio._to_body_frame(self._pose(Rz, [5.0, 0.0, 0.0]))
        np.testing.assert_allclose(out.t, [5.0, -0.2, 0.0], atol=1e-9)

    def test_metric_lever_arm_divided_by_scale(self):
        # The lever arm is metric; the visual position is in scale units.
        vio = self._vio((np.eye(3), np.array([0.2, 0.0, 0.0])))
        vio.ekf.state.scale = 2.0
        out = vio._to_body_frame(self._pose(np.eye(3), [5.0, 0.0, 0.0]))
        np.testing.assert_allclose(out.t, [4.9, 0.0, 0.0], atol=1e-9)

    def test_extrinsic_rotation_composes(self):
        R_ic = R.from_rotvec([0, 0, np.pi / 2]).as_matrix()
        vio = self._vio((R_ic, np.zeros(3)))
        out = vio._to_body_frame(self._pose(np.eye(3), np.zeros(3)))
        np.testing.assert_allclose(out.R, R_ic.T, atol=1e-9)


# ---------------------------------------------------------------------------
# Anchor on the first visual frame
# ---------------------------------------------------------------------------

class TestFirstVisualAnchor:
    """The first visual frame must snap the EKF nominal pose to match
    the (default: untransformed) visual measurement, so the EKF world
    frame aligns with the visual estimator's. Subsequent frames run
    the normal Kalman update."""

    @staticmethod
    def _vio(extrinsic, visual):
        return VIOPoseEstimator(
            visual_estimator=visual, imu=NullIMU(),
            ekf_cfg=_zero_gravity_cfg(),
            cam_imu_extrinsic=extrinsic,
        )

    @staticmethod
    def _pose(R_mat, t, idx=0):
        return CameraPose(R=R_mat, t=np.asarray(t, float),
                          timestamp=0.0, frame_idx=idx, source="test")

    def test_first_visual_snaps_state_to_visual_pose_by_default(self):
        # Default behaviour: body-frame transform off. Even with a big
        # extrinsic the anchor uses the raw visual pose (camera-as-IMU
        # convention — see _apply_body_frame in VIOPoseEstimator).
        R_ic = R.from_rotvec([0, np.pi / 2, 0]).as_matrix()
        first = self._pose(np.eye(3), [0.7, -0.3, 0.0])
        vio = self._vio((R_ic, np.array([0.1, 0.0, 0.0])),
                        _FixedVisual(first))
        out = vio.estimate(_frame(t=0.0))
        np.testing.assert_allclose(vio.ekf.state.R_w_i, np.eye(3),
                                   atol=1e-9)
        np.testing.assert_allclose(vio.ekf.state.p_w_i,
                                   [0.7, -0.3, 0.0], atol=1e-9)
        np.testing.assert_allclose(out.R, np.eye(3), atol=1e-9)

    def test_first_visual_snaps_to_body_frame_pose_when_enabled(self):
        # Flipping _apply_body_frame back on (Phase 5 re-activation
        # path) restores the body-frame-transformed anchor: state R
        # becomes R_w_c · R_imu_camᵀ = R_icᵀ on identity visual input.
        R_ic = R.from_rotvec([0, np.pi / 2, 0]).as_matrix()
        first = self._pose(np.eye(3), [0.0, 0.0, 0.0])
        vio = self._vio((R_ic, np.array([0.1, 0.0, 0.0])),
                        _FixedVisual(first))
        vio._apply_body_frame = True
        out = vio.estimate(_frame(t=0.0))
        np.testing.assert_allclose(vio.ekf.state.R_w_i, R_ic.T, atol=1e-9)
        np.testing.assert_allclose(out.R, R_ic.T, atol=1e-9)

    def test_anchor_skips_the_kalman_update(self):
        # An anchored frame must not move the covariance through update();
        # the orientation block stays at its initial value.
        R_ic = R.from_rotvec([0, np.pi / 2, 0]).as_matrix()
        first = self._pose(np.eye(3), [1.0, 0.0, 0.0])
        vio = self._vio((R_ic, np.zeros(3)), _FixedVisual(first))
        P0 = vio.ekf.covariance.copy()
        vio.estimate(_frame(t=0.0))
        # Orientation P block unchanged → no update was applied.
        from state_estimation.visual_inertial_ekf import T_IDX
        np.testing.assert_allclose(vio.ekf.covariance[T_IDX, T_IDX],
                                   P0[T_IDX, T_IDX], atol=1e-12)

    def test_second_visual_runs_normal_update(self):
        # Two frames in: first anchors, second must shrink the
        # orientation covariance via the standard update.
        first  = self._pose(np.eye(3), [0.0, 0.0, 0.0], idx=0)
        second = self._pose(np.eye(3), [0.01, 0.0, 0.0], idx=1)
        vio = self._vio((np.eye(3), np.zeros(3)),
                        _SequenceVisual([first, second]))
        vio.estimate(_frame(t=0.0, idx=0))           # anchor
        P_after_anchor = vio.ekf.covariance.copy()
        vio.estimate(_frame(t=0.1, idx=1))           # real update
        from state_estimation.visual_inertial_ekf import T_IDX, P_IDX
        # P[orientation] must shrink — the update added information.
        assert (vio.ekf.covariance[T_IDX, T_IDX][0, 0]
                < P_after_anchor[T_IDX, T_IDX][0, 0])
        # P[position] must shrink too.
        assert (vio.ekf.covariance[P_IDX, P_IDX][0, 0]
                < P_after_anchor[P_IDX, P_IDX][0, 0])

    def test_reset_un_anchors(self):
        first = self._pose(np.eye(3), [1.0, 0.0, 0.0])
        vio = self._vio((np.eye(3), np.zeros(3)), _FixedVisual(first))
        vio.estimate(_frame(t=0.0))
        assert vio._anchored is True
        vio.reset()
        assert vio._anchored is False

    def test_identity_extrinsic_anchor_matches_visual(self):
        # With identity extrinsic the anchor is just the visual pose.
        first = self._pose(np.eye(3), [0.7, -0.3, 0.1])
        vio = self._vio(None, _FixedVisual(first))
        vio.estimate(_frame(t=0.0))
        np.testing.assert_allclose(vio.ekf.state.p_w_i,
                                   [0.7, -0.3, 0.1], atol=1e-9)
        np.testing.assert_allclose(vio.ekf.state.R_w_i, np.eye(3),
                                   atol=1e-9)
