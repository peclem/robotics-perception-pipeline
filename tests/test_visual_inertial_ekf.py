"""
Unit tests for the visual-inertial error-state EKF.

TestConstruction         : initial covariance + at-rest defaults
TestPredictMath          : nominal-state propagation correctness
TestPredictCovariance    : covariance growth + PSD-ness through prediction
TestUpdate               : innovation, gain, Joseph-form PSD-ness
TestEndToEnd             : SyntheticIMU + IMUPreintegrator + visual updates
                           reproduce known trajectories
TestBiasEstimation       : visual updates correct a deliberately biased IMU
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from perception.imu_interface import IMUSample, SyntheticIMU
from perception.pose_estimator import CameraPose
from state_estimation.imu_preintegration import (
    IMUPreintegrator, PreintegratedMeasurement,
)
from state_estimation.visual_inertial_ekf import (
    VIOConfig, VIONominalState, VisualInertialEKF,
    P_IDX, V_IDX, T_IDX, BG_IDX, BA_IDX, S_IDX,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _identity_preint(dt: float = 0.0) -> PreintegratedMeasurement:
    """An empty pre-integration window — caller can override dt."""
    pim = PreintegratedMeasurement.identity()
    pim.dt = dt
    return pim


def _zero_gravity_cfg() -> VIOConfig:
    """Zero-out gravity so 'no motion' really means no nominal drift."""
    return VIOConfig(gravity_w=(0.0, 0.0, 0.0))


def _imu_batch(
    motion: str, n_samples: int, start_t: float, rate_hz: float = 200.0,
    gyro_bias: np.ndarray | None = None,
) -> list[IMUSample]:
    """
    Deterministic IMU batch from SyntheticIMU.generate_batch, with an
    optional additive gyro bias for the bias-estimation tests.
    """
    imu = SyntheticIMU(motion=motion, rate_hz=rate_hz, seed=0)
    samples = imu.generate_batch(n_samples=n_samples, start_t=start_t)
    if gyro_bias is not None:
        biased = []
        for s in samples:
            biased.append(IMUSample(
                accel=s.accel, gyro=s.gyro + gyro_bias, timestamp=s.timestamp,
            ))
        return biased
    return samples


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestConstruction:

    def test_default_state_is_at_rest(self):
        ekf = VisualInertialEKF(VIOConfig())
        s = ekf.state
        assert np.allclose(s.p_w_i, 0.0)
        assert np.allclose(s.v_w_i, 0.0)
        assert np.allclose(s.R_w_i, np.eye(3))
        assert np.allclose(s.b_g, 0.0)
        assert np.allclose(s.b_a, 0.0)

    def test_initial_covariance_matches_config(self):
        cfg = VIOConfig(
            init_position_std_m=0.2, init_velocity_std_mps=0.3,
            init_orientation_std_rad=0.05, init_bias_gyro_std=1e-3,
            init_bias_accel_std=1e-2,
        )
        ekf = VisualInertialEKF(cfg)
        P = ekf.covariance
        assert np.isclose(P[0, 0], 0.2 ** 2)
        assert np.isclose(P[3, 3], 0.3 ** 2)
        assert np.isclose(P[6, 6], 0.05 ** 2)
        assert np.isclose(P[9, 9], 1e-3 ** 2)
        assert np.isclose(P[12, 12], 1e-2 ** 2)
        # PSD + symmetric.
        assert np.allclose(P, P.T)
        eigvals = np.linalg.eigvalsh(P)
        assert eigvals.min() >= -1e-12

    def test_custom_initial_state(self):
        s0 = VIONominalState(
            p_w_i=np.array([1.0, 2.0, 3.0]),
            v_w_i=np.array([0.1, 0.0, 0.0]),
            R_w_i=R.from_rotvec([0.0, 0.0, 0.5]).as_matrix(),
            b_g=np.zeros(3), b_a=np.zeros(3),
        )
        ekf = VisualInertialEKF(VIOConfig(), initial_state=s0,
                                initial_timestamp=10.0)
        assert np.allclose(ekf.state.p_w_i, [1.0, 2.0, 3.0])
        assert ekf.timestamp == 10.0


# ---------------------------------------------------------------------------
# Predict — nominal-state math
# ---------------------------------------------------------------------------

class TestPredictMath:

    def test_empty_preint_is_noop(self):
        ekf = VisualInertialEKF(_zero_gravity_cfg())
        before_p = ekf.state.p_w_i.copy()
        before_P = ekf.covariance.copy()
        ekf.predict(_identity_preint(dt=0.0))
        assert np.allclose(ekf.state.p_w_i, before_p)
        assert np.allclose(ekf.covariance, before_P)

    def test_gravity_only_freefall(self):
        """No motion in body frame + zero rotation → position drops by 0.5gt²."""
        cfg = VIOConfig()  # default gravity (0, 0, -9.81)
        ekf = VisualInertialEKF(cfg)
        # Identity preint with dt=1 s (no IMU motion).
        ekf.predict(_identity_preint(dt=1.0))
        # In freefall the body doesn't measure motion (a = 0), but the
        # world-frame state should reflect gravity:
        #   p_z = -0.5 * 9.81 * 1² = -4.905 m
        #   v_z = -9.81 m/s
        assert np.isclose(ekf.state.p_w_i[2], -4.905, atol=1e-6)
        assert np.isclose(ekf.state.v_w_i[2], -9.81,  atol=1e-6)

    def test_constant_velocity_in_world(self):
        """
        With zero gravity + identity preint, an initial v_w_i should
        carry through linearly: p(t) = p₀ + v·t.
        """
        cfg = _zero_gravity_cfg()
        s0 = VIONominalState(
            p_w_i=np.zeros(3), v_w_i=np.array([1.0, 0.0, 0.0]),
            R_w_i=np.eye(3), b_g=np.zeros(3), b_a=np.zeros(3),
        )
        ekf = VisualInertialEKF(cfg, initial_state=s0)
        ekf.predict(_identity_preint(dt=2.5))
        assert np.allclose(ekf.state.p_w_i, [2.5, 0.0, 0.0])
        assert np.allclose(ekf.state.v_w_i, [1.0, 0.0, 0.0])

    def test_body_frame_dp_rotated_into_world(self):
        """
        With the body rotated 90° about Z, a body-frame Δp = (1, 0, 0)
        should become world-frame (0, 1, 0).
        """
        cfg = _zero_gravity_cfg()
        Rz90 = R.from_rotvec([0, 0, np.pi / 2]).as_matrix()
        s0 = VIONominalState(
            p_w_i=np.zeros(3), v_w_i=np.zeros(3), R_w_i=Rz90,
            b_g=np.zeros(3), b_a=np.zeros(3),
        )
        ekf = VisualInertialEKF(cfg, initial_state=s0)
        preint = PreintegratedMeasurement(
            delta_R=np.eye(3),
            delta_v=np.zeros(3),
            delta_p=np.array([1.0, 0.0, 0.0]),
            dt=1.0, covariance=np.zeros((9, 9)), n_samples=2,
        )
        ekf.predict(preint)
        assert np.allclose(ekf.state.p_w_i, [0.0, 1.0, 0.0], atol=1e-12)


# ---------------------------------------------------------------------------
# Predict — covariance
# ---------------------------------------------------------------------------

class TestPredictCovariance:

    def test_covariance_grows_under_bias_random_walk(self):
        cfg = VIOConfig(
            bias_gyro_random_walk=1e-3, bias_accel_random_walk=1e-2,
        )
        ekf = VisualInertialEKF(cfg)
        bg_var0 = ekf.covariance[BG_IDX, BG_IDX][0, 0]
        ba_var0 = ekf.covariance[BA_IDX, BA_IDX][0, 0]
        ekf.predict(_identity_preint(dt=1.0))
        bg_var1 = ekf.covariance[BG_IDX, BG_IDX][0, 0]
        ba_var1 = ekf.covariance[BA_IDX, BA_IDX][0, 0]
        # Random walk adds σ² · dt to the bias variance.
        assert bg_var1 > bg_var0
        assert ba_var1 > ba_var0
        assert np.isclose(bg_var1 - bg_var0, (1e-3) ** 2 * 1.0, atol=1e-12)
        assert np.isclose(ba_var1 - ba_var0, (1e-2) ** 2 * 1.0, atol=1e-12)

    def test_covariance_remains_symmetric_and_psd(self):
        """Run the pre-integrator over real IMU samples and ensure P stays sane."""
        pre = IMUPreintegrator()
        ekf = VisualInertialEKF(VIOConfig())
        t = 0.0
        for _ in range(5):
            samples = _imu_batch("stationary_with_gravity", 20, t)
            preint = pre.integrate(samples)
            ekf.predict(preint)
            P = ekf.covariance
            assert np.allclose(P, P.T, atol=1e-9)
            assert np.linalg.eigvalsh(P).min() >= -1e-9
            t = samples[-1].timestamp + 1.0 / 200.0


# ---------------------------------------------------------------------------
# Update
# ---------------------------------------------------------------------------

class TestUpdate:

    def test_perfect_position_update_zeros_innovation(self):
        """A measurement that matches the predicted state shouldn't move it."""
        cfg = _zero_gravity_cfg()
        ekf = VisualInertialEKF(cfg)
        before_p = ekf.state.p_w_i.copy()
        before_t = ekf.state.R_w_i.copy()
        # Measurement equal to current state → y=0 → no correction.
        z = CameraPose(R=ekf.state.R_w_i.copy(), t=ekf.state.p_w_i.copy(),
                       timestamp=0.0, frame_idx=0, source="test")
        ekf.update(z)
        assert np.allclose(ekf.state.p_w_i, before_p, atol=1e-12)
        assert np.allclose(ekf.state.R_w_i, before_t, atol=1e-12)

    def test_position_jump_pulls_state_toward_measurement(self):
        cfg = _zero_gravity_cfg()
        ekf = VisualInertialEKF(cfg)
        z_t = np.array([0.5, 0.0, 0.0])
        z = CameraPose(R=np.eye(3), t=z_t, timestamp=0.0, frame_idx=0,
                       source="test")
        ekf.update(z)
        # With a tight visual std (5 cm) and a 10 cm initial position std,
        # the gain on position is ≈ 0.8 → state should move ~80% of the way.
        moved = ekf.state.p_w_i[0]
        assert 0.3 < moved < z_t[0] + 1e-9, f"unexpected jump: {moved}"

    def test_rotation_update_via_log_map(self):
        cfg = _zero_gravity_cfg()
        ekf = VisualInertialEKF(cfg)
        # Measurement: rotate 0.1 rad about Z.
        z_R = R.from_rotvec([0.0, 0.0, 0.1]).as_matrix()
        z = CameraPose(R=z_R, t=np.zeros(3), timestamp=0.0,
                       frame_idx=0, source="test")
        ekf.update(z)
        # Recover the on-axis angle from the resulting state.
        ang = R.from_matrix(ekf.state.R_w_i).as_rotvec()
        assert ang[0] == pytest.approx(0.0, abs=1e-6)
        assert ang[1] == pytest.approx(0.0, abs=1e-6)
        assert 0.03 < ang[2] < 0.1 + 1e-9

    def test_update_keeps_covariance_psd_joseph_form(self):
        cfg = VIOConfig()
        ekf = VisualInertialEKF(cfg)
        z = CameraPose(R=R.from_rotvec([0.05, 0.0, 0.0]).as_matrix(),
                       t=np.array([1.0, 0.0, 0.0]),
                       timestamp=0.0, frame_idx=0, source="test")
        ekf.update(z)
        P = ekf.covariance
        assert np.allclose(P, P.T, atol=1e-9)
        # Joseph form is the reason this stays PSD even with marginal
        # innovation variance — assert it.
        assert np.linalg.eigvalsh(P).min() >= -1e-9

    def test_position_covariance_shrinks_after_update(self):
        ekf = VisualInertialEKF(VIOConfig())
        var_before = ekf.covariance[P_IDX, P_IDX][0, 0]
        z = CameraPose(R=np.eye(3), t=np.array([0.1, 0.0, 0.0]),
                       timestamp=0.0, frame_idx=0, source="test")
        ekf.update(z)
        var_after = ekf.covariance[P_IDX, P_IDX][0, 0]
        assert var_after < var_before


# ---------------------------------------------------------------------------
# End-to-end: SyntheticIMU → preintegrator → EKF
# ---------------------------------------------------------------------------

class TestEndToEnd:

    def test_stationary_imu_with_periodic_visual_locks_at_origin(self):
        """
        IMU is stationary with gravity. Pre-integration's body-frame
        Δv ≈ R·(0,0,g)·dt, which the EKF cancels against the world-frame
        gravity term: state should stay at rest under prediction alone.
        Visual updates ('still at origin') reinforce this.
        """
        pre = IMUPreintegrator()
        cfg = VIOConfig(
            visual_position_std_m=1e-3,
            visual_orientation_std_rad=1e-3,
        )
        ekf = VisualInertialEKF(cfg)
        t = 0.0
        for _ in range(10):
            samples = _imu_batch("stationary_with_gravity", 20, t)
            preint = pre.integrate(samples)
            ekf.predict(preint)
            ekf.update(CameraPose(
                R=np.eye(3), t=np.zeros(3),
                timestamp=ekf.timestamp, frame_idx=0, source="test",
            ))
            t = samples[-1].timestamp + 1.0 / 200.0
        assert np.linalg.norm(ekf.state.p_w_i) < 0.05, ekf.state.p_w_i
        assert np.linalg.norm(ekf.state.v_w_i) < 0.05, ekf.state.v_w_i

    def test_constant_world_motion_tracked_by_visual(self):
        """
        IMU stationary-with-gravity (no body acceleration apart from
        gravity reaction). Visual says: 'camera slides at 0.2 m/s along
        +X'. The EKF should track the position and back out velocity.

        Scale is LOCKED here (init_scale_std≈0): with no acceleration the
        visual scale is unobservable, so this test fixes it to isolate
        the position-tracking behaviour. Scale estimation under proper
        excitation is covered by TestScaleEstimation.
        """
        pre = IMUPreintegrator()
        cfg = VIOConfig(
            visual_position_std_m=5e-3, visual_orientation_std_rad=5e-3,
            init_scale_std=1e-6, scale_random_walk=0.0,
        )
        ekf = VisualInertialEKF(cfg)

        v_world = 0.2  # m/s along +X
        t = 0.0
        t_world = 0.0
        dt_visual = 0.1
        for k in range(20):
            samples = _imu_batch("stationary_with_gravity", 20, t)
            preint = pre.integrate(samples)
            ekf.predict(preint)
            t = samples[-1].timestamp + 1.0 / 200.0
            t_world += dt_visual
            ekf.update(CameraPose(
                R=np.eye(3),
                t=np.array([v_world * t_world, 0.0, 0.0]),
                timestamp=ekf.timestamp, frame_idx=k, source="test",
            ))
        # Final position should have caught up to the visual measurement.
        assert np.isclose(ekf.state.p_w_i[0], v_world * t_world, atol=0.05)
        # Velocity inferred through pose-only updates converges slowly —
        # tolerate a generous error band but require directional correctness.
        assert 0.05 < ekf.state.v_w_i[0] < 0.4


# ---------------------------------------------------------------------------
# Bias estimation
# ---------------------------------------------------------------------------

class TestBiasEstimation:

    def test_visual_update_drives_position_correction_with_biased_imu(self):
        """
        Stationary world, *gyro-biased* IMU. IMU-only prediction would
        rotate the body. Tight visual updates ('no rotation') keep the
        orientation near identity — the loop the fuser is designed to close.
        """
        bias = np.array([0.0, 0.0, 0.05])  # 0.05 rad/s gyro bias on Z
        pre = IMUPreintegrator()
        cfg = VIOConfig(
            visual_position_std_m=1e-3,
            visual_orientation_std_rad=1e-3,
        )
        ekf = VisualInertialEKF(cfg)
        t = 0.0
        for _ in range(20):
            samples = _imu_batch("stationary_with_gravity", 20, t,
                                 gyro_bias=bias)
            preint = pre.integrate(samples)
            ekf.predict(preint)
            ekf.update(CameraPose(
                R=np.eye(3), t=np.zeros(3),
                timestamp=ekf.timestamp, frame_idx=0, source="test",
            ))
            t = samples[-1].timestamp + 1.0 / 200.0

        # Orientation stays near identity thanks to the visual updates.
        ang = np.linalg.norm(R.from_matrix(ekf.state.R_w_i).as_rotvec())
        assert ang < 0.1, f"orientation drifted to {ang:.3f} rad"
        # Bias covariance should not have *grown* materially under
        # repeated updates (the small bias_gyro_random_walk·dt growth
        # is fine; the visual updates contribute weak information that
        # at least keeps it bounded).
        bg_var = ekf.covariance[BG_IDX, BG_IDX][2, 2]
        assert bg_var < cfg.init_bias_gyro_std ** 2 * 1.05


# ---------------------------------------------------------------------------
# Visual scale estimation (state 16)
# ---------------------------------------------------------------------------

class TestScaleEstimation:

    def test_default_scale_is_unity(self):
        assert VIONominalState.at_rest().scale == 1.0
        assert VisualInertialEKF(VIOConfig()).state.scale == 1.0

    def test_covariance_is_16x16(self):
        assert VisualInertialEKF(VIOConfig()).covariance.shape == (16, 16)

    def test_initial_scale_variance_from_config(self):
        ekf = VisualInertialEKF(VIOConfig(init_scale_std=0.4))
        assert np.isclose(ekf.covariance[S_IDX, S_IDX][0, 0], 0.4 ** 2)

    def test_scale_survives_state_copy(self):
        s = VIONominalState(
            p_w_i=np.zeros(3), v_w_i=np.zeros(3), R_w_i=np.eye(3),
            b_g=np.zeros(3), b_a=np.zeros(3), scale=1.7,
        )
        assert s.copy().scale == 1.7

    def test_locked_scale_stays_unity(self):
        # init_scale_std≈0 ⇒ scale frozen at 1.0 through visual updates.
        cfg = VIOConfig(init_scale_std=1e-7, scale_random_walk=0.0)
        ekf = VisualInertialEKF(cfg)
        for k in range(10):
            ekf.update(CameraPose(
                R=np.eye(3), t=np.array([0.3 * k, 0.0, 0.0]),
                timestamp=0.0, frame_idx=k, source="test",
            ))
        assert ekf.state.scale == pytest.approx(1.0, abs=1e-3)

    def test_scale_converges_under_excitation(self):
        """
        The body genuinely accelerates, so the IMU sees true metric
        motion; the visual estimator reports that same trajectory at
        half scale (z = p_metric / 2). The EKF must recover scale → 2.0
        from the IMU-vs-visual disagreement — the thing a 15-D
        loosely-coupled VIO structurally cannot do.
        """
        pre = IMUPreintegrator()
        cfg = VIOConfig(gravity_w=(0.0, 0.0, 0.0),
                        visual_position_std_m=1e-2, init_scale_std=0.8)
        ekf = VisualInertialEKF(cfg)
        scale_true = 2.0
        t = 0.0
        for k in range(40):
            samples = _imu_batch("accel_x", 20, t)
            ekf.predict(pre.integrate(samples))
            t = samples[-1].timestamp + 1.0 / 200.0
            z = ekf.state.p_w_i / scale_true          # half-scale visual
            ekf.update(CameraPose(
                R=np.eye(3), t=z, timestamp=ekf.timestamp,
                frame_idx=k, source="test",
            ))
        assert ekf.state.scale == pytest.approx(scale_true, rel=0.2)

    def test_scale_unobservable_without_excitation_stays_bounded(self):
        # No acceleration ⇒ scale is not pinned, but the filter must
        # not diverge: scale stays finite and strictly positive.
        ekf = VisualInertialEKF(VIOConfig(gravity_w=(0.0, 0.0, 0.0)))
        for k in range(15):
            ekf.update(CameraPose(
                R=np.eye(3), t=np.array([0.1 * k, 0.0, 0.0]),
                timestamp=0.0, frame_idx=k, source="test",
            ))
        assert 1e-3 <= ekf.state.scale < 100.0


# ---------------------------------------------------------------------------
# Zero-velocity update (ZUPT)
# ---------------------------------------------------------------------------

class TestZeroVelocityUpdate:

    @staticmethod
    def _moving_ekf(v):
        s0 = VIONominalState(
            p_w_i=np.zeros(3), v_w_i=np.asarray(v, dtype=float),
            R_w_i=np.eye(3), b_g=np.zeros(3), b_a=np.zeros(3),
        )
        return VisualInertialEKF(VIOConfig(), initial_state=s0)

    def test_zupt_pulls_velocity_toward_zero(self):
        ekf = self._moving_ekf([1.0, -0.5, 0.3])     # ‖v‖ ≈ 1.16
        ekf.update_zero_velocity()
        assert np.linalg.norm(ekf.state.v_w_i) < 0.1

    def test_zupt_shrinks_velocity_covariance(self):
        ekf = self._moving_ekf([1.0, 0.0, 0.0])
        before = ekf.covariance[V_IDX, V_IDX][0, 0]
        ekf.update_zero_velocity()
        assert ekf.covariance[V_IDX, V_IDX][0, 0] < before

    def test_zupt_keeps_covariance_psd(self):
        ekf = self._moving_ekf([0.8, 0.2, -0.4])
        ekf.update_zero_velocity()
        P = ekf.covariance
        assert np.allclose(P, P.T)
        assert np.linalg.eigvalsh(P).min() >= -1e-12

    def test_zupt_noop_when_already_at_rest(self):
        ekf = self._moving_ekf([0.0, 0.0, 0.0])
        ekf.update_zero_velocity()
        np.testing.assert_allclose(ekf.state.v_w_i, np.zeros(3), atol=1e-12)

    def test_large_std_makes_zupt_weak(self):
        # A huge ZUPT 1σ ⇒ the filter barely trusts v=0 ⇒ little change.
        ekf = self._moving_ekf([1.0, 0.0, 0.0])
        ekf.update_zero_velocity(velocity_std=1e3)
        assert ekf.state.v_w_i[0] > 0.9
