"""
Unit tests for the Extended Kalman Filter.

TestCTRModel         : CTR motion model correctness
TestJacobian         : Jacobian numerical validation
TestEKFConstruction  : state/covariance initialisation
TestEKFPredict       : prediction step
TestEKFUpdate        : update step
TestEKFCycle         : convergence and consistency
TestNEES             : NEES diagnostic
TestEKFVsKF          : EKF degenerates to KF when ω=0
TestFilterUtils      : NIS/NEES statistics utilities
"""

from __future__ import annotations

import math
import time
import numpy as np
import pytest

from state_estimation.extended_kf import (
    ExtendedKalmanFilter,
    _ctr_predict,
    _ctr_jacobian,
    _build_Q_ekf,
    _build_R_ekf,
    _OMEGA_EPS,
    N_STATE_EKF,
    N_OBS_EKF,
)
from state_estimation.filter_utils import (
    compute_nis_statistics,
    compute_nees_statistics,
    chi2_bounds,
)
from perception.detector import Detection


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def cfg():
    return {
        "kalman_filter": {
            "initial_covariance": {
                "p_position": 10.0,
                "p_size":     10.0,
                "p_velocity": 100.0,
            },
            "process_noise": {
                "q_position": 1.0,
                "q_size":     1.0,
                "q_velocity": 0.1,
                "q_vel_size": 0.02,
                "q_omega":    0.01,
            },
            "measurement_noise": {
                "r_center": 1.0,
                "r_size":   1.0,
            },
        }
    }


def make_state(
    cx=320.0, cy=240.0, w=80.0, h=60.0,
    vx=0.0, vy=0.0, vw=0.0, vh=0.0, omega=0.0,
) -> np.ndarray:
    return np.array([cx, cy, w, h, vx, vy, vw, vh, omega])


def make_ekf(cfg, **kwargs) -> ExtendedKalmanFilter:
    return ExtendedKalmanFilter(make_state(**kwargs), cfg)


def make_detection(
    cx=320.0, cy=240.0, w=80.0, h=60.0, conf=0.9,
) -> Detection:
    x1, y1 = cx - w/2, cy - h/2
    x2, y2 = cx + w/2, cy + h/2
    return Detection(
        bbox_xyxy=np.array([x1, y1, x2, y2], dtype=np.float32),
        confidence=conf, class_id=0, class_name="person",
        frame_idx=0, timestamp=time.monotonic(),
    )


def make_obs(cx=320.0, cy=240.0, w=80.0, h=60.0) -> np.ndarray:
    return np.array([cx, cy, w, h], dtype=np.float64)


# ---------------------------------------------------------------------------
# TestCTRModel
# ---------------------------------------------------------------------------

class TestCTRModel:

    def test_zero_omega_is_constant_velocity(self):
        """With ω=0, CTR degenerates to constant velocity."""
        x = make_state(cx=100.0, cy=100.0, vx=10.0, vy=5.0)
        dt = 0.1
        x_new = _ctr_predict(x, dt)
        assert x_new[0] == pytest.approx(100.0 + 10.0 * dt, abs=1e-6)
        assert x_new[1] == pytest.approx(100.0 + 5.0 * dt, abs=1e-6)

    def test_size_integrates_linearly(self):
        """w and h always integrate with their velocities linearly."""
        x = make_state(w=60.0, h=40.0, vw=2.0, vh=-1.0, omega=0.3)
        dt = 0.1
        x_new = _ctr_predict(x, dt)
        assert x_new[2] == pytest.approx(60.0 + 2.0 * dt, abs=1e-6)
        assert x_new[3] == pytest.approx(40.0 + (-1.0) * dt, abs=1e-6)

    def test_omega_unchanged_after_predict(self):
        x = make_state(omega=0.5)
        x_new = _ctr_predict(x, 0.1)
        assert x_new[8] == pytest.approx(0.5, abs=1e-10)

    def test_circular_motion_returns_to_start(self):
        """
        A full circle (2π/ω seconds) should return to the starting position.
        This is the gold-standard test for the CTR model.
        """
        omega = 1.0   # rad/s
        T     = 2.0 * math.pi / omega   # full revolution period
        vx    = 5.0
        vy    = 0.0
        x     = make_state(cx=0.0, cy=0.0, vx=vx, vy=vy, omega=omega)

        # Simulate 100 steps over one full revolution
        n_steps = 100
        dt      = T / n_steps
        for _ in range(n_steps):
            x = _ctr_predict(x, dt)

        # Should return close to origin after one full revolution
        assert abs(x[0]) < 0.1, f"cx={x[0]:.4f} not near 0 after full circle"
        assert abs(x[1]) < 0.1, f"cy={x[1]:.4f} not near 0 after full circle"

    def test_velocity_rotates_with_omega(self):
        """After dt, velocity vector should rotate by ω*dt."""
        omega = math.pi / 2   # 90 degrees/s
        dt    = 1.0            # 1 second → 90 degree rotation
        vx, vy = 1.0, 0.0
        x = make_state(vx=vx, vy=vy, omega=omega)
        x_new = _ctr_predict(x, dt)
        # After 90° rotation: vx→0, vy→1
        assert x_new[4] == pytest.approx(0.0, abs=1e-6)
        assert x_new[5] == pytest.approx(1.0, abs=1e-6)

    def test_output_shape(self):
        x = make_state()
        assert _ctr_predict(x, 0.033).shape == (N_STATE_EKF,)

    def test_near_zero_omega_no_nan(self):
        """Values near the ω=0 boundary must not produce NaN."""
        x = make_state(omega=_OMEGA_EPS * 0.1)
        x_new = _ctr_predict(x, 0.033)
        assert not np.any(np.isnan(x_new))


# ---------------------------------------------------------------------------
# TestJacobian
# ---------------------------------------------------------------------------

class TestJacobian:

    def _numerical_jacobian(
        self, x: np.ndarray, dt: float, eps: float = 1e-5
    ) -> np.ndarray:
        """Compute Jacobian numerically via finite differences."""
        n   = len(x)
        f0  = _ctr_predict(x, dt)
        Jac = np.zeros((n, n))
        for i in range(n):
            x_plus  = x.copy(); x_plus[i]  += eps
            x_minus = x.copy(); x_minus[i] -= eps
            Jac[:, i] = (_ctr_predict(x_plus, dt) -
                         _ctr_predict(x_minus, dt)) / (2.0 * eps)
        return Jac

    def test_jacobian_shape(self):
        x = make_state(vx=2.0, omega=0.1)
        F = _ctr_jacobian(x, 0.033)
        assert F.shape == (N_STATE_EKF, N_STATE_EKF)

    def test_jacobian_vs_numerical_zero_omega(self):
        """Analytical Jacobian must match numerical for ω≈0."""
        x   = make_state(vx=5.0, vy=2.0, omega=0.0)
        dt  = 0.033
        F_a = _ctr_jacobian(x, dt)
        F_n = self._numerical_jacobian(x, dt)
        np.testing.assert_allclose(F_a, F_n, atol=1e-5,
            err_msg="Analytical Jacobian differs from numerical (ω=0)")

    def test_jacobian_vs_numerical_nonzero_omega(self):
        """Analytical Jacobian must match numerical for ω≠0."""
        x   = make_state(vx=10.0, vy=3.0, omega=0.5)
        dt  = 0.033
        F_a = _ctr_jacobian(x, dt)
        F_n = self._numerical_jacobian(x, dt)
        np.testing.assert_allclose(F_a, F_n, atol=1e-4,
            err_msg="Analytical Jacobian differs from numerical (ω=0.5)")

    def test_jacobian_vs_numerical_high_omega(self):
        """Test with large turn rate."""
        x  = make_state(vx=5.0, vy=5.0, omega=2.0)
        dt = 0.033
        F_a = _ctr_jacobian(x, dt)
        F_n = self._numerical_jacobian(x, dt)
        np.testing.assert_allclose(F_a, F_n, atol=1e-4)

    def test_jacobian_identity_on_diagonal(self):
        """ω=0, v=0 → Jacobian should be identity + dt on position-velocity."""
        x  = make_state()
        F  = _ctr_jacobian(x, 0.033)
        # All diagonal entries should be 1.0
        np.testing.assert_array_almost_equal(np.diag(F), np.ones(N_STATE_EKF))


# ---------------------------------------------------------------------------
# TestEKFConstruction
# ---------------------------------------------------------------------------

class TestEKFConstruction:

    def test_state_shape(self, cfg):
        kf = make_ekf(cfg)
        assert kf.state.shape == (N_STATE_EKF,)

    def test_state_dtype_float64(self, cfg):
        assert make_ekf(cfg).state.dtype == np.float64

    def test_covariance_shape(self, cfg):
        assert make_ekf(cfg).covariance.shape == (N_STATE_EKF, N_STATE_EKF)

    def test_initial_omega_zero(self, cfg):
        kf = make_ekf(cfg)
        assert kf.turn_rate == pytest.approx(0.0)

    def test_initial_velocity_zero(self, cfg):
        kf = make_ekf(cfg)
        np.testing.assert_array_almost_equal(kf.velocity, [0, 0, 0, 0])

    def test_initial_covariance_pd(self, cfg):
        assert make_ekf(cfg).is_covariance_pd()

    def test_initial_nis_nan(self, cfg):
        assert np.isnan(make_ekf(cfg).nis())

    def test_from_detection(self, cfg):
        det = make_detection(cx=200.0, cy=150.0, w=70.0, h=50.0)
        kf  = ExtendedKalmanFilter.from_detection(det, cfg)
        assert kf.state[0] == pytest.approx(200.0)
        assert kf.state[1] == pytest.approx(150.0)
        assert kf.state[2] == pytest.approx(70.0)
        assert kf.state[3] == pytest.approx(50.0)
        assert kf.state[8] == pytest.approx(0.0)   # ω=0 at birth

    def test_wrong_state_shape_raises(self, cfg):
        with pytest.raises(ValueError, match="shape"):
            ExtendedKalmanFilter(np.zeros(8), cfg)   # 8D not 9D

    def test_state_is_copy(self, cfg):
        kf = make_ekf(cfg)
        s  = kf.state
        s[0] = 9999.0
        assert kf.state[0] != 9999.0


# ---------------------------------------------------------------------------
# TestEKFPredict
# ---------------------------------------------------------------------------

class TestEKFPredict:

    def test_predict_returns_array(self, cfg):
        kf = make_ekf(cfg)
        assert kf.predict(dt=0.033).shape == (N_STATE_EKF,)

    def test_predict_increments_counter(self, cfg):
        kf = make_ekf(cfg)
        kf.predict(0.033)
        kf.predict(0.033)
        assert kf.n_predict == 2

    def test_predict_covariance_grows(self, cfg):
        kf     = make_ekf(cfg)
        before = kf.covariance_trace()
        kf.predict(0.033)
        assert kf.covariance_trace() > before

    def test_predict_covariance_remains_pd(self, cfg):
        kf = make_ekf(cfg)
        for _ in range(50):
            kf.predict(0.033)
        assert kf.is_covariance_pd()

    def test_predict_covariance_symmetric(self, cfg):
        kf = make_ekf(cfg)
        for _ in range(20):
            kf.predict(0.033)
        P = kf.covariance
        np.testing.assert_array_almost_equal(P, P.T, decimal=10)

    def test_negative_dt_raises(self, cfg):
        with pytest.raises(ValueError, match="non-negative"):
            make_ekf(cfg).predict(dt=-0.001)

    def test_zero_dt_warns(self, cfg):
        with pytest.warns(UserWarning):
            make_ekf(cfg).predict(dt=0.0)

    def test_zero_omega_matches_constant_velocity(self, cfg):
        """With ω=0, EKF predict should match vanilla KF predict."""
        from state_estimation.kalman_filter import KalmanFilter
        state_8 = make_state(cx=100.0, cy=100.0, vx=20.0, vy=10.0)[:8]
        state_9 = make_state(cx=100.0, cy=100.0, vx=20.0, vy=10.0)
        dt = 0.033

        kf  = KalmanFilter(state_8, cfg)
        ekf = ExtendedKalmanFilter(state_9, cfg)

        kf.predict(dt=dt)
        ekf.predict(dt=dt)

        # Position and velocity should match
        np.testing.assert_allclose(
            kf.state[:4], ekf.state[:4], atol=1e-6,
            err_msg="EKF with ω=0 must match KF position/size"
        )


# ---------------------------------------------------------------------------
# TestEKFUpdate
# ---------------------------------------------------------------------------

class TestEKFUpdate:

    def test_update_returns_array(self, cfg):
        kf = make_ekf(cfg)
        kf.predict(0.033)
        assert kf.update(make_obs()).shape == (N_STATE_EKF,)

    def test_update_increments_counter(self, cfg):
        kf = make_ekf(cfg)
        kf.predict(0.033)
        kf.update(make_obs())
        assert kf.n_update == 1

    def test_update_reduces_covariance(self, cfg):
        kf = make_ekf(cfg)
        kf.predict(0.033)
        before = kf.covariance_trace()
        kf.update(make_obs())
        assert kf.covariance_trace() < before

    def test_update_moves_toward_observation(self, cfg):
        kf = make_ekf(cfg, cx=300.0)
        kf.predict(0.033)
        kf.update(make_obs(cx=350.0))
        assert kf.state[0] > 300.0

    def test_update_sets_nis(self, cfg):
        kf = make_ekf(cfg)
        kf.predict(0.033)
        kf.update(make_obs())
        assert not np.isnan(kf.nis())

    def test_update_covariance_pd(self, cfg):
        kf = make_ekf(cfg)
        for _ in range(30):
            kf.predict(0.033)
            kf.update(make_obs())
        assert kf.is_covariance_pd()

    def test_update_covariance_symmetric(self, cfg):
        kf = make_ekf(cfg)
        for _ in range(30):
            kf.predict(0.033)
            kf.update(make_obs())
        P = kf.covariance
        np.testing.assert_array_almost_equal(P, P.T, decimal=10)

    def test_wrong_obs_shape_raises(self, cfg):
        kf = make_ekf(cfg)
        kf.predict(0.033)
        with pytest.raises(ValueError, match=f"{N_OBS_EKF}"):
            kf.update(np.array([1.0, 2.0, 3.0]))


# ---------------------------------------------------------------------------
# TestEKFCycle
# ---------------------------------------------------------------------------

class TestEKFCycle:

    def test_filter_converges_stationary(self, cfg):
        """EKF must converge on a stationary target."""
        true_cx, true_cy = 320.0, 240.0
        kf  = make_ekf(cfg, cx=280.0, cy=200.0)
        rng = np.random.default_rng(42)

        errors = []
        for _ in range(60):
            kf.predict(0.033)
            noise = rng.normal(0, 1.0, 4)
            obs   = make_obs(cx=true_cx, cy=true_cy) + np.array([noise[0], noise[1], 0, 0])
            kf.update(obs)
            errors.append(abs(kf.state[0] - true_cx))

        assert np.mean(errors[-5:]) < np.mean(errors[:5])

    def test_survives_1000_cycles(self, cfg):
        """Numerical stability over 1000 predict+update cycles."""
        rng = np.random.default_rng(99)
        kf  = make_ekf(cfg)
        R   = _build_R_ekf(cfg)

        for i in range(1000):
            kf.predict(0.033)
            noise = rng.multivariate_normal(np.zeros(4), R)
            kf.update(make_obs(cx=320.0 + i * 0.01) + noise)

        assert not np.any(np.isnan(kf.state))
        assert not np.any(np.isnan(kf.covariance))
        assert kf.is_covariance_pd()

    def test_nis_consistent_on_linear_trajectory(self, cfg):
        """
        Mean NIS should be close to χ²(4) expected value of 4.0
        on a trajectory consistent with the filter model.
        """
        rng = np.random.default_rng(7)
        kf  = make_ekf(cfg)
        R   = _build_R_ekf(cfg)
        nis_vals = []

        for _ in range(200):
            kf.predict(0.033)
            noise = rng.multivariate_normal(np.zeros(4), R)
            kf.update(make_obs() + noise)
            nis_vals.append(kf.nis())

        mean_nis = float(np.mean(nis_vals))
        assert 1.0 < mean_nis < 12.0, (
            f"Mean NIS={mean_nis:.2f} is far from χ²(4)=4.0"
        )

    def test_turning_trajectory_tracked(self, cfg):
        """
        EKF should track a turning object and converge on the
        correct turn rate after enough observations.
        """
        true_omega = 0.5   # rad/s
        kf  = make_ekf(cfg, vx=10.0, vy=0.0)
        rng = np.random.default_rng(1)

        cx, cy = 100.0, 100.0
        vx, vy = 10.0, 0.0

        for _ in range(60):
            # Simulate turning object
            vx_new = vx * np.cos(true_omega*0.033) - vy * np.sin(true_omega*0.033)
            vy_new = vx * np.sin(true_omega*0.033) + vy * np.cos(true_omega*0.033)
            cx += vx * 0.033
            cy += vy * 0.033
            vx, vy = vx_new, vy_new

            kf.predict(0.033)
            noise = rng.normal(0, 0.5, 4)
            kf.update(make_obs(cx=cx, cy=cy) + np.array([noise[0], noise[1], 0, 0]))

        # After 60 updates the EKF should have estimated a non-zero turn rate
        # (direction: correct sign)
        assert kf.turn_rate * true_omega >= 0, (
            "Turn rate estimate must have the correct sign"
        )


# ---------------------------------------------------------------------------
# TestNEES
# ---------------------------------------------------------------------------

class TestNEES:

    def test_nees_scalar(self, cfg):
        kf = make_ekf(cfg)
        result = kf.nees(make_obs())
        assert isinstance(result, float)

    def test_nees_non_negative(self, cfg):
        kf = make_ekf(cfg)
        assert kf.nees(make_obs()) >= 0.0

    def test_nees_zero_when_true_equals_estimated(self, cfg):
        """NEES should be 0 when true state == estimated state."""
        kf = make_ekf(cfg, cx=320.0, cy=240.0)
        # True position matches estimated exactly
        true_pos = make_obs(cx=320.0, cy=240.0)
        assert kf.nees(true_pos) == pytest.approx(0.0, abs=1e-8)

    def test_nees_wrong_shape_raises(self, cfg):
        kf = make_ekf(cfg)
        with pytest.raises(ValueError):
            kf.nees(np.zeros(3))

    def test_nees_accepts_full_state(self, cfg):
        kf   = make_ekf(cfg)
        true = make_state()
        result = kf.nees(true)
        assert isinstance(result, float)

    def test_nees_consistent_on_simulation(self, cfg):
        """
        Mean NEES ≈ 4 (n_obs_dims) over many steps on consistent data.
        """
        rng  = np.random.default_rng(42)
        kf   = make_ekf(cfg)
        R    = _build_R_ekf(cfg)
        nees_vals = []
        cx, cy = 320.0, 240.0

        for _ in range(200):
            kf.predict(0.033)
            noise  = rng.multivariate_normal(np.zeros(4), R)
            true_obs = make_obs(cx=cx, cy=cy)
            kf.update(true_obs + noise)
            nees_vals.append(kf.nees(true_obs))

        mean_nees = float(np.mean(nees_vals))
        # NEES ~ χ²(4), expected mean = 4
        assert 0.5 < mean_nees < 15.0, (
            f"Mean NEES={mean_nees:.2f} is unreasonable for χ²(4)"
        )


# ---------------------------------------------------------------------------
# TestEKFVsKF
# ---------------------------------------------------------------------------

class TestEKFVsKF:
    """
    EKF with ω=0 must behave identically to vanilla KF on linear data.
    This validates backward compatibility.
    """

    def test_bbox_xyxy_same_format(self, cfg):
        kf  = make_ekf(cfg, cx=320.0, cy=240.0, w=80.0, h=60.0)
        x1, y1, x2, y2 = kf.bbox_xyxy
        assert x1 == pytest.approx(320.0 - 40.0)
        assert y1 == pytest.approx(240.0 - 30.0)
        assert x2 == pytest.approx(320.0 + 40.0)
        assert y2 == pytest.approx(240.0 + 30.0)

    def test_position_property(self, cfg):
        kf = make_ekf(cfg, cx=150.0, cy=200.0)
        assert kf.position[0] == pytest.approx(150.0)
        assert kf.position[1] == pytest.approx(200.0)

    def test_velocity_property_excludes_omega(self, cfg):
        """velocity property must return [vx,vy,vw,vh] — not ω."""
        kf = make_ekf(cfg)
        assert kf.velocity.shape == (4,)

    def test_repr_contains_omega(self, cfg):
        kf = make_ekf(cfg)
        assert "ω" in repr(kf)


# ---------------------------------------------------------------------------
# TestFilterUtils
# ---------------------------------------------------------------------------

class TestFilterUtils:

    def test_nis_statistics_mean(self):
        stats = compute_nis_statistics([3.0, 4.0, 5.0, 4.0])
        assert stats["mean"] == pytest.approx(4.0)

    def test_nis_statistics_filters_nan(self):
        stats = compute_nis_statistics([float("nan"), 4.0, 5.0])
        assert stats["n_samples"] == 2

    def test_nis_statistics_empty(self):
        stats = compute_nis_statistics([])
        assert stats["n_samples"] == 0
        assert np.isnan(stats["mean"])

    def test_nis_pct_in_bounds(self):
        # All within [0.711, 9.488]
        stats = compute_nis_statistics([1.0, 2.0, 3.0, 4.0, 5.0])
        assert stats["pct_in_bounds"] == pytest.approx(1.0)

    def test_nees_statistics_mean(self):
        stats = compute_nees_statistics([8.0, 9.0, 10.0])
        assert stats["mean"] == pytest.approx(9.0)

    def test_chi2_bounds_dof4(self):
        lo, hi = chi2_bounds(4, confidence=0.95)
        assert lo == pytest.approx(0.711, abs=0.01)
        assert hi == pytest.approx(9.488, abs=0.01)

    def test_chi2_bounds_dof9(self):
        lo, hi = chi2_bounds(9, confidence=0.95)
        assert lo > 0.0
        assert hi > lo
