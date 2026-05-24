"""
Unit tests for stationary IMU initialisation (state_estimation/imu_init.py).

Pure functions over IMUSample lists — pinned on inputs with a known
answer: a stationary window whose gyro is a constant bias must yield
exactly that bias; a moving or shaky start must yield None.
"""

from __future__ import annotations

import numpy as np
import pytest

from perception.imu_interface import IMUSample
from state_estimation.imu_init import (
    estimate_gyro_bias, estimate_initial_state, is_stationary,
)
from state_estimation.visual_inertial_ekf import VIONominalState


_BIAS = np.array([0.012, -0.020, 0.005])   # small constant gyro bias
_GRAV = np.array([0.0, 0.0, 9.81])         # stationary specific force


def _samples(gyros, accels=None, dt=0.05):
    """Build IMUSamples from per-sample gyro (and optional accel) vectors."""
    n = len(gyros)
    if accels is None:
        accels = [_GRAV] * n
    return [
        IMUSample(accel=np.asarray(a, float), gyro=np.asarray(g, float),
                  timestamp=i * dt)
        for i, (g, a) in enumerate(zip(gyros, accels))
    ]


# ---------------------------------------------------------------------------
# estimate_gyro_bias
# ---------------------------------------------------------------------------

class TestEstimateGyroBias:
    def test_recovers_constant_bias(self):
        s = _samples([_BIAS] * 40)
        b = estimate_gyro_bias(s)
        np.testing.assert_allclose(b, _BIAS, atol=1e-9)

    def test_averages_out_zero_mean_noise(self):
        rng = np.random.default_rng(0)
        gyros = [_BIAS + rng.normal(0, 0.005, 3) for _ in range(200)]
        b = estimate_gyro_bias(_samples(gyros))
        np.testing.assert_allclose(b, _BIAS, atol=2e-3)

    def test_moving_start_returns_none(self):
        # Large angular rate from the first sample — no stationary run.
        assert estimate_gyro_bias(_samples([[0.5, 0.0, 0.0]] * 40)) is None

    def test_too_few_samples_returns_none(self):
        assert estimate_gyro_bias(_samples([_BIAS] * 5)) is None

    def test_shaky_accel_returns_none(self):
        # Gyro looks at-rest, but acceleration swings — platform moving.
        rng = np.random.default_rng(1)
        accels = [_GRAV + rng.normal(0, 3.0, 3) for _ in range(40)]
        assert estimate_gyro_bias(_samples([_BIAS] * 40, accels)) is None

    def test_uses_only_the_stationary_prefix(self):
        # 20 stationary samples, then the robot starts turning.
        gyros = [_BIAS] * 20 + [[0.4, 0.0, 0.0]] * 20
        b = estimate_gyro_bias(_samples(gyros))
        np.testing.assert_allclose(b, _BIAS, atol=1e-9)

    def test_short_stationary_prefix_rejected(self):
        # Only 6 stationary samples before motion — below min_samples.
        gyros = [_BIAS] * 6 + [[0.4, 0.0, 0.0]] * 30
        assert estimate_gyro_bias(_samples(gyros)) is None


# ---------------------------------------------------------------------------
# estimate_initial_state
# ---------------------------------------------------------------------------

class TestEstimateInitialState:
    def test_returns_state_with_bias(self):
        st = estimate_initial_state(_samples([_BIAS] * 40))
        assert isinstance(st, VIONominalState)
        np.testing.assert_allclose(st.b_g, _BIAS, atol=1e-9)

    def test_other_fields_at_rest(self):
        st = estimate_initial_state(_samples([_BIAS] * 40))
        np.testing.assert_allclose(st.p_w_i, np.zeros(3))
        np.testing.assert_allclose(st.v_w_i, np.zeros(3))
        np.testing.assert_allclose(st.R_w_i, np.eye(3))
        np.testing.assert_allclose(st.b_a, np.zeros(3))

    def test_none_when_not_stationary(self):
        assert estimate_initial_state(_samples([[0.5, 0.0, 0.0]] * 40)) is None


# ---------------------------------------------------------------------------
# is_stationary
# ---------------------------------------------------------------------------

class TestIsStationary:
    def test_at_rest_window_is_stationary(self):
        # Small gyro bias, accel = pure gravity reaction.
        assert is_stationary(_samples([_BIAS] * 5)) is True

    def test_rotating_window_is_not_stationary(self):
        assert is_stationary(_samples([[0.5, 0.0, 0.0]] * 5)) is False

    def test_linear_acceleration_is_not_stationary(self):
        # Gyro looks at rest, but the accel magnitude is well off |g|.
        accels = [[5.0, 0.0, 9.81]] * 5          # ‖a‖ ≈ 11 m/s²
        assert is_stationary(_samples([_BIAS] * 5, accels)) is False

    def test_empty_window_is_not_stationary(self):
        assert is_stationary([]) is False

    def test_single_moving_sample_breaks_stationarity(self):
        gyros = [_BIAS, _BIAS, [0.5, 0.0, 0.0], _BIAS]
        assert is_stationary(_samples(gyros)) is False
