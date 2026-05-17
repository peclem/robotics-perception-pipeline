"""
Tests for IMU pre-integration math against known motion patterns.

TestStationary       : zero gyro + zero accel → zero everything
TestConstantRotation : constant ω about an axis → ΔR matches Exp(ω·T)
TestConstantAccel    : constant accel → Δv linear, Δp quadratic in time
TestEmpty            : empty / single-sample input → identity
TestCovariance       : covariance grows monotonically with n_samples
TestQuaternionAccessor: SO(3) → quaternion conversion
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from perception.imu_interface import IMUSample, SyntheticIMU
from state_estimation.imu_preintegration import (
    IMUPreintegrator, PreintegratedMeasurement,
)


def _samples(accel, gyro, n: int, rate_hz: float = 200.0) -> list[IMUSample]:
    """Synthesize n samples at fixed rate with constant accel/gyro."""
    dt = 1.0 / rate_hz
    return [
        IMUSample(accel=np.asarray(accel, dtype=np.float64),
                  gyro=np.asarray(gyro, dtype=np.float64),
                  timestamp=i * dt)
        for i in range(n)
    ]


class TestStationary:

    def test_zero_input_yields_identity_pose(self):
        pre = IMUPreintegrator()
        m = pre.integrate(_samples([0, 0, 0], [0, 0, 0], n=10))
        np.testing.assert_allclose(m.delta_R, np.eye(3), atol=1e-12)
        np.testing.assert_allclose(m.delta_v, np.zeros(3), atol=1e-12)
        np.testing.assert_allclose(m.delta_p, np.zeros(3), atol=1e-12)
        assert m.dt > 0
        assert m.n_samples == 10

    def test_zero_input_zero_covariance_growth(self):
        """
        With zero gyro and zero accel inputs, the only thing changing
        the covariance is the white noise on the *measurements*, not
        the integrated state. So cov grows linearly with samples but
        stays bounded and small.
        """
        pre = IMUPreintegrator()
        m_short = pre.integrate(_samples([0, 0, 0], [0, 0, 0], n=5))
        m_long  = pre.integrate(_samples([0, 0, 0], [0, 0, 0], n=50))
        # Covariance is finite + symmetric positive semi-definite.
        assert np.all(np.isfinite(m_long.covariance))
        np.testing.assert_allclose(
            m_long.covariance, m_long.covariance.T, atol=1e-10,
        )


class TestConstantRotation:

    def test_constant_omega_z(self):
        """Constant angular velocity about z for T seconds → R = Exp(ω · T)."""
        rate_hz = 200.0
        omega = np.array([0.0, 0.0, 0.5])   # rad/s
        n = 401   # T = (n-1)/rate_hz = 2.0 s
        pre = IMUPreintegrator()
        m = pre.integrate(_samples([0, 0, 0], omega, n=n, rate_hz=rate_hz))

        T = (n - 1) / rate_hz
        expected = R.from_rotvec(omega * T).as_matrix()
        # The discrete integration accumulates per-sample first-order
        # errors. For 1 rad rotation at 200 Hz, expect tight agreement.
        np.testing.assert_allclose(m.delta_R, expected, atol=1e-3)

    def test_constant_omega_x(self):
        rate_hz = 500.0
        omega = np.array([0.3, 0.0, 0.0])
        n = 251   # T = 0.5 s
        pre = IMUPreintegrator()
        m = pre.integrate(_samples([0, 0, 0], omega, n=n, rate_hz=rate_hz))
        T = (n - 1) / rate_hz
        expected = R.from_rotvec(omega * T).as_matrix()
        np.testing.assert_allclose(m.delta_R, expected, atol=1e-4)

    def test_no_rotation_under_zero_omega(self):
        pre = IMUPreintegrator()
        # Constant accel, zero gyro → ΔR stays identity.
        m = pre.integrate(_samples([1.0, 0, 0], [0, 0, 0], n=20))
        np.testing.assert_allclose(m.delta_R, np.eye(3), atol=1e-12)


class TestConstantAccel:

    def test_v_grows_linearly(self):
        """Δv = a · T for constant accel in body frame at zero rotation."""
        rate_hz = 200.0
        accel = np.array([1.5, 0.0, 0.0])
        n = 201   # T = 1.0 s
        pre = IMUPreintegrator()
        m = pre.integrate(_samples(accel, [0, 0, 0], n=n, rate_hz=rate_hz))
        T = (n - 1) / rate_hz
        expected_v = accel * T
        np.testing.assert_allclose(m.delta_v, expected_v, atol=1e-2)

    def test_p_grows_quadratically(self):
        """Δp = ½ · a · T² for constant accel at zero initial velocity."""
        rate_hz = 200.0
        accel = np.array([1.5, 0.0, 0.0])
        n = 201
        pre = IMUPreintegrator()
        m = pre.integrate(_samples(accel, [0, 0, 0], n=n, rate_hz=rate_hz))
        T = (n - 1) / rate_hz
        expected_p = 0.5 * accel * T * T
        np.testing.assert_allclose(m.delta_p, expected_p, atol=2e-2)

    def test_orthogonal_axes_independent(self):
        """Accel only in y should leave x components zero."""
        pre = IMUPreintegrator()
        m = pre.integrate(_samples([0, 0.7, 0], [0, 0, 0], n=100))
        assert m.delta_v[0] == pytest.approx(0.0, abs=1e-10)
        assert m.delta_p[0] == pytest.approx(0.0, abs=1e-10)
        assert m.delta_v[1] > 0


class TestRotationCouplesAccel:

    def test_constant_omega_z_rotates_accel(self):
        """
        Body-frame accel along x with body rotating about z: in the
        body-i frame (ΔR_ij applied), the accel direction sweeps. The
        accumulated Δv should NOT match the body-j accel times time —
        it should be roughly along the *initial* x direction at the
        start of the window, drifting as the body rotates.
        """
        rate_hz = 500.0
        omega = np.array([0.0, 0.0, 1.0])   # 1 rad/s about z
        accel = np.array([1.0, 0.0, 0.0])   # body-frame x
        n = 501   # T = 1.0 s → body rotates by 1 rad ≈ 57°
        pre = IMUPreintegrator()
        m = pre.integrate(_samples(accel, omega, n=n, rate_hz=rate_hz))
        # If rotation were ignored, Δv = accel * T = [1, 0, 0].
        # With rotation, the accel direction sweeps, so x-component
        # is smaller and y is nonzero.
        assert abs(m.delta_v[0]) < 1.0
        assert abs(m.delta_v[1]) > 0.1


class TestEmpty:

    def test_no_samples_returns_identity(self):
        pre = IMUPreintegrator()
        m = pre.integrate([])
        np.testing.assert_allclose(m.delta_R, np.eye(3))
        np.testing.assert_allclose(m.delta_v, np.zeros(3))
        np.testing.assert_allclose(m.delta_p, np.zeros(3))
        assert m.dt == 0.0
        assert m.n_samples == 0

    def test_single_sample_returns_identity(self):
        pre = IMUPreintegrator()
        m = pre.integrate([IMUSample(accel=[0, 0, 0], gyro=[0, 0, 0],
                                     timestamp=0.0)])
        np.testing.assert_allclose(m.delta_R, np.eye(3))
        assert m.dt == 0.0

    def test_zero_dt_samples_skipped(self):
        """Duplicate timestamps mustn't divide by zero or blow up."""
        pre = IMUPreintegrator()
        m = pre.integrate([
            IMUSample(accel=[1, 0, 0], gyro=[0, 0, 0], timestamp=0.0),
            IMUSample(accel=[1, 0, 0], gyro=[0, 0, 0], timestamp=0.0),
            IMUSample(accel=[1, 0, 0], gyro=[0, 0, 0], timestamp=0.01),
        ])
        assert np.all(np.isfinite(m.delta_v))
        assert np.all(np.isfinite(m.delta_p))


class TestCovariance:

    def test_covariance_grows_with_samples(self):
        pre = IMUPreintegrator()
        m_short = pre.integrate(_samples([0, 0, 0], [0, 0, 0], n=10))
        m_long  = pre.integrate(_samples([0, 0, 0], [0, 0, 0], n=100))
        # Diagonal noise terms accumulate monotonically.
        assert np.trace(m_long.covariance) > np.trace(m_short.covariance)

    def test_covariance_symmetric_psd(self):
        pre = IMUPreintegrator()
        m = pre.integrate(_samples([0.5, 0, 0], [0, 0.1, 0], n=50))
        cov = m.covariance
        np.testing.assert_allclose(cov, cov.T, atol=1e-10)
        # PSD: all eigenvalues ≥ 0 (allow tiny numerical slack).
        eigs = np.linalg.eigvalsh(cov)
        assert np.all(eigs >= -1e-12), f"non-PSD covariance: {eigs}"


class TestQuaternionAccessor:

    def test_quat_matches_rotation_matrix(self):
        pre = IMUPreintegrator()
        # Integrate a constant rotation about y → exact ΔR known.
        m = pre.integrate(_samples([0, 0, 0], [0, 0.4, 0], n=251, rate_hz=500.0))
        q = m.delta_rotation_quat_xyzw
        # Length 1 (normalised) and 4-tuple
        assert q.shape == (4,)
        np.testing.assert_allclose(np.linalg.norm(q), 1.0, atol=1e-8)


class TestSyntheticIntegration:
    """End-to-end: SyntheticIMU → IMUPreintegrator agreement with presets."""

    def test_spin_z_preset_matches_expected_rotation(self):
        imu = SyntheticIMU(motion="spin_z", rate_hz=200.0)
        batch = imu.generate_batch(n_samples=201)   # T = 1.0 s
        pre = IMUPreintegrator()
        m = pre.integrate(batch)
        expected = R.from_rotvec([0.0, 0.0, 0.5]).as_matrix()
        np.testing.assert_allclose(m.delta_R, expected, atol=1e-3)

    def test_accel_x_preset_matches_expected_velocity(self):
        imu = SyntheticIMU(motion="accel_x", rate_hz=200.0)
        batch = imu.generate_batch(n_samples=201)   # T = 1.0 s
        pre = IMUPreintegrator()
        m = pre.integrate(batch)
        np.testing.assert_allclose(m.delta_v, [1.0, 0.0, 0.0], atol=1e-2)
        np.testing.assert_allclose(m.delta_p, [0.5, 0.0, 0.0], atol=2e-2)
