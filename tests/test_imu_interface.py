"""
Unit tests for the IMU sample DTO and synthetic backend.

TestIMUSample          : DTO shape + repr
TestNullIMU            : always returns empty
TestSyntheticIMU       : motion presets, rate, noise, batch generator
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from perception.imu_interface import (
    IMUInterface, IMUSample, NullIMU, SyntheticIMU,
)


class TestIMUSample:

    def test_construction_and_shape(self):
        s = IMUSample(accel=[0.0, 0.0, 9.81], gyro=[0.0, 0.0, 0.0],
                      timestamp=1.0)
        assert s.accel.shape == (3,)
        assert s.gyro.shape == (3,)
        assert s.timestamp == 1.0

    def test_rejects_wrong_dimensions(self):
        with pytest.raises(AssertionError):
            IMUSample(accel=[0.0, 0.0], gyro=[0.0, 0.0, 0.0], timestamp=0.0)
        with pytest.raises(AssertionError):
            IMUSample(accel=[0.0, 0.0, 0.0], gyro=[0.0, 0.0],
                      timestamp=0.0)


class TestNullIMU:

    def test_empty_samples(self):
        imu = NullIMU()
        imu.open()
        assert imu.get_samples_since(0.0) == []
        assert imu.rate_hz == 0.0
        imu.release()

    def test_context_manager(self):
        with NullIMU() as imu:
            assert imu.get_samples_since(0.0) == []


class TestSyntheticIMU:

    def test_motion_preset_stationary(self):
        imu = SyntheticIMU(motion="stationary", rate_hz=100.0)
        samples = imu.generate_batch(n_samples=5, start_t=0.0)
        assert len(samples) == 5
        for s in samples:
            np.testing.assert_allclose(s.accel, np.zeros(3))
            np.testing.assert_allclose(s.gyro, np.zeros(3))

    def test_motion_preset_stationary_with_gravity(self):
        imu = SyntheticIMU(motion="stationary_with_gravity", rate_hz=100.0)
        samples = imu.generate_batch(n_samples=3, start_t=0.0)
        for s in samples:
            np.testing.assert_allclose(s.accel, [0.0, 0.0, 9.81])
            np.testing.assert_allclose(s.gyro, np.zeros(3))

    def test_motion_preset_spin_z(self):
        imu = SyntheticIMU(motion="spin_z", rate_hz=100.0)
        samples = imu.generate_batch(n_samples=3, start_t=0.0)
        for s in samples:
            np.testing.assert_allclose(s.gyro, [0.0, 0.0, 0.5])

    def test_timestamps_advance_at_rate(self):
        imu = SyntheticIMU(motion="stationary", rate_hz=200.0)
        samples = imu.generate_batch(n_samples=4, start_t=0.0)
        # 200 Hz → dt = 5 ms
        expected = [0.0, 0.005, 0.010, 0.015]
        for s, t in zip(samples, expected):
            assert s.timestamp == pytest.approx(t, abs=1e-9)

    def test_custom_motion_tuple(self):
        imu = SyntheticIMU(motion=([1.0, 2.0, 3.0], [0.1, 0.2, 0.3]),
                           rate_hz=50.0)
        samples = imu.generate_batch(n_samples=2)
        np.testing.assert_allclose(samples[0].accel, [1.0, 2.0, 3.0])
        np.testing.assert_allclose(samples[0].gyro, [0.1, 0.2, 0.3])

    def test_unknown_preset_rejected(self):
        with pytest.raises(ValueError, match="unknown motion preset"):
            SyntheticIMU(motion="rocket_launch")

    def test_noise_perturbs_samples(self):
        imu = SyntheticIMU(motion="stationary", rate_hz=100.0,
                           noise_std_accel=0.1, noise_std_gyro=0.01, seed=42)
        samples = imu.generate_batch(n_samples=20)
        accels = np.stack([s.accel for s in samples])
        gyros  = np.stack([s.gyro  for s in samples])
        # With non-zero noise, samples deviate from zero.
        assert (np.abs(accels) > 0).any()
        assert (np.abs(gyros)  > 0).any()
        # Noise is bounded: a few sigma from the mean.
        assert np.all(np.abs(accels) < 1.0)
        assert np.all(np.abs(gyros)  < 0.1)

    def test_get_samples_since_advances_cursor(self):
        imu = SyntheticIMU(motion="stationary", rate_hz=1000.0)
        imu.open()
        time.sleep(0.01)   # let cursor accumulate >10 samples
        s1 = imu.get_samples_since(t=0.0)
        # Second pull should not duplicate.
        s2 = imu.get_samples_since(t=s1[-1].timestamp if s1 else 0.0)
        ids = {s.timestamp for s in s1}
        assert not any(s.timestamp in ids for s in s2), \
            "consecutive pulls should not return overlapping samples"

    def test_rate_property(self):
        imu = SyntheticIMU(rate_hz=400.0)
        assert imu.rate_hz == 400.0
