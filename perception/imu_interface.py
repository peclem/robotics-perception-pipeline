"""
IMU sensor interface.

Architecture
------------
IMUInterface (ABC)
    └── NullIMU       — no-op, used when no IMU is configured
    └── SyntheticIMU  — generates plausible samples for offline testing
                        + pre-integration math validation

A real backend (Bosch BNO055, Bosch BMI088, Microstrain CV7-INS,
phone IMU, Meta-glasses IMU) would slot in as a third concrete
implementation. The interface is sample-pull rather than callback-push
to fit cleanly into the existing frame-loop architecture.

Coordinate convention
---------------------
Each IMUSample carries accelerometer and gyroscope readings in the
IMU body frame. By convention (matching most automotive / robotics
IMUs and the ROS REP-145 conventions):

    accel : (3,) m/s² in body frame. Includes gravity reaction.
            On a stationary, level IMU: a ≈ [0, 0, +9.81] m/s²
            (gravity pushes the proof mass "up" relative to the
            housing, which the sensor reports as +z acceleration).
    gyro  : (3,) rad/s in body frame. Right-hand-rule about each axis.
    timestamp : monotonic seconds.

The static extrinsic `base_link → imu_link` (configured via
coordinate_frames.static_extrinsics) defines the rigid mounting.

Reference
---------
ROS REP-145 — IMU data conventions
Forster et al. (2017) — On-Manifold Pre-integration for Real-Time
                        Visual-Inertial Odometry. arXiv:1512.02363
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class IMUSample:
    """
    One inertial measurement at a single timestamp.

    Attributes
    ----------
    accel : (3,) float64 — proper acceleration in body frame, m/s²
                           (includes gravity reaction).
    gyro  : (3,) float64 — angular velocity in body frame, rad/s.
    timestamp : monotonic seconds.
    """
    accel:     np.ndarray
    gyro:      np.ndarray
    timestamp: float

    def __post_init__(self) -> None:
        self.accel = np.asarray(self.accel, dtype=np.float64)
        self.gyro  = np.asarray(self.gyro,  dtype=np.float64)
        assert self.accel.shape == (3,), \
            f"accel must be (3,), got {self.accel.shape}"
        assert self.gyro.shape == (3,), \
            f"gyro must be (3,), got {self.gyro.shape}"

    def __repr__(self) -> str:
        return (
            f"IMUSample(a=[{self.accel[0]:+.2f},{self.accel[1]:+.2f},"
            f"{self.accel[2]:+.2f}]m/s² "
            f"ω=[{self.gyro[0]:+.3f},{self.gyro[1]:+.3f},"
            f"{self.gyro[2]:+.3f}]rad/s t={self.timestamp:.3f})"
        )


class IMUInterface(ABC):
    """
    Abstract base for IMU backends.

    `get_samples_since(t)` is sample-pull: returns all samples with
    timestamps strictly greater than `t`. The pipeline calls this
    once per camera frame and passes the batch to the pre-integrator.
    """

    @abstractmethod
    def open(self) -> None:
        """Initialise the underlying sensor."""
        ...

    @abstractmethod
    def release(self) -> None:
        """Release sensor resources."""
        ...

    @abstractmethod
    def get_samples_since(self, t: float) -> List[IMUSample]:
        """
        Return all buffered samples with timestamp > `t`, in order.
        Implementations may drop very old samples to bound memory.
        """
        ...

    @property
    @abstractmethod
    def rate_hz(self) -> float:
        """Nominal sample rate. Used for budget / sanity checks."""
        ...

    def __enter__(self) -> "IMUInterface":
        self.open()
        return self

    def __exit__(self, *_) -> None:
        self.release()


class NullIMU(IMUInterface):
    """
    No-op IMU. Always returns an empty sample list.

    Used when imu.type='null' or when an IMU is configured but
    unavailable. Downstream consumers must tolerate the empty case
    (the IMUPreintegrator's empty-input branch returns an identity
    PreintegratedMeasurement).
    """

    def open(self) -> None:
        pass

    def release(self) -> None:
        pass

    def get_samples_since(self, t: float) -> List[IMUSample]:
        return []

    @property
    def rate_hz(self) -> float:
        return 0.0

    def __repr__(self) -> str:
        return "NullIMU()"


# Synthetic motion presets — chosen to give the pre-integration tests
# clean ground-truth references.
_MOTION_PRESETS = {
    "stationary":            (np.zeros(3), np.zeros(3)),
    # Gravity vector only (level IMU sitting on a table)
    "stationary_with_gravity": (np.array([0.0, 0.0, 9.81]), np.zeros(3)),
    # Constant rotation about z
    "spin_z":                (np.zeros(3), np.array([0.0, 0.0, 0.5])),
    # Constant linear acceleration along x
    "accel_x":               (np.array([1.0, 0.0, 0.0]), np.zeros(3)),
}


class SyntheticIMU(IMUInterface):
    """
    Generates IMU samples following a configurable motion preset.

    Useful for offline pre-integration validation, integration tests,
    and as a stand-in when no hardware is present. Samples are
    produced lazily on get_samples_since() — the synthetic source
    "catches up" to the requested time at the configured rate.

    Parameters
    ----------
    motion : one of _MOTION_PRESETS keys, OR a tuple (accel, gyro) of
             constant body-frame vectors.
    rate_hz : sample rate. 200 Hz is typical for consumer IMUs;
              MEMS chips run 1 kHz+; phone IMUs are 100-500 Hz.
    noise_std_accel / noise_std_gyro : optional Gaussian noise added
              per sample. Defaults to zero (deterministic) for
              reproducible tests; non-zero values let users stress
              the downstream estimator.
    seed    : RNG seed for reproducibility.
    """

    def __init__(
        self,
        motion:           str | tuple        = "stationary",
        rate_hz:          float              = 200.0,
        noise_std_accel:  float              = 0.0,
        noise_std_gyro:   float              = 0.0,
        seed:             int                = 0,
    ) -> None:
        if isinstance(motion, str):
            if motion not in _MOTION_PRESETS:
                raise ValueError(
                    f"SyntheticIMU: unknown motion preset {motion!r}. "
                    f"Choose from {sorted(_MOTION_PRESETS)} or pass an "
                    f"(accel, gyro) tuple of constants."
                )
            self._accel_const, self._gyro_const = _MOTION_PRESETS[motion]
        else:
            a, g = motion
            self._accel_const = np.asarray(a, dtype=np.float64)
            self._gyro_const  = np.asarray(g, dtype=np.float64)

        self._rate_hz = float(rate_hz)
        self._dt = 1.0 / self._rate_hz
        self._noise_a = float(noise_std_accel)
        self._noise_g = float(noise_std_gyro)
        self._rng = np.random.default_rng(seed)

        self._is_open = False
        self._next_sample_t: Optional[float] = None

    def open(self) -> None:
        self._is_open = True
        self._next_sample_t = time.monotonic()

    def release(self) -> None:
        self._is_open = False

    def get_samples_since(self, t: float) -> List[IMUSample]:
        if not self._is_open:
            return []
        # Sit at the later of (caller's cutoff, our cursor) so we
        # don't replay old samples on time rewind.
        cursor = max(self._next_sample_t or 0.0, t + self._dt)
        now = time.monotonic()
        samples: List[IMUSample] = []
        while cursor <= now:
            accel = self._accel_const.copy()
            gyro  = self._gyro_const.copy()
            if self._noise_a > 0:
                accel = accel + self._rng.normal(0, self._noise_a, size=3)
            if self._noise_g > 0:
                gyro  = gyro  + self._rng.normal(0, self._noise_g, size=3)
            samples.append(IMUSample(accel=accel, gyro=gyro, timestamp=cursor))
            cursor += self._dt
        self._next_sample_t = cursor
        return samples

    def generate_batch(
        self,
        n_samples: int,
        start_t:   float = 0.0,
    ) -> List[IMUSample]:
        """
        Deterministic batch generator for unit tests — bypasses the
        wall-clock cursor and produces exactly `n_samples` samples
        starting at `start_t` at the configured rate.
        """
        samples: List[IMUSample] = []
        t = start_t
        for _ in range(n_samples):
            accel = self._accel_const.copy()
            gyro  = self._gyro_const.copy()
            if self._noise_a > 0:
                accel = accel + self._rng.normal(0, self._noise_a, size=3)
            if self._noise_g > 0:
                gyro  = gyro  + self._rng.normal(0, self._noise_g, size=3)
            samples.append(IMUSample(accel=accel, gyro=gyro, timestamp=t))
            t += self._dt
        return samples

    @property
    def rate_hz(self) -> float:
        return self._rate_hz

    def __repr__(self) -> str:
        return (
            f"SyntheticIMU(a={self._accel_const.tolist()} "
            f"ω={self._gyro_const.tolist()} {self._rate_hz:.0f}Hz)"
        )
