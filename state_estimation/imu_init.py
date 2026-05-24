"""
Stationary IMU initialisation for the visual-inertial EKF.

A gyroscope reports bias + true angular rate. When the platform is at
rest the true rate is zero, so the reading *is* the bias. Estimating it
from a stationary window at the start of a sequence and seeding the EKF
with it removes the orientation-drift transient the filter would
otherwise accumulate while its online bias estimate converges — and
orientation error integrates straight into unrecoverable position drift.

Scope (VIO v1.1)
----------------
This estimates the **gyro bias** only. It deliberately does NOT:
  - estimate accel bias — inseparable from the initial gravity
    direction without a known orientation;
  - gravity-align the initial orientation — that needs the EKF world
    and the visual estimator's world to share the aligned frame, i.e.
    frame-consistency surgery across both. Left for VIO v1.2.
Gyro bias, by contrast, is a frame-independent sensor-intrinsic
correction — safe to seed unconditionally.

References: Forster et al. (2017); standard VINS stationary init.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np

from perception.imu_interface import IMUSample
from state_estimation.visual_inertial_ekf import VIONominalState

log = logging.getLogger(__name__)


def _leading_stationary_run(
    samples: List[IMUSample],
    gyro_mag_thresh: float,
) -> int:
    """Length of the leading run of samples with |gyro| below threshold."""
    n = 0
    for s in samples:
        if float(np.linalg.norm(s.gyro)) >= gyro_mag_thresh:
            break
        n += 1
    return n


def is_stationary(
    samples: List[IMUSample],
    *,
    gyro_mag_thresh: float = 0.06,
    accel_dev_thresh: float = 0.6,
    gravity_mag: float = 9.81,
) -> bool:
    """
    True when every IMU sample in the window looks at-rest.

    At rest the angular rate is ~0 and the accelerometer reads only the
    gravity reaction, so its magnitude sits near |g|; a moving platform
    adds linear acceleration on top. Used to gate zero-velocity updates
    (ZUPT). An empty window is treated as not-stationary.

    Parameters
    ----------
    samples          : IMU samples for one inter-frame window.
    gyro_mag_thresh  : |gyro| (rad/s) ceiling for "at rest".
    accel_dev_thresh : allowed |‖accel‖ − |g|| deviation (m/s²).
    gravity_mag      : gravity magnitude (m/s²).
    """
    if not samples:
        return False
    for s in samples:
        if float(np.linalg.norm(s.gyro)) >= gyro_mag_thresh:
            return False
        if abs(float(np.linalg.norm(s.accel)) - gravity_mag) >= accel_dev_thresh:
            return False
    return True


def estimate_gyro_bias(
    samples: List[IMUSample],
    *,
    gyro_mag_thresh: float = 0.05,
    accel_std_thresh: float = 0.5,
    min_samples: int = 10,
) -> Optional[np.ndarray]:
    """
    Estimate the gyro bias from a leading stationary IMU window.

    Scans the start of `samples` for a run where the angular rate stays
    small; if that run is long enough and its acceleration is steady
    (low per-axis std — a moving platform shakes), returns the mean
    gyro over it as the bias estimate. Returns None when the start is
    not confidently stationary, so the caller can fall back to zero.

    Parameters
    ----------
    samples          : IMU samples in time order (e.g. the first ~2 s).
    gyro_mag_thresh  : |gyro| (rad/s) under which a sample counts as
                       at-rest. Loose enough to admit real bias.
    accel_std_thresh : max per-axis accel std (m/s²) over the run for
                       it to count as stationary rather than moving.
    min_samples      : shortest run accepted — too few and the mean
                       is noise, not bias.

    Returns
    -------
    (3,) float64 gyro bias estimate, or None.
    """
    if len(samples) < min_samples:
        return None

    run = _leading_stationary_run(samples, gyro_mag_thresh)
    if run < min_samples:
        return None

    window = samples[:run]
    accel = np.array([s.accel for s in window], dtype=np.float64)
    if np.any(accel.std(axis=0) > accel_std_thresh):
        return None  # acceleration is not steady — platform is moving

    gyro = np.array([s.gyro for s in window], dtype=np.float64)
    return gyro.mean(axis=0)


def estimate_initial_state(
    samples: List[IMUSample],
    *,
    gyro_mag_thresh: float = 0.05,
    accel_std_thresh: float = 0.5,
    min_samples: int = 10,
) -> Optional[VIONominalState]:
    """
    Build a VIONominalState seeded with the stationary gyro bias.

    Position, velocity, orientation and accel bias are left at rest
    (see module docstring for why orientation/accel are out of scope).
    Returns None when no trustworthy stationary window is found — the
    caller then uses VIONominalState.at_rest().
    """
    b_g = estimate_gyro_bias(
        samples,
        gyro_mag_thresh=gyro_mag_thresh,
        accel_std_thresh=accel_std_thresh,
        min_samples=min_samples,
    )
    if b_g is None:
        return None

    log.info(
        "VIO stationary init: gyro bias = [%+.5f %+.5f %+.5f] rad/s",
        b_g[0], b_g[1], b_g[2],
    )
    return VIONominalState(
        p_w_i=np.zeros(3), v_w_i=np.zeros(3), R_w_i=np.eye(3),
        b_g=b_g, b_a=np.zeros(3),
    )
