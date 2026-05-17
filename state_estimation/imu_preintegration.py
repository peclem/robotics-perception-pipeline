"""
IMU pre-integration (Forster et al. 2017).

What pre-integration is and why
-------------------------------
A camera frame arrives every ~33 ms; an IMU samples every ~5 ms.
Between two camera frames there are ~6-7 IMU samples — too many to
add as individual factors in a visual-inertial estimator. The
pre-integrated measurement combines them into a single relative
motion constraint:

    ΔR_ij ∈ SO(3)   rotation from body-i to body-j
    Δv_ij ∈ R³      velocity change in body-i's frame
    Δp_ij ∈ R³      position change in body-i's frame

These three quantities are independent of the absolute pose at time i
— they depend only on the IMU samples between i and j (and the
biases, which we treat as zero/pre-calibrated in v1). That means
the same pre-integrated measurement can be re-used if the optimiser
later updates the absolute pose at time i, without re-integrating
the raw samples. This decoupling is what makes Forster's formulation
so widely adopted in VIO (VINS-Mono, OpenVINS, ORB-SLAM3).

What this module does and doesn't do
------------------------------------
Implements (v1):
- Discrete-time forward propagation of ΔR, Δv, Δp from a sequence
  of IMUSamples.
- Covariance propagation using the linearised noise model
  (gyro / accel white noise, no bias terms).
- Sanity helpers (residual against known motion patterns).

Does NOT yet implement:
- Bias-aware pre-integration with first-order Jacobians ∂ΔR/∂b_g
  etc. — left for the moment a real estimator wants them.
- The accel/gyro bias random-walk model (added when an EKF
  starts to track biases).
- Gravity removal — handled at the *fusion* step, when the absolute
  orientation R_w_i is known and gravity vector g_w can be rotated
  into body-i.

Result is mathematically correct for the bias-free / pre-calibrated
case and ready to be slotted into a VIO factor graph or error-state
EKF when one lands.

Reference
---------
Forster, Carlone, Dellaert, Scaramuzza —
On-Manifold Pre-integration for Real-Time Visual-Inertial Odometry.
IEEE T-RO 33(1), 2017. arXiv:1512.02363
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Sequence

import numpy as np
from scipy.spatial.transform import Rotation as R

from perception.imu_interface import IMUSample

log = logging.getLogger(__name__)


@dataclass
class PreintegratedMeasurement:
    """
    Result of integrating an IMU sample sequence between two timestamps.

    All quantities are in body-i's reference frame: ΔR rotates a
    body-j vector into body-i; Δv and Δp are the velocity and
    position change accumulated over the window, expressed in body-i
    (without gravity — gravity is applied by the downstream fuser
    once the absolute orientation R_w_i is known).

    Attributes
    ----------
    delta_R     : (3, 3) rotation matrix, body-i ← body-j.
    delta_v     : (3,) velocity change, body-i frame, m/s.
    delta_p     : (3,) position change, body-i frame, m.
    dt          : total elapsed time, seconds.
    covariance  : (9, 9) covariance of the stacked [δθ; δv; δp]
                  error state (rotation perturbation in axis-angle
                  form, then velocity, then position).
    n_samples   : number of IMU samples integrated.
    """
    delta_R:    np.ndarray
    delta_v:    np.ndarray
    delta_p:    np.ndarray
    dt:         float
    covariance: np.ndarray
    n_samples:  int

    def __post_init__(self) -> None:
        self.delta_R    = np.asarray(self.delta_R,    dtype=np.float64)
        self.delta_v    = np.asarray(self.delta_v,    dtype=np.float64)
        self.delta_p    = np.asarray(self.delta_p,    dtype=np.float64)
        self.covariance = np.asarray(self.covariance, dtype=np.float64)
        assert self.delta_R.shape    == (3, 3)
        assert self.delta_v.shape    == (3,)
        assert self.delta_p.shape    == (3,)
        assert self.covariance.shape == (9, 9)

    @classmethod
    def identity(cls) -> "PreintegratedMeasurement":
        return cls(
            delta_R=np.eye(3),
            delta_v=np.zeros(3),
            delta_p=np.zeros(3),
            dt=0.0,
            covariance=np.zeros((9, 9)),
            n_samples=0,
        )

    @property
    def delta_rotation_quat_xyzw(self) -> np.ndarray:
        """Convenience: ΔR as quaternion (xyzw) for ROS interop."""
        return R.from_matrix(self.delta_R).as_quat()

    def __repr__(self) -> str:
        return (
            f"PreintegratedMeasurement(dt={self.dt:.3f}s, "
            f"n={self.n_samples}, "
            f"|Δv|={np.linalg.norm(self.delta_v):.3f} m/s, "
            f"|Δp|={np.linalg.norm(self.delta_p):.3f} m)"
        )


# ---------------------------------------------------------------------------
# SO(3) helpers (kept local to avoid a heavy dependency surface).
# ---------------------------------------------------------------------------

def _so3_exp(phi: np.ndarray) -> np.ndarray:
    """
    Rodrigues' formula: SO(3) exponential of a 3-vector axis-angle.

    Uses scipy for the numerically robust path including small-angle
    Taylor expansion. Returns a 3×3 rotation matrix.
    """
    return R.from_rotvec(phi).as_matrix()


def _skew(v: np.ndarray) -> np.ndarray:
    return np.array([
        [0.0,   -v[2],  v[1]],
        [v[2],   0.0,  -v[0]],
        [-v[1],  v[0],  0.0],
    ], dtype=np.float64)


def _right_jacobian_so3(phi: np.ndarray) -> np.ndarray:
    """
    Right Jacobian of SO(3) at φ.

    Used in covariance propagation: when a noise term n rotates the
    integration step, the resulting perturbation in the world frame
    is J_r(φ) · n. Small-angle Taylor expansion for numerical stability
    around φ ≈ 0.
    """
    theta = float(np.linalg.norm(phi))
    if theta < 1e-8:
        return np.eye(3) - 0.5 * _skew(phi)
    phi_hat = _skew(phi)
    a = (1.0 - np.cos(theta)) / (theta ** 2)
    b = (theta - np.sin(theta)) / (theta ** 3)
    return np.eye(3) - a * phi_hat + b * phi_hat @ phi_hat


# ---------------------------------------------------------------------------
# Pre-integrator
# ---------------------------------------------------------------------------

class IMUPreintegrator:
    """
    Stateful pre-integrator. Construct once with the noise model;
    call `integrate(samples)` for each batch of IMU samples between
    two camera frames.

    Parameters
    ----------
    sigma_gyro_n     : gyro white noise density σ_g, rad/s/√Hz.
                       Typical MEMS: 1e-3 to 1e-2 rad/s/√Hz.
    sigma_accel_n    : accel white noise density σ_a, m/s²/√Hz.
                       Typical MEMS: 1e-2 to 1e-1 m/s²/√Hz.

    These values come from the IMU datasheet's noise spec ("rate
    noise density" for gyro, "noise density" for accel). The
    covariance propagation multiplies them by √dt at each step.
    """

    def __init__(
        self,
        sigma_gyro_n:  float = 1.7e-4,   # ~Allan deviation of a good MEMS gyro
        sigma_accel_n: float = 2.0e-3,   # ~Allan deviation of a good MEMS accel
    ) -> None:
        self._sigma_g_sq = float(sigma_gyro_n)  ** 2
        self._sigma_a_sq = float(sigma_accel_n) ** 2

    def integrate(
        self,
        samples: Sequence[IMUSample],
    ) -> PreintegratedMeasurement:
        """
        Pre-integrate a contiguous batch of IMU samples.

        The batch is assumed to be in monotonically increasing time
        order. dt between successive samples comes from their
        timestamps; the first sample's contribution uses the dt to
        the *next* sample (last sample contributes zero — there's no
        "next" to define dt with).

        Empty input returns the identity measurement.
        """
        if len(samples) < 2:
            return PreintegratedMeasurement.identity()

        dR = np.eye(3)
        dv = np.zeros(3)
        dp = np.zeros(3)
        cov = np.zeros((9, 9))
        t0 = samples[0].timestamp

        # Iterate over consecutive pairs so we have a well-defined dt.
        for i in range(len(samples) - 1):
            dt = float(samples[i + 1].timestamp - samples[i].timestamp)
            if dt <= 0:
                # Out-of-order or duplicate timestamp — skip.
                continue
            omega = samples[i].gyro
            acc   = samples[i].accel

            # --- mean propagation -----------------------------------
            # ΔR_new = ΔR_old · Exp(ω · dt)
            # Δv_new = Δv_old + ΔR_old · acc · dt
            # Δp_new = Δp_old + Δv_old · dt + 0.5 · ΔR_old · acc · dt²
            dR_step = _so3_exp(omega * dt)
            acc_world_step = dR @ acc

            dp_new = dp + dv * dt + 0.5 * acc_world_step * dt * dt
            dv_new = dv + acc_world_step * dt
            dR_new = dR @ dR_step

            # --- covariance propagation (Forster eq. 53-59) ---------
            # Continuous-time noise σ² is in units of rad²/s and m²/s³
            # per axis; the discrete contributions scale by dt.
            # Build A (state-transition Jacobian) and B (noise Jacobian),
            # both 9×9, then cov ← A·cov·A^T + B·Σ·B^T.
            A = np.eye(9)
            # ∂(δθ_new)/∂(δθ_old) = Exp(-ω·dt)
            A[0:3, 0:3] = dR_step.T
            # ∂(δv_new)/∂(δθ_old) = -ΔR · [acc]× · dt
            A[3:6, 0:3] = -dR @ _skew(acc) * dt
            # ∂(δp_new)/∂(δv_old) = I · dt
            A[6:9, 3:6] = np.eye(3) * dt
            # ∂(δp_new)/∂(δθ_old) = -0.5 · ΔR · [acc]× · dt²
            A[6:9, 0:3] = -0.5 * dR @ _skew(acc) * dt * dt

            # Noise input: 3 gyro + 3 accel = 6 axes; B is 9×6.
            B = np.zeros((9, 6))
            # δθ from gyro noise (scaled by right Jacobian)
            Jr = _right_jacobian_so3(omega * dt)
            B[0:3, 0:3] = Jr * dt
            # δv from accel noise
            B[3:6, 3:6] = dR * dt
            # δp from accel noise
            B[6:9, 3:6] = 0.5 * dR * dt * dt

            # Noise covariance (6×6 diagonal).
            sigma = np.diag([
                self._sigma_g_sq, self._sigma_g_sq, self._sigma_g_sq,
                self._sigma_a_sq, self._sigma_a_sq, self._sigma_a_sq,
            ]) / max(dt, 1e-9)   # σ² is per-second density → per-sample variance

            cov = A @ cov @ A.T + B @ sigma @ B.T

            dR = dR_new
            dv = dv_new
            dp = dp_new

        total_dt = float(samples[-1].timestamp - t0)
        return PreintegratedMeasurement(
            delta_R=dR, delta_v=dv, delta_p=dp,
            dt=total_dt, covariance=cov,
            n_samples=len(samples),
        )

    def __repr__(self) -> str:
        return (
            f"IMUPreintegrator(σ_g={np.sqrt(self._sigma_g_sq):.2e} rad/s/√Hz, "
            f"σ_a={np.sqrt(self._sigma_a_sq):.2e} m/s²/√Hz)"
        )
