"""
Visual-inertial error-state EKF (the missing fuser between IMU
pre-integration and the visual pose estimator).

What this is
------------
A loosely-coupled error-state EKF that consumes:
  - PreintegratedMeasurement (ΔR, Δv, Δp + 9×9 cov + bias Jacobians)
    between two visual frames — produced by IMUPreintegrator.
  - CameraPose (R_w_c, t_w_c) at the visual frame — produced by any
    PoseEstimator (DPVO ships, OpenVINS or ORB-SLAM3 slot in here too).

It outputs:
  - The maintained nominal pose at the visual frame (R_w_i, p_w_i),
    plus body-frame velocity, gyro bias, accel bias.
  - The 16×16 error-state covariance.

The error-state factorisation is the standard one for VIO (Sola 2017,
Forster 2017): nominal state lives on the SE(3) × R⁹ manifold,
correction lives in R¹⁵ tangent space, with the rotation perturbation
applied via SO(3) right-multiplication: R_w_i ← R̂_w_i · Exp(δθ).

State ordering
--------------
nominal:  (p_w_i, v_w_i, R_w_i, b_g, b_a, scale)
error:    (δp,   δv,    δθ,   δb_g, δb_a, δs)    — 16-vector

Indices into the 16-vector / 16×16 covariance, used everywhere:
    P 0:3
    V 3:6
    THETA 6:9
    BG 9:12
    BA 12:15
    SCALE 15:16

Visual scale (state 16)
-----------------------
A monocular visual estimator (DPVO) produces a trajectory with an
arbitrary, drifting metric scale. `scale` is the visual→metric factor:
the visual position measurement is modelled as z = p_w_i / scale. The
IMU predict is metric, so whenever the platform accelerates the filter
*observes* scale from the disagreement between the metric prediction
and the scaled visual measurement — which a 15-D loosely-coupled VIO
cannot do (it consumes the visual pose as if already metric). Scale is
unobservable without acceleration excitation; its covariance simply
stays large then.

Coordinate-frame assumptions (v1)
---------------------------------
- World frame: gravity is (0, 0, -9.81) m/s² by default.
- IMU body frame == camera frame. A non-trivial camera-IMU extrinsic
  R_c_i / p_c_i can be folded in by transforming the visual measurement
  before update; that lives outside this class to keep it focused on
  the EKF math. The existing TransformTree is the natural place.

What is not in v1
-----------------
- Tight coupling (no landmark feature factors here — the visual pose is
  consumed pre-cooked as a 6-DOF measurement; tightly-coupled VIO
  belongs in OpenVINS / ORB-SLAM3 if absolute SOTA is the goal).
- Initialisation from rest with gravity alignment — caller passes the
  initial pose. A robotics-grade initialiser would solve for gravity
  + initial velocity from a few seconds of IMU-only data (Qin 2018,
  Mur-Artal 2017); deferred.
- Outlier rejection / chi-square gating on the visual update — easy
  add when we get a real-data trace with outliers.

References
----------
Sola, J. (2017) — Quaternion kinematics for the error-state Kalman
                  filter. arXiv:1711.02508
Forster et al. (2017) — On-Manifold Preintegration for Real-Time
                  Visual-Inertial Odometry. IEEE T-RO 33(1).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation as R

from perception.pose_estimator import CameraPose
from state_estimation.imu_preintegration import PreintegratedMeasurement

log = logging.getLogger(__name__)


# State block indices into the 16-vector / 16×16 covariance.
P_IDX = slice(0,  3)
V_IDX = slice(3,  6)
T_IDX = slice(6,  9)
BG_IDX = slice(9, 12)
BA_IDX = slice(12, 15)
S_IDX = slice(15, 16)

# Error-state dimension.
_N = 16


@dataclass
class VIONominalState:
    """
    Nominal (mean) state of the error-state EKF.

    The error correction lives in 16-D tangent space; this struct holds
    the full SE(3) × R⁹ nominal that the correction is applied onto.
    """
    p_w_i: np.ndarray   # (3,)  body position in world frame, metres
    v_w_i: np.ndarray   # (3,)  body velocity in world frame, m/s
    R_w_i: np.ndarray   # (3,3) rotation: world ← body
    b_g:   np.ndarray   # (3,)  gyro bias, rad/s
    b_a:   np.ndarray   # (3,)  accel bias, m/s²
    scale: float = 1.0  # visual→metric scale: z_visual = p_w_i / scale

    def __post_init__(self) -> None:
        self.p_w_i = np.asarray(self.p_w_i, dtype=np.float64).reshape(3)
        self.v_w_i = np.asarray(self.v_w_i, dtype=np.float64).reshape(3)
        self.R_w_i = np.asarray(self.R_w_i, dtype=np.float64).reshape(3, 3)
        self.b_g   = np.asarray(self.b_g,   dtype=np.float64).reshape(3)
        self.b_a   = np.asarray(self.b_a,   dtype=np.float64).reshape(3)
        self.scale = float(self.scale)

    @classmethod
    def at_rest(cls) -> "VIONominalState":
        """Identity orientation, zero position / velocity / bias, unit scale."""
        return cls(
            p_w_i=np.zeros(3), v_w_i=np.zeros(3), R_w_i=np.eye(3),
            b_g=np.zeros(3),   b_a=np.zeros(3),   scale=1.0,
        )

    def copy(self) -> "VIONominalState":
        return VIONominalState(
            p_w_i=self.p_w_i.copy(), v_w_i=self.v_w_i.copy(),
            R_w_i=self.R_w_i.copy(),
            b_g=self.b_g.copy(),     b_a=self.b_a.copy(),
            scale=self.scale,
        )


@dataclass
class VIOConfig:
    """
    Initial covariances + process / measurement noise model.

    Defaults are reasonable for a hand-held device with a good MEMS
    IMU + a decent visual estimator. Tune in config for your hardware.
    """
    # --- initial uncertainty (1σ, before first measurement) -----------
    init_position_std_m:      float = 0.10
    init_velocity_std_mps:    float = 0.10
    init_orientation_std_rad: float = 0.05    # ~3°
    init_bias_gyro_std:       float = 1e-3    # rad/s
    init_bias_accel_std:      float = 1e-2    # m/s²
    # --- bias random-walk process noise -------------------------------
    bias_gyro_random_walk:    float = 1e-5    # rad/s · √Hz
    bias_accel_random_walk:   float = 1e-4    # m/s² · √Hz
    # --- visual scale -------------------------------------------------
    # Initial 1σ on the visual→metric scale (nominal 1.0). Monocular
    # scale is unknown a priori, so this prior is deliberately loose.
    init_scale_std:           float = 0.5
    # Random walk that lets the estimated scale track DPVO's slow scale
    # drift along a trajectory rather than pinning one global value.
    scale_random_walk:        float = 1e-3    # per √s
    # --- visual measurement noise -------------------------------------
    visual_position_std_m:    float = 0.05
    visual_orientation_std_rad: float = 0.02  # ~1.1°
    # --- zero-velocity update -----------------------------------------
    # Measurement 1σ for a ZUPT (m/s). Small: when the platform is
    # detected at rest, the body velocity really is ~0.
    zupt_velocity_std:        float = 0.02
    # --- world ---------------------------------------------------------
    # Gravity expressed in the world frame. Default: Z-up world, so
    # gravity points along -Z. Override for ENU / NED / whatever your
    # downstream consumers use.
    gravity_w: tuple = (0.0, 0.0, -9.81)


class VisualInertialEKF:
    """
    Loosely-coupled visual-inertial error-state EKF.

    Lifecycle
    ---------
        ekf = VisualInertialEKF(VIOConfig(), initial_state, initial_timestamp)
        # ... each visual frame:
        ekf.predict(preintegrated_measurement)
        ekf.update(camera_pose)
        pose = ekf.current_pose(frame_idx=k)   # → CameraPose

    Conventions
    -----------
    - Covariance ordering: [p, v, θ, b_g, b_a] (each 3-D) + scale (1-D),
      total 16.
    - All rotations are stored as 3×3 matrices to match the IMU
      pre-integrator's convention.
    - Joseph form is used for the covariance update step
      (CLAUDE.md project rule).
    """

    def __init__(
        self,
        cfg:              VIOConfig,
        initial_state:    Optional[VIONominalState] = None,
        initial_timestamp: float = 0.0,
    ) -> None:
        self._cfg = cfg
        self._x   = (initial_state.copy() if initial_state is not None
                     else VIONominalState.at_rest())
        self._t   = float(initial_timestamp)

        self._P = np.zeros((_N, _N), dtype=np.float64)
        s = cfg
        self._P[P_IDX,  P_IDX]  = np.eye(3) * (s.init_position_std_m      ** 2)
        self._P[V_IDX,  V_IDX]  = np.eye(3) * (s.init_velocity_std_mps    ** 2)
        self._P[T_IDX,  T_IDX]  = np.eye(3) * (s.init_orientation_std_rad ** 2)
        self._P[BG_IDX, BG_IDX] = np.eye(3) * (s.init_bias_gyro_std       ** 2)
        self._P[BA_IDX, BA_IDX] = np.eye(3) * (s.init_bias_accel_std      ** 2)
        self._P[S_IDX,  S_IDX]  = np.eye(1) * (s.init_scale_std           ** 2)

        self._g_w = np.asarray(cfg.gravity_w, dtype=np.float64).reshape(3)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def state(self) -> VIONominalState:
        return self._x

    @property
    def covariance(self) -> np.ndarray:
        return self._P

    @property
    def timestamp(self) -> float:
        return self._t

    def current_pose(self, frame_idx: int = 0) -> CameraPose:
        """Snapshot the current body pose as a CameraPose (camera=body in v1)."""
        return CameraPose(
            R=self._x.R_w_i.copy(),
            t=self._x.p_w_i.copy(),
            timestamp=self._t,
            frame_idx=frame_idx,
            confidence=1.0,
            source="vio_ekf",
        )

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict(self, preint: PreintegratedMeasurement) -> None:
        """
        Propagate nominal state + covariance with one pre-integrated
        IMU measurement (between visual frames i and j).

        Bias correction
        ---------------
        If `preint.b_g_used` / `b_a_used` differ from the EKF's current
        bias estimate, the pre-integrated measurement is first updated
        in place via Forster's first-order correction (already
        implemented on PreintegratedMeasurement). This is the cheap path
        — no raw-sample re-integration.
        """
        # Apply current bias estimate to the pre-integrated measurement.
        if (not np.allclose(preint.b_g_used, self._x.b_g) or
                not np.allclose(preint.b_a_used, self._x.b_a)):
            preint = preint.correct_for_bias(self._x.b_g, self._x.b_a)

        dt = float(preint.dt)
        if dt <= 0:
            # No IMU samples in window — nothing to integrate.
            return

        R_w_i = self._x.R_w_i
        g     = self._g_w

        # ---- nominal state propagation -----------------------------------
        # Standard preintegration roll-up at the fusion step:
        #   p_j = p_i + v_i Δt + 0.5 g Δt² + R_w_i · Δp_ij
        #   v_j = v_i + g Δt   + R_w_i · Δv_ij
        #   R_j = R_w_i · ΔR_ij
        p_new = (self._x.p_w_i + self._x.v_w_i * dt
                 + 0.5 * g * dt * dt + R_w_i @ preint.delta_p)
        v_new = self._x.v_w_i + g * dt + R_w_i @ preint.delta_v
        R_new = R_w_i @ preint.delta_R

        # ---- error-state transition F (16×16) ----------------------------
        # Sola 2017 eq. 270, written in the right-perturbation
        # convention R = R̂·Exp(δθ). The scale row/column stay identity:
        # an IMU pre-integration carries no scale information.
        F = np.eye(_N, dtype=np.float64)
        F[P_IDX, V_IDX] = np.eye(3) * dt
        F[P_IDX, T_IDX] = -R_w_i @ _skew(preint.delta_p)
        F[V_IDX, T_IDX] = -R_w_i @ _skew(preint.delta_v)
        F[T_IDX, T_IDX] = preint.delta_R.T
        # Bias couplings via pre-integration's bias Jacobians.
        F[P_IDX,  BG_IDX] = R_w_i @ preint.J_p_g
        F[P_IDX,  BA_IDX] = R_w_i @ preint.J_p_a
        F[V_IDX,  BG_IDX] = R_w_i @ preint.J_v_g
        F[V_IDX,  BA_IDX] = R_w_i @ preint.J_v_a
        F[T_IDX,  BG_IDX] = preint.J_R_g
        # Biases evolve as random walks → identity on themselves
        # (already set by np.eye init).

        # ---- process noise Q (16×16) -------------------------------------
        # Two sources:
        #   1. Pre-integration measurement covariance — 9×9 in body-i frame,
        #      order [δθ, δv, δp]. Rotate position/velocity blocks into the
        #      world frame; rotation block stays (it's already a tangent-
        #      space perturbation, frame-independent at first order).
        #   2. Bias random walks — diagonal, scaled by dt.
        Q = np.zeros((_N, _N), dtype=np.float64)
        Qpre = preint.covariance.copy()
        # Lift body-frame covariance blocks into world frame.
        # G has the rotation acting on velocity + position only.
        # We rebuild a 9×9 G that maps the body-frame error to the
        # world-frame error order [δθ, δv, δp] → [δθ, δv_w, δp_w].
        G = np.eye(9, dtype=np.float64)
        G[3:6, 3:6] = R_w_i
        G[6:9, 6:9] = R_w_i
        Qpre_w = G @ Qpre @ G.T
        # Distribute into the 16×16: ordering inside Qpre is [θ, v, p],
        # so map to T_IDX, V_IDX, P_IDX in the EKF state.
        Q[T_IDX, T_IDX] = Qpre_w[0:3, 0:3]
        Q[V_IDX, V_IDX] = Qpre_w[3:6, 3:6]
        Q[P_IDX, P_IDX] = Qpre_w[6:9, 6:9]
        # Cross-terms (T-V, T-P, V-P) — keep them so the fuser doesn't
        # underestimate correlated uncertainty.
        Q[T_IDX, V_IDX] = Qpre_w[0:3, 3:6]; Q[V_IDX, T_IDX] = Q[T_IDX, V_IDX].T
        Q[T_IDX, P_IDX] = Qpre_w[0:3, 6:9]; Q[P_IDX, T_IDX] = Q[T_IDX, P_IDX].T
        Q[V_IDX, P_IDX] = Qpre_w[3:6, 6:9]; Q[P_IDX, V_IDX] = Q[V_IDX, P_IDX].T
        # Bias random walks (independent of pre-integration cov).
        Q[BG_IDX, BG_IDX] = np.eye(3) * (self._cfg.bias_gyro_random_walk  ** 2) * dt
        Q[BA_IDX, BA_IDX] = np.eye(3) * (self._cfg.bias_accel_random_walk ** 2) * dt
        # Scale random walk — lets the estimate follow DPVO's scale drift.
        Q[S_IDX, S_IDX] = np.eye(1) * (self._cfg.scale_random_walk ** 2) * dt

        # ---- propagate ---------------------------------------------------
        self._P = F @ self._P @ F.T + Q
        self._P = 0.5 * (self._P + self._P.T)   # numerical symmetry

        self._x.p_w_i = p_new
        self._x.v_w_i = v_new
        self._x.R_w_i = R_new
        self._t += dt

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(
        self,
        visual_pose: CameraPose,
        position_std_m:   Optional[float] = None,
        orientation_std_rad: Optional[float] = None,
    ) -> None:
        """
        Apply a 6-DOF visual pose measurement (camera=body in v1).

        Innovation
        ----------
        y_p = z_t - p̂_w_i / ŝ                 (3-vector)
        y_θ = Log(R̂_w_i^T · z_R)              (3-vector, axis-angle)

        Measurement Jacobian H (6×16), state ordering [p, v, θ, b_g, b_a, s]:
        the visual position observes the metric position divided by scale,
            h_p = p_w_i / s  → ∂h_p/∂δp = (1/ŝ)I,  ∂h_p/∂δs = -p̂_w_i/ŝ²
        and the orientation is observed directly (scale-free):
            R̂_w_i^T · h_R(x̂ ⊕ δx) ≈ Exp(δθ) → ∂y_R/∂δθ = +I
        Everything else is zero. With ŝ = 1 this reduces to the plain
        direct-observation Jacobian — except the δs column, which is what
        lets a visual-position innovation also correct scale.

        Joseph form is used for the covariance update — the symmetric +
        positive-semidefinite-preserving form CLAUDE.md mandates.
        """
        z_R = np.asarray(visual_pose.R, dtype=np.float64).reshape(3, 3)
        z_t = np.asarray(visual_pose.t, dtype=np.float64).reshape(3)
        s   = self._x.scale

        # Innovation. The monocular visual position is the metric
        # position divided by scale; rotation carries no scale.
        y_p = z_t - self._x.p_w_i / s
        y_R = R.from_matrix(self._x.R_w_i.T @ z_R).as_rotvec()
        y = np.concatenate([y_p, y_R])

        # Measurement Jacobian H (6×16). With h_p = p_w_i / scale:
        #   ∂h_p/∂δp = (1/s) I       ∂h_p/∂δs = -p_w_i / s²
        # rotation is observed directly: ∂y_R/∂δθ = I.
        H = np.zeros((6, _N), dtype=np.float64)
        H[0:3, P_IDX] = np.eye(3) / s
        H[0:3, S_IDX] = (-self._x.p_w_i / (s * s)).reshape(3, 1)
        H[3:6, T_IDX] = np.eye(3)

        # Measurement covariance
        pos_var = (position_std_m    if position_std_m    is not None
                   else self._cfg.visual_position_std_m)    ** 2
        rot_var = (orientation_std_rad if orientation_std_rad is not None
                   else self._cfg.visual_orientation_std_rad) ** 2
        Rm = np.diag([pos_var, pos_var, pos_var,
                      rot_var, rot_var, rot_var])

        # Gain
        S = H @ self._P @ H.T + Rm
        K = self._P @ H.T @ np.linalg.inv(S)

        dx = K @ y                   # 16-vector correction

        # Joseph form: P = (I - K H) P (I - K H)^T + K R K^T
        I_KH = np.eye(_N) - K @ H
        self._P = I_KH @ self._P @ I_KH.T + K @ Rm @ K.T
        self._P = 0.5 * (self._P + self._P.T)

        self._inject(dx)

    def update_zero_velocity(self, velocity_std: Optional[float] = None) -> None:
        """
        Zero-velocity update (ZUPT).

        When the platform is known to be at rest the body velocity is
        ~0. Applying that as a 3-DOF measurement of v_w_i bounds
        velocity drift, and — through the velocity↔accel-bias
        covariance the predict step builds up — also sharpens the
        accel-bias estimate. Caller gates this on a stationarity test
        (see state_estimation.imu_init.is_stationary). Joseph form, as
        for the visual update.
        """
        std = (velocity_std if velocity_std is not None
               else self._cfg.zupt_velocity_std)

        y = -self._x.v_w_i                       # innovation: 0 − v̂_w_i
        H = np.zeros((3, _N), dtype=np.float64)
        H[:, V_IDX] = np.eye(3)
        Rm = np.eye(3) * (std ** 2)

        S = H @ self._P @ H.T + Rm
        K = self._P @ H.T @ np.linalg.inv(S)
        dx = K @ y

        I_KH = np.eye(_N) - K @ H
        self._P = I_KH @ self._P @ I_KH.T + K @ Rm @ K.T
        self._P = 0.5 * (self._P + self._P.T)

        self._inject(dx)

    def _inject(self, dx: np.ndarray) -> None:
        """
        Apply the error-state correction to the nominal manifold.

        Position / velocity / biases compose additively. Rotation
        composes multiplicatively via right-Exp:
            R_new = R̂ · Exp(δθ)
        After injection the error state is implicitly zero (we never
        store it); covariance reset Jacobians are approximately identity
        for the small corrections this filter produces and are omitted.
        """
        self._x.p_w_i = self._x.p_w_i + dx[P_IDX]
        self._x.v_w_i = self._x.v_w_i + dx[V_IDX]
        self._x.R_w_i = self._x.R_w_i @ _so3_exp(dx[T_IDX])
        self._x.b_g   = self._x.b_g   + dx[BG_IDX]
        self._x.b_a   = self._x.b_a   + dx[BA_IDX]
        # Scale composes additively; clamp away from zero so the
        # h_p = p / scale measurement model can never divide by ~0.
        self._x.scale = max(1e-3, self._x.scale + float(dx[15]))


# ---------------------------------------------------------------------------
# SO(3) helpers — duplicated from imu_preintegration to avoid a leaky import.
# ---------------------------------------------------------------------------

def _so3_exp(phi: np.ndarray) -> np.ndarray:
    return R.from_rotvec(phi).as_matrix()


def _skew(v: np.ndarray) -> np.ndarray:
    return np.array([
        [0.0,   -v[2],  v[1]],
        [v[2],   0.0,  -v[0]],
        [-v[1],  v[0],  0.0],
    ], dtype=np.float64)
