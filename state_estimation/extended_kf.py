"""
Extended Kalman Filter for single-object bounding box tracking.

Upgrade from KalmanFilter (Step 4)
------------------------------------
KalmanFilter  — linear constant-velocity model, F is constant.
ExtendedKalmanFilter — nonlinear constant-turn-rate model, F(x,dt)
                       is recomputed every predict step via Jacobian.

State vector (9D)
-----------------
x = [cx, cy, w, h, vx, vy, vw, vh, ω]

    cx, cy : bounding box centre (pixels)
    w, h   : bounding box width and height (pixels)
    vx, vy : centre velocity (pixels/s)
    vw, vh : size change velocity (pixels/s)
    ω      : turn rate (radians/s)
              Positive ω = anticlockwise rotation of the velocity vector.

Constant turn rate (CTR) motion model
--------------------------------------
For a non-zero turn rate ω over interval dt:

    cx_new = cx + (vx*sin(ω*dt) + vy*(cos(ω*dt)-1)) / ω
    cy_new = cy + (vy*sin(ω*dt) - vx*(cos(ω*dt)-1)) / ω
    vx_new = vx*cos(ω*dt) - vy*sin(ω*dt)
    vy_new = vx*sin(ω*dt) + vy*cos(ω*dt)
    w_new  = w  + vw*dt
    h_new  = h  + vh*dt
    vw_new = vw
    vh_new = vh
    ω_new  = ω

For |ω| < ε (near-zero), degenerates to constant velocity:
    cx_new = cx + vx*dt
    cy_new = cy + vy*dt

Observation model
-----------------
Same as KalmanFilter: H selects [cx, cy, w, h] from the 9D state.

Jacobian F_jac
--------------
∂f/∂x evaluated at current state — 9×9 matrix.
Derived analytically for the CTR model.
Used in: P_new = F_jac @ P @ F_jac.T + Q

Joseph form update
------------------
Same as KalmanFilter — numerically stable for sustained operation.

Interface contract
------------------
ExtendedKalmanFilter is a drop-in replacement for KalmanFilter.
All public properties and methods have identical signatures.
ByteTracker.update() calls kf.predict(dt) and kf.update(obs) —
zero changes required to the tracker.

Upgrade path
------------
Step 11: World model reads kf.covariance for uncertainty-weighted
         occupancy grid updates.
Phase 3: Replace CTR model with a learned neural motion model —
         the Jacobian structure is preserved, only f() changes.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from numpy.linalg import LinAlgError

from perception.detector import Detection
from state_estimation.kalman_filter import KFSnapshot

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_STATE_EKF = 9   # [cx, cy, w, h, vx, vy, vw, vh, ω]
N_OBS_EKF   = 4   # [cx, cy, w, h]

# Turn rate threshold below which we use linear approximation
_OMEGA_EPS = 1e-4  # rad/s

# Chi-squared bounds for NIS with 4 DOF (same as vanilla KF)
NIS_LOWER = 0.711
NIS_UPPER = 9.488


# ---------------------------------------------------------------------------
# Observation matrix (4×9) — identical structure to vanilla KF
# ---------------------------------------------------------------------------

def _build_H_ekf() -> np.ndarray:
    """H selects [cx, cy, w, h] from 9D state."""
    H = np.zeros((N_OBS_EKF, N_STATE_EKF), dtype=np.float64)
    H[0, 0] = 1.0   # cx
    H[1, 1] = 1.0   # cy
    H[2, 2] = 1.0   # w
    H[3, 3] = 1.0   # h
    return H


# ---------------------------------------------------------------------------
# CTR motion model and Jacobian
# ---------------------------------------------------------------------------

def _ctr_predict(x: np.ndarray, dt: float) -> np.ndarray:
    """
    Apply constant turn rate (CTR) motion model.

    Parameters
    ----------
    x  : (9,) state vector [cx,cy,w,h,vx,vy,vw,vh,ω]
    dt : time step in seconds

    Returns
    -------
    (9,) predicted state vector
    """
    cx, cy, w, h, vx, vy, vw, vh, omega = x
    x_new = x.copy()

    if abs(omega) < _OMEGA_EPS:
        # Near-zero turn rate — degenerate to constant velocity
        x_new[0] = cx + vx * dt
        x_new[1] = cy + vy * dt
    else:
        sin_odt = np.sin(omega * dt)
        cos_odt = np.cos(omega * dt)

        x_new[0] = cx + (vx * sin_odt + vy * (cos_odt - 1.0)) / omega
        x_new[1] = cy + (vy * sin_odt - vx * (cos_odt - 1.0)) / omega
        x_new[4] = vx * cos_odt - vy * sin_odt
        x_new[5] = vx * sin_odt + vy * cos_odt

    # Size components — linear (ω does not affect w/h)
    x_new[2] = w  + vw * dt
    x_new[3] = h  + vh * dt
    # vw, vh, ω unchanged
    return x_new


def _ctr_jacobian(x: np.ndarray, dt: float) -> np.ndarray:
    """
    Compute the Jacobian of the CTR motion model: F_jac = ∂f/∂x.

    Parameters
    ----------
    x  : (9,) state vector at which to linearise
    dt : time step in seconds

    Returns
    -------
    (9,9) Jacobian matrix

    Derivation
    ----------
    For the nonlinear terms (rows 0,1,4,5 — cx,cy,vx,vy):

    ∂cx_new/∂vx  =  sin(ωdt)/ω
    ∂cx_new/∂vy  = (cos(ωdt)-1)/ω
    ∂cx_new/∂ω   =  (vx*(ωdt*cos(ωdt)-sin(ωdt)) + vy*ωdt*sin(ωdt)) / ω²

    ∂cy_new/∂vx  = -(cos(ωdt)-1)/ω
    ∂cy_new/∂vy  =  sin(ωdt)/ω
    ∂cy_new/∂ω   =  (vy*(ωdt*cos(ωdt)-sin(ωdt)) - vx*ωdt*sin(ωdt)) / ω²

    ∂vx_new/∂vx  =  cos(ωdt)
    ∂vx_new/∂vy  = -sin(ωdt)
    ∂vx_new/∂ω   = -dt*(vx*sin(ωdt)+vy*cos(ωdt))

    ∂vy_new/∂vx  =  sin(ωdt)
    ∂vy_new/∂vy  =  cos(ωdt)
    ∂vy_new/∂ω   =  dt*(vx*cos(ωdt)-vy*sin(ωdt))
    """
    cx, cy, w, h, vx, vy, vw, vh, omega = x

    F = np.eye(N_STATE_EKF, dtype=np.float64)

    if abs(omega) < _OMEGA_EPS:
        # Linear fallback Jacobian — same as vanilla KF
        F[0, 4] = dt   # ∂cx/∂vx
        F[1, 5] = dt   # ∂cy/∂vy
        F[2, 6] = dt   # ∂w/∂vw
        F[3, 7] = dt   # ∂h/∂vh
        # Within the linear regime |ω| < _OMEGA_EPS, _ctr_predict
        # uses cx += vx*dt regardless of ω, so ∂cx/∂ω = 0 here.
        # The L'Hôpital continuous limit is non-zero but does not
        # reflect the actual piecewise function behaviour.
        F[0, 8] = 0.0
        F[1, 8] = 0.0
        return F

    sin_odt = np.sin(omega * dt)
    cos_odt = np.cos(omega * dt)
    odt     = omega * dt
    o2      = omega * omega

    # Row 0: ∂cx_new / ∂[vx, vy, ω]
    F[0, 4] =  sin_odt / omega
    F[0, 5] = (cos_odt - 1.0) / omega
    F[0, 8] = (vx * (odt * cos_odt - sin_odt)
               - vy * (odt * sin_odt + cos_odt - 1.0)) / o2

    # Row 1: ∂cy_new / ∂[vx, vy, ω]
    F[1, 4] = -(cos_odt - 1.0) / omega
    F[1, 5] =  sin_odt / omega
    F[1, 8] = (vy * (odt * cos_odt - sin_odt)
               + vx * (odt * sin_odt + cos_odt - 1.0)) / o2

    # Row 2: ∂w_new / ∂vw
    F[2, 6] = dt

    # Row 3: ∂h_new / ∂vh
    F[3, 7] = dt

    # Row 4: ∂vx_new / ∂[vx, vy, ω]
    F[4, 4] =  cos_odt
    F[4, 5] = -sin_odt
    F[4, 8] = -dt * (vx * sin_odt + vy * cos_odt)

    # Row 5: ∂vy_new / ∂[vx, vy, ω]
    F[5, 4] =  sin_odt
    F[5, 5] =  cos_odt
    F[5, 8] =  dt * (vx * cos_odt - vy * sin_odt)

    # Rows 6,7,8: vw, vh, ω — identity (no change)
    return F


def _build_Q_ekf(dt: float, config: dict) -> np.ndarray:
    """
    Process noise Q (9×9) for the EKF.
    Extends the vanilla KF Q with an additional ω noise term.
    """
    cfg = config.get("extended_kalman_filter",
            config.get("kalman_filter", {})).get("process_noise", {})

    q_pos   = float(cfg.get("q_position",  1.0))
    q_size  = float(cfg.get("q_size",      1.0))
    q_vel   = float(cfg.get("q_velocity",  0.1))
    q_vsize = float(cfg.get("q_vel_size",  0.02))
    q_omega = float(cfg.get("q_omega",     0.01))

    diag = np.array([
        q_pos,   # cx
        q_pos,   # cy
        q_size,  # w
        q_size,  # h
        q_vel,   # vx
        q_vel,   # vy
        q_vsize, # vw
        q_vsize, # vh
        q_omega, # ω
    ], dtype=np.float64)

    return np.diag(diag) * dt


def _build_R_ekf(config: dict, confidence: Optional[float] = None) -> np.ndarray:
    """
    Measurement noise R (4×4) — identical to vanilla KF.
    Scaled by 1/confidence when provided.
    """
    cfg = config.get("extended_kalman_filter",
            config.get("kalman_filter", {})).get("measurement_noise", {})

    r_center = float(cfg.get("r_center", 1.0))
    r_size   = float(cfg.get("r_size",   1.0))

    R = np.diag([r_center, r_center, r_size, r_size]).astype(np.float64)

    if confidence is not None and confidence > 0.0:
        R = R / float(np.clip(confidence, 0.01, 1.0))

    return R


# ---------------------------------------------------------------------------
# Extended Kalman Filter
# ---------------------------------------------------------------------------

class ExtendedKalmanFilter:
    """
    Extended Kalman Filter with constant turn rate (CTR) motion model.

    Drop-in replacement for KalmanFilter — identical public interface.

    State: x = [cx, cy, w, h, vx, vy, vw, vh, ω]  (9D)
    Obs:   z = [cx, cy, w, h]                       (4D)

    Usage
    -----
    kf = ExtendedKalmanFilter.from_detection(det, config)
    kf.predict(dt=0.033)
    kf.update(obs=det.bbox_xywh, confidence=det.confidence)

    state     = kf.state          # (9,)
    covariance = kf.covariance    # (9,9)
    bbox       = kf.bbox_xyxy     # (4,)
    nis_val    = kf.nis()         # float
    """

    _H: np.ndarray = _build_H_ekf()

    def __init__(self, initial_state: np.ndarray, config: dict) -> None:
        if initial_state.shape != (N_STATE_EKF,):
            raise ValueError(
                f"initial_state must have shape ({N_STATE_EKF},), "
                f"got {initial_state.shape}"
            )

        self._config = config
        self._x = initial_state.copy().astype(np.float64)

        cfg = config.get("extended_kalman_filter",
                config.get("kalman_filter", {})).get(
                "initial_covariance", {})

        p_pos  = float(cfg.get("p_position",  10.0))
        p_size = float(cfg.get("p_size",      10.0))
        p_vel  = float(cfg.get("p_velocity", 100.0))
        p_omega = float(cfg.get("p_omega",    1.0))

        self._P = np.diag([
            p_pos, p_pos, p_size, p_size,
            p_vel, p_vel, p_vel,  p_vel,
            p_omega,
        ]).astype(np.float64)

        self._last_nis:   float          = float("nan")
        self._n_predict:  int            = 0
        self._n_update:   int            = 0
        self._innovation: Optional[np.ndarray] = None
        self._innov_cov:  Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_detection(
        cls, det: Detection, config: dict
    ) -> "ExtendedKalmanFilter":
        """Initialise from a Detection. Turn rate initialised to zero."""
        cx, cy, w, h = det.bbox_xywh
        state = np.array(
            [cx, cy, w, h, 0.0, 0.0, 0.0, 0.0, 0.0],
            dtype=np.float64,
        )
        return cls(state, config)

    # ------------------------------------------------------------------
    # Core EKF cycle
    # ------------------------------------------------------------------

    def predict(self, dt: float) -> np.ndarray:
        """
        EKF prediction step.

        x̂ₖ = f(xₖ₋₁, dt)           — nonlinear CTR propagation
        Pₖ = F_jac @ Pₖ₋₁ @ F_jacᵀ + Q   — linearised covariance

        Parameters
        ----------
        dt : time since last predict (seconds, must be positive)
        """
        if dt < 0.0:
            raise ValueError(
                f"dt must be non-negative, got {dt:.6f}s. "
                "Check that you are using time.monotonic() consistently."
            )
        if dt == 0.0:
            warnings.warn(
                "predict() called with dt=0. State unchanged.",
                stacklevel=2,
            )
            return self._x.copy()

        # Nonlinear state propagation
        self._x = _ctr_predict(self._x, dt)

        # Linearised covariance propagation
        F = _ctr_jacobian(self._x, dt)
        Q = _build_Q_ekf(dt, self._config)

        self._P = F @ self._P @ F.T + Q
        self._P = (self._P + self._P.T) * 0.5   # enforce symmetry

        self._n_predict += 1
        return self._x.copy()

    def update(
        self,
        observation: np.ndarray,
        confidence: Optional[float] = None,
    ) -> np.ndarray:
        """
        EKF update step — identical math to vanilla KF because H is linear.

        Innovation:    y  = z − H @ x̂
        Innov cov:     S  = H @ P @ Hᵀ + R
        Kalman gain:   K  = P @ Hᵀ @ S⁻¹
        State update:  x  = x̂ + K @ y
        Cov (Joseph):  P  = (I−KH) @ P @ (I−KH)ᵀ + K @ R @ Kᵀ
        NIS:           yᵀ S⁻¹ y  ~ χ²(4)
        """
        z = np.asarray(observation, dtype=np.float64)
        if z.shape != (N_OBS_EKF,):
            raise ValueError(
                f"observation must have shape ({N_OBS_EKF},), got {z.shape}"
            )

        H = self._H
        R = _build_R_ekf(self._config, confidence)

        y = z - H @ self._x                        # innovation (4,)
        S = H @ self._P @ H.T + R                  # innovation cov (4,4)

        self._innovation = y.copy()
        self._innov_cov  = S.copy()

        PH_T = self._P @ H.T                       # (9,4)
        try:
            K = np.linalg.solve(S.T, PH_T.T).T    # (9,4)
        except LinAlgError:
            warnings.warn(
                "Innovation covariance S is singular. "
                "Falling back to pseudo-inverse.",
                stacklevel=2,
            )
            K = PH_T @ np.linalg.pinv(S)

        self._x = self._x + K @ y

        # Joseph form covariance update
        I_KH = np.eye(N_STATE_EKF) - K @ H
        self._P = I_KH @ self._P @ I_KH.T + K @ R @ K.T
        self._P = (self._P + self._P.T) * 0.5

        self._last_nis = float(y @ np.linalg.solve(S, y))
        self._n_update += 1
        return self._x.copy()

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def nis(self) -> float:
        """
        Normalised Innovation Squared from last update.
        NIS ~ χ²(4). Consistent filter: NIS ∈ [0.71, 9.49].
        Returns NaN before first update.
        """
        return self._last_nis

    def nees(
        self, true_state: np.ndarray, n_obs_dims: int = 4
    ) -> float:
        """
        Normalised Estimation Error Squared.

        NEES = eᵀ P⁻¹ e   where e = x_true - x_estimated

        Requires ground truth — only available in simulation.
        NEES ~ χ²(n_state). Consistent filter: mean NEES ≈ n_state.

        Parameters
        ----------
        true_state : (9,) or (4,) ground truth state vector.
                     If (4,), only position/size is checked.
        n_obs_dims : dimensionality for the chi-squared check.

        Returns
        -------
        NEES scalar (float). NaN if P is singular.
        """
        true_state = np.asarray(true_state, dtype=np.float64)

        if true_state.shape == (N_OBS_EKF,):
            # Only check position/size block
            e = true_state - self._x[:N_OBS_EKF]
            P_sub = self._P[:N_OBS_EKF, :N_OBS_EKF]
        elif true_state.shape == (N_STATE_EKF,):
            e = true_state - self._x
            P_sub = self._P
        else:
            raise ValueError(
                f"true_state must be shape ({N_OBS_EKF},) or "
                f"({N_STATE_EKF},), got {true_state.shape}"
            )

        try:
            return float(e @ np.linalg.solve(P_sub, e))
        except LinAlgError:
            return float("nan")

    def is_covariance_pd(self, tol: float = 1e-8) -> bool:
        """Check that P is positive definite via Cholesky."""
        try:
            np.linalg.cholesky(self._P + np.eye(N_STATE_EKF) * tol)
            return True
        except LinAlgError:
            return False

    def nis_is_consistent(self) -> bool:
        """True if last NIS is within χ²(4) 95% CI."""
        if np.isnan(self._last_nis):
            return False
        return NIS_LOWER <= self._last_nis <= NIS_UPPER

    def covariance_trace(self) -> float:
        return float(np.trace(self._P))

    def snapshot(self, timestamp: float, frame_idx: int) -> KFSnapshot:
        return KFSnapshot(
            timestamp=timestamp,
            frame_idx=frame_idx,
            state=self._x.copy(),
            covariance=self._P.copy(),
            nis=self._last_nis,
            n_updates=self._n_update,
        )

    # ------------------------------------------------------------------
    # State accessors — identical interface to KalmanFilter
    # ------------------------------------------------------------------

    @property
    def state(self) -> np.ndarray:
        """Full 9D state [cx,cy,w,h,vx,vy,vw,vh,ω]."""
        return self._x.copy()

    @property
    def covariance(self) -> np.ndarray:
        return self._P.copy()

    @property
    def position(self) -> np.ndarray:
        return self._x[:2].copy()

    @property
    def size(self) -> np.ndarray:
        return self._x[2:4].copy()

    @property
    def velocity(self) -> np.ndarray:
        """[vx, vy, vw, vh] — same slice as KalmanFilter for compatibility."""
        return self._x[4:8].copy()

    @property
    def turn_rate(self) -> float:
        """Estimated turn rate ω in radians/second."""
        return float(self._x[8])

    @property
    def bbox_xywh(self) -> np.ndarray:
        return self._x[:4].copy()

    @property
    def bbox_xyxy(self) -> np.ndarray:
        cx, cy, w, h = self._x[:4]
        w = max(w, 1.0)
        h = max(h, 1.0)
        return np.array([
            cx - w / 2.0,
            cy - h / 2.0,
            cx + w / 2.0,
            cy + h / 2.0,
        ], dtype=np.float64)

    @property
    def position_variance(self) -> np.ndarray:
        return np.diag(self._P)[:2].copy()

    @property
    def n_predict(self) -> int:
        return self._n_predict

    @property
    def n_update(self) -> int:
        return self._n_update

    def __repr__(self) -> str:
        cx, cy, w, h = self._x[:4]
        omega = self._x[8]
        return (
            f"ExtendedKalmanFilter("
            f"cx={cx:.1f} cy={cy:.1f} w={w:.1f} h={h:.1f} "
            f"ω={omega:.3f}rad/s "
            f"updates={self._n_update} nis={self._last_nis:.2f})"
        )
