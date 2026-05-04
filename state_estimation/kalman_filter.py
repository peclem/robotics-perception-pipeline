"""
Kalman Filter for single-object state estimation.

State vector
------------
x = [cx, cy, w, h, vx, vy, vw, vh]

    cx, cy : bounding box center in pixels
    w, h   : bounding box width and height in pixels
    vx, vy : center velocity (pixels/second)
    vw, vh : size change velocity (pixels/second)

Motion model
------------
Constant velocity: position integrates velocity over dt.
No acceleration term — objects accelerate slowly relative to the
frame rate, so the process noise Q absorbs acceleration.

Observation model
-----------------
We observe [cx, cy, w, h] directly from the detector.
H is the 4×8 selection matrix that extracts the observed states.

Joseph form update
------------------
P = (I−KH)P(I−KH)ᵀ + KRKᵀ

Algebraically identical to P = (I−KH)P but numerically stable
under sustained operation because it preserves symmetry of P
even when floating-point errors accumulate.

NIS diagnostic
--------------
NIS = yᵀ S⁻¹ y  where y = z − Hx̂ (innovation), S = HPHᵀ + R

NIS ~ χ²(n_obs). For n_obs=4:
    χ²(4, 0.05) = 9.49   (95th percentile upper bound)
    χ²(4, 0.95) = 0.71   (5th percentile lower bound)

A filter running with NIS consistently outside [0.71, 9.49] has
miscalibrated Q or R. Check this before trusting state estimates.

Upgrade path
------------
Step 5: Each ByteTrack Track owns one KalmanFilter instance.
Step 10: Replace with ExtendedKalmanFilter (nonlinear motion model).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from numpy.linalg import LinAlgError

from perception.detector import Detection


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_STATE = 8   # dimension of state vector x
N_OBS   = 4   # dimension of observation vector z

# Chi-squared bounds for NIS with 4 DOF at 95% confidence interval
NIS_LOWER_BOUND = 0.711   # chi2.ppf(0.05, df=4)
NIS_UPPER_BOUND = 9.488   # chi2.ppf(0.95, df=4)


# ---------------------------------------------------------------------------
# Matrices — built once, never mutated
# ---------------------------------------------------------------------------

def _build_H() -> np.ndarray:
    """
    Observation matrix H (4×8).
    Maps full state [cx,cy,w,h,vx,vy,vw,vh] → observation [cx,cy,w,h].
    """
    H = np.zeros((N_OBS, N_STATE), dtype=np.float64)
    H[0, 0] = 1.0   # cx
    H[1, 1] = 1.0   # cy
    H[2, 2] = 1.0   # w
    H[3, 3] = 1.0   # h
    return H


def _build_F(dt: float) -> np.ndarray:
    """
    State transition matrix F (8×8) for constant-velocity model.

    x_k = F @ x_{k-1}

    Position integrates velocity:
        cx_new = cx + vx * dt
        cy_new = cy + vy * dt
        w_new  = w  + vw * dt
        h_new  = h  + vh * dt

    Velocity is constant (no acceleration term):
        vx_new = vx,  etc.
    """
    F = np.eye(N_STATE, dtype=np.float64)
    # Position rows: x += v * dt
    F[0, 4] = dt   # cx += vx * dt
    F[1, 5] = dt   # cy += vy * dt
    F[2, 6] = dt   # w  += vw * dt
    F[3, 7] = dt   # h  += vh * dt
    return F


def _build_Q(dt: float, config: dict) -> np.ndarray:
    """
    Process noise covariance Q (8×8).

    Models how much the true state can deviate from the constant-velocity
    prediction between frames. Larger Q = more trust in measurements,
    faster response to manoeuvres but noisier estimates.

    We use a diagonal Q scaled by dt. All variances are in the config
    so they can be tuned without touching source code.

    Robotics note
    -------------
    A principled alternative is the discretised continuous-time model:
        Q = G @ q @ Gᵀ  where G = [dt²/2, dt]ᵀ and q is acceleration variance.
    We use the simpler diagonal form here — sufficient for pedestrian/vehicle
    tracking at 30fps. The EKF in Step 10 uses the principled form.
    """
    cfg = config.get("kalman_filter", {}).get("process_noise", {})

    # Position noise variance (pixels²) — how much the center can jump
    q_pos  = float(cfg.get("q_position",  4.0))
    # Size noise variance (pixels²) — how much w/h can change
    q_size = float(cfg.get("q_size",      4.0))
    # Velocity noise variance (pixels²/s²)
    q_vel  = float(cfg.get("q_velocity",  0.5))
    # Size-velocity noise variance
    q_vsize = float(cfg.get("q_vel_size", 0.1))

    diag = np.array([
        q_pos,   # cx
        q_pos,   # cy
        q_size,  # w
        q_size,  # h
        q_vel,   # vx
        q_vel,   # vy
        q_vsize, # vw
        q_vsize, # vh
    ], dtype=np.float64)

    return np.diag(diag) * dt


def _build_R(config: dict, confidence: Optional[float] = None) -> np.ndarray:
    """
    Measurement noise covariance R (4×4).

    Models how much we trust the detector's bbox output.
    Optionally scaled by detection confidence:
        R_scaled = R_base / confidence

    A low-confidence detection (e.g. 0.3) gets R scaled by 1/0.3 ≈ 3×,
    meaning the filter trusts the measurement less and relies more on
    the prediction — exactly the right behaviour during partial occlusion.

    Robotics note
    -------------
    In a real system, R comes from sensor calibration — you measure the
    detector's output variance on a calibration set. Here we use tunable
    defaults that work well for YOLOv8 on typical scenes.
    """
    cfg = config.get("kalman_filter", {}).get("measurement_noise", {})

    r_center = float(cfg.get("r_center", 1.0))   # cx, cy variance (pixels²)
    r_size   = float(cfg.get("r_size",   4.0))   # w, h variance (pixels²)

    R = np.diag([r_center, r_center, r_size, r_size]).astype(np.float64)

    if confidence is not None and confidence > 0.0:
        scale = 1.0 / float(np.clip(confidence, 0.01, 1.0))
        R = R * scale

    return R


# ---------------------------------------------------------------------------
# Filter state snapshot (immutable, for history buffer)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class KFSnapshot:
    """
    Immutable snapshot of filter state at a given timestamp.

    Stored in the Track's history buffer (Step 5) and used by the
    world model's scene graph (Step 11) for trajectory smoothing.
    """
    timestamp: float
    frame_idx: int
    state: np.ndarray          # (8,) float64
    covariance: np.ndarray     # (8,8) float64
    nis: float                 # Normalized Innovation Squared (NaN if no update)
    n_updates: int


# ---------------------------------------------------------------------------
# Kalman Filter
# ---------------------------------------------------------------------------

class KalmanFilter:
    """
    Kalman Filter for single-object bounding box tracking.

    Usage
    -----
    # Initialise from a Detection
    kf = KalmanFilter.from_detection(det, config)

    # Predict (called once per frame, before detections arrive)
    kf.predict(dt=0.033)

    # Update (called when a detection is associated to this track)
    kf.update(observation=det.bbox_xywh, confidence=det.confidence)

    # Read state
    kf.bbox_xyxy      # current estimate as [x1,y1,x2,y2]
    kf.state          # full 8D state vector
    kf.covariance     # 8×8 covariance matrix
    kf.nis            # last NIS value (NaN before first update)
    """

    _H: np.ndarray = _build_H()   # shared across all instances

    def __init__(self, initial_state: np.ndarray, config: dict):
        """
        Parameters
        ----------
        initial_state : (8,) array [cx, cy, w, h, vx, vy, vw, vh]
                        Velocities are typically initialised to zero.
        config        : full pipeline config dict (kalman_filter section used)
        """
        if initial_state.shape != (N_STATE,):
            raise ValueError(
                f"initial_state must have shape ({N_STATE},), "
                f"got {initial_state.shape}"
            )

        self._config = config
        self._x = initial_state.copy().astype(np.float64)

        # Initial covariance: high uncertainty on velocity, low on position
        # (we observed position directly, but velocity is inferred)
        cfg = config.get("kalman_filter", {}).get("initial_covariance", {})
        p_pos  = float(cfg.get("p_position",  10.0))
        p_size = float(cfg.get("p_size",      10.0))
        p_vel  = float(cfg.get("p_velocity", 100.0))

        self._P = np.diag([
            p_pos, p_pos, p_size, p_size,
            p_vel, p_vel, p_vel,  p_vel,
        ]).astype(np.float64)

        self._last_nis: float = float("nan")
        self._n_predict: int = 0
        self._n_update: int = 0

        # Innovation and innovation covariance from last update
        # (exposed for diagnostics and debugging)
        self._innovation: Optional[np.ndarray] = None
        self._innovation_cov: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_detection(cls, det: Detection, config: dict) -> "KalmanFilter":
        """
        Construct a KalmanFilter from a Detection.

        Initialises position from the detection's bbox_xywh.
        Velocity is initialised to zero — we have no velocity information
        from a single detection.
        """
        cx, cy, w, h = det.bbox_xywh
        initial_state = np.array(
            [cx, cy, w, h, 0.0, 0.0, 0.0, 0.0],
            dtype=np.float64,
        )
        return cls(initial_state, config)

    # ------------------------------------------------------------------
    # Core filter cycle
    # ------------------------------------------------------------------

    def predict(self, dt: float) -> np.ndarray:
        """
        Prediction step: propagate state and covariance forward by dt seconds.

        x̂ₖ|ₖ₋₁ = F @ xₖ₋₁
        Pₖ|ₖ₋₁  = F @ Pₖ₋₁ @ Fᵀ + Q

        Parameters
        ----------
        dt : time since last predict, in seconds.
             Must be positive. Negative dt indicates a timestamp bug
             (e.g. mixing monotonic and wall-clock time) — we guard against it.

        Returns
        -------
        Predicted state vector (8,).
        """
        if dt < 0.0:
            raise ValueError(
                f"dt must be non-negative, got {dt:.6f}s. "
                "Check that you are using time.monotonic() consistently — "
                "mixing monotonic and wall-clock timestamps causes negative dt."
            )
        if dt == 0.0:
            warnings.warn(
                "predict() called with dt=0. State unchanged. "
                "Did you call predict() twice on the same frame?",
                stacklevel=2,
            )
            return self._x.copy()

        F = _build_F(dt)
        Q = _build_Q(dt, self._config)

        self._x = F @ self._x
        self._P = F @ self._P @ F.T + Q

        # Enforce symmetry — floating-point arithmetic can make P slightly
        # asymmetric over many cycles, which eventually causes issues in update
        self._P = (self._P + self._P.T) * 0.5

        self._n_predict += 1
        return self._x.copy()

    def update(
        self,
        observation: np.ndarray,
        confidence: Optional[float] = None,
    ) -> np.ndarray:
        """
        Update step: correct state using a new observation.

        Innovation:       y  = z − H @ x̂
        Innovation cov:   S  = H @ P @ Hᵀ + R
        Kalman gain:      K  = P @ Hᵀ @ S⁻¹
        State update:     x  = x̂ + K @ y
        Covariance (Joseph form):
                          P  = (I−KH) @ P @ (I−KH)ᵀ + K @ R @ Kᵀ

        Parameters
        ----------
        observation : (4,) array [cx, cy, w, h] from Detection.bbox_xywh
        confidence  : detector confidence in [0,1], used to scale R.
                      None → use base R without scaling.

        Returns
        -------
        Updated state vector (8,).
        """
        z = np.asarray(observation, dtype=np.float64)
        if z.shape != (N_OBS,):
            raise ValueError(
                f"observation must have shape ({N_OBS},) [cx,cy,w,h], "
                f"got {z.shape}"
            )

        H = self._H
        R = _build_R(self._config, confidence)

        # Innovation
        y = z - H @ self._x                           # (4,)

        # Innovation covariance
        S = H @ self._P @ H.T + R                     # (4,4)

        # Store for NIS computation before we overwrite anything
        self._innovation = y.copy()
        self._innovation_cov = S.copy()

        # Kalman gain — solve S @ Kᵀ = (P @ Hᵀ)ᵀ via lstsq for stability
        # equivalent to K = P @ Hᵀ @ S⁻¹ but avoids explicit inversion
        PH_T = self._P @ H.T                          # (8,4)
        try:
            K = np.linalg.solve(S.T, PH_T.T).T        # (8,4)
        except LinAlgError:
            # S is singular — R is under-specified or P has collapsed.
            # Fall back to pseudo-inverse. Log a warning — this should not
            # happen in normal operation.
            warnings.warn(
                "Innovation covariance S is singular during update. "
                "Falling back to pseudo-inverse. Check R and Q tuning.",
                stacklevel=2,
            )
            K = PH_T @ np.linalg.pinv(S)

        # State update
        self._x = self._x + K @ y

        # Covariance update — Joseph form for numerical stability
        I_KH = np.eye(N_STATE) - K @ H
        self._P = I_KH @ self._P @ I_KH.T + K @ R @ K.T

        # Enforce symmetry
        self._P = (self._P + self._P.T) * 0.5

        # NIS: scalar, chi-squared distributed with N_OBS=4 DOF
        self._last_nis = float(y @ np.linalg.solve(S, y))

        self._n_update += 1
        return self._x.copy()

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def nis(self) -> float:
        """
        Normalized Innovation Squared from the last update step.

        NIS = yᵀ S⁻¹ y  ~ χ²(n_obs=4)

        Interpretation
        --------------
        Consistent filter:  NIS ∈ [0.71, 9.49]  (95% CI for χ²(4))
        NIS >> 9.49:  filter is overconfident (Q too small or R too large)
        NIS << 0.71:  filter is underconfident (Q too large or R too small)

        Returns NaN if update() has not been called yet.
        """
        return self._last_nis

    def is_covariance_pd(self, tol: float = 1e-8) -> bool:
        """
        Check that the covariance matrix P is positive definite.

        A non-PD covariance means the filter has become numerically
        degenerate — negative variances are physically meaningless and
        will cause the next predict/update to produce garbage.

        Uses Cholesky decomposition: P is PD iff Cholesky succeeds.
        Cheaper than full eigendecomposition.

        Parameters
        ----------
        tol : small regularisation added to diagonal before the check.
              Catches near-singular matrices before they become singular.
        """
        try:
            np.linalg.cholesky(self._P + np.eye(N_STATE) * tol)
            return True
        except LinAlgError:
            return False

    def nis_is_consistent(self) -> bool:
        """
        Returns True if the last NIS value is within the 95% CI for χ²(4).
        Returns False if NIS is NaN (no update yet) or out of bounds.
        """
        if np.isnan(self._last_nis):
            return False
        return NIS_LOWER_BOUND <= self._last_nis <= NIS_UPPER_BOUND

    def covariance_trace(self) -> float:
        """
        Sum of diagonal variances — a scalar measure of total uncertainty.
        Used to track filter convergence: trace should decrease after updates
        and increase during predict-only periods.
        """
        return float(np.trace(self._P))

    def snapshot(self, timestamp: float, frame_idx: int) -> KFSnapshot:
        """Return an immutable snapshot of current state for history logging."""
        return KFSnapshot(
            timestamp=timestamp,
            frame_idx=frame_idx,
            state=self._x.copy(),
            covariance=self._P.copy(),
            nis=self._last_nis,
            n_updates=self._n_update,
        )

    # ------------------------------------------------------------------
    # State accessors
    # ------------------------------------------------------------------

    @property
    def state(self) -> np.ndarray:
        """Full 8D state vector [cx, cy, w, h, vx, vy, vw, vh]."""
        return self._x.copy()

    @property
    def covariance(self) -> np.ndarray:
        """8×8 covariance matrix P."""
        return self._P.copy()

    @property
    def position(self) -> np.ndarray:
        """Estimated center [cx, cy]."""
        return self._x[:2].copy()

    @property
    def size(self) -> np.ndarray:
        """Estimated size [w, h]."""
        return self._x[2:4].copy()

    @property
    def velocity(self) -> np.ndarray:
        """Estimated velocity [vx, vy, vw, vh]."""
        return self._x[4:].copy()

    @property
    def bbox_xywh(self) -> np.ndarray:
        """Current estimate as [cx, cy, w, h]."""
        return self._x[:4].copy()

    @property
    def bbox_xyxy(self) -> np.ndarray:
        """
        Current estimate as [x1, y1, x2, y2].
        Clips negative dimensions to zero — a physically impossible
        state (negative width) can occur transiently during aggressive
        manoeuvres and must not propagate to the tracker.
        """
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
        """Diagonal position variances [var_cx, var_cy]."""
        return np.diag(self._P)[:2].copy()

    @property
    def n_predict(self) -> int:
        return self._n_predict

    @property
    def n_update(self) -> int:
        return self._n_update

    def __repr__(self) -> str:
        cx, cy, w, h = self._x[:4]
        return (
            f"KalmanFilter(cx={cx:.1f} cy={cy:.1f} w={w:.1f} h={h:.1f} "
            f"updates={self._n_update} nis={self._last_nis:.2f})"
        )
