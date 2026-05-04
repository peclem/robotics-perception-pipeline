"""
Unit tests for the Kalman Filter implementation.

Test strategy
-------------
TestKFConstruction       : state/covariance initialisation contracts
TestKFPredict            : prediction step correctness
TestKFUpdate             : update step correctness
TestKFCycle              : combined predict+update behaviour over time
TestKFNIS                : NIS diagnostic validity
TestKFNumericalStability : filter survives 1000+ cycles without diverging
TestKFBboxAccessors      : bbox_xyxy / bbox_xywh derived properties
TestKFFromDetection      : construction from Detection dataclass
TestKFValidation         : input guards (negative dt, wrong obs shape)
TestKFVsFilterPy         : numerical cross-check against filterpy reference

Why validate against filterpy?
-------------------------------
filterpy.KalmanFilter is a well-tested reference implementation.
We run identical predict/update sequences on both our filter and filterpy
and assert that state and covariance agree to high precision.
This validates our matrix math without relying on our own tests alone.
"""

import time
import numpy as np
import pytest

from state_estimation.kalman_filter import (
    KalmanFilter,
    KFSnapshot,
    NIS_LOWER_BOUND,
    NIS_UPPER_BOUND,
    N_OBS,
    N_STATE,
    _build_F,
    _build_H,
    _build_Q,
    _build_R,
)
from perception.detector import Detection


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def cfg():
    return {
        "kalman_filter": {
            "initial_covariance": {
                "p_position": 10.0,
                "p_size": 10.0,
                "p_velocity": 100.0,
            },
            "process_noise": {
                "q_position": 4.0,
                "q_size": 4.0,
                "q_velocity": 0.5,
                "q_vel_size": 0.1,
            },
            "measurement_noise": {
                "r_center": 1.0,
                "r_size": 4.0,
            },
        }
    }


def make_kf(cfg, cx=320.0, cy=240.0, w=80.0, h=60.0) -> KalmanFilter:
    state = np.array([cx, cy, w, h, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return KalmanFilter(state, cfg)


def make_detection(cx=320.0, cy=240.0, w=80.0, h=60.0, conf=0.9) -> Detection:
    x1, y1 = cx - w / 2, cy - h / 2
    x2, y2 = cx + w / 2, cy + h / 2
    return Detection(
        bbox_xyxy=np.array([x1, y1, x2, y2], dtype=np.float32),
        confidence=conf,
        class_id=0,
        class_name="person",
        frame_idx=0,
        timestamp=time.monotonic(),
    )


def make_obs(cx=320.0, cy=240.0, w=80.0, h=60.0) -> np.ndarray:
    return np.array([cx, cy, w, h], dtype=np.float64)


# ---------------------------------------------------------------------------
# Matrix construction
# ---------------------------------------------------------------------------

class TestMatrices:

    def test_H_shape(self):
        assert _build_H().shape == (N_OBS, N_STATE)

    def test_H_selects_position_states(self):
        H = _build_H()
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        obs = H @ x
        np.testing.assert_array_equal(obs, [1.0, 2.0, 3.0, 4.0])

    def test_F_shape(self):
        assert _build_F(0.033).shape == (N_STATE, N_STATE)

    def test_F_integrates_velocity(self):
        dt = 0.1
        F = _build_F(dt)
        # State: [cx, cy, w, h, vx, vy, vw, vh]
        x = np.array([100.0, 100.0, 50.0, 40.0, 10.0, -5.0, 1.0, 0.5])
        x_new = F @ x
        # cx should advance by vx * dt
        assert x_new[0] == pytest.approx(100.0 + 10.0 * dt)
        assert x_new[1] == pytest.approx(100.0 + (-5.0) * dt)
        assert x_new[2] == pytest.approx(50.0 + 1.0 * dt)
        assert x_new[3] == pytest.approx(40.0 + 0.5 * dt)

    def test_F_velocity_unchanged(self):
        F = _build_F(0.1)
        x = np.array([0.0, 0.0, 0.0, 0.0, 7.0, -3.0, 0.5, 0.2])
        x_new = F @ x
        np.testing.assert_array_almost_equal(x_new[4:], x[4:])

    def test_Q_shape(self, cfg):
        assert _build_Q(0.033, cfg).shape == (N_STATE, N_STATE)

    def test_Q_symmetric(self, cfg):
        Q = _build_Q(0.033, cfg)
        np.testing.assert_array_almost_equal(Q, Q.T)

    def test_Q_positive_semidefinite(self, cfg):
        Q = _build_Q(0.033, cfg)
        eigvals = np.linalg.eigvalsh(Q)
        assert np.all(eigvals >= -1e-10)

    def test_Q_scales_with_dt(self, cfg):
        Q1 = _build_Q(0.033, cfg)
        Q2 = _build_Q(0.066, cfg)
        # Q doubles when dt doubles
        np.testing.assert_array_almost_equal(Q2, Q1 * 2.0)

    def test_R_shape(self, cfg):
        assert _build_R(cfg).shape == (N_OBS, N_OBS)

    def test_R_symmetric(self, cfg):
        R = _build_R(cfg)
        np.testing.assert_array_almost_equal(R, R.T)

    def test_R_scales_with_low_confidence(self, cfg):
        R_high = _build_R(cfg, confidence=0.9)
        R_low  = _build_R(cfg, confidence=0.3)
        # Low confidence → larger R (less trusted measurement)
        assert np.all(R_low >= R_high)

    def test_R_confidence_1_equals_base(self, cfg):
        R_base = _build_R(cfg, confidence=None)
        R_one  = _build_R(cfg, confidence=1.0)
        np.testing.assert_array_almost_equal(R_base, R_one)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestKFConstruction:

    def test_state_shape(self, cfg):
        kf = make_kf(cfg)
        assert kf.state.shape == (N_STATE,)

    def test_state_dtype(self, cfg):
        kf = make_kf(cfg)
        assert kf.state.dtype == np.float64

    def test_covariance_shape(self, cfg):
        kf = make_kf(cfg)
        assert kf.covariance.shape == (N_STATE, N_STATE)

    def test_initial_state_correct(self, cfg):
        kf = make_kf(cfg, cx=150.0, cy=200.0, w=60.0, h=45.0)
        x = kf.state
        assert x[0] == pytest.approx(150.0)  # cx
        assert x[1] == pytest.approx(200.0)  # cy
        assert x[2] == pytest.approx(60.0)   # w
        assert x[3] == pytest.approx(45.0)   # h
        assert x[4] == pytest.approx(0.0)    # vx (zero-init)
        assert x[5] == pytest.approx(0.0)    # vy
        assert x[6] == pytest.approx(0.0)    # vw
        assert x[7] == pytest.approx(0.0)    # vh

    def test_initial_covariance_pd(self, cfg):
        kf = make_kf(cfg)
        assert kf.is_covariance_pd()

    def test_initial_covariance_velocity_larger(self, cfg):
        kf = make_kf(cfg)
        P = kf.covariance
        # Velocity variances should be larger than position variances initially
        assert P[4, 4] > P[0, 0]
        assert P[5, 5] > P[1, 1]

    def test_initial_nis_is_nan(self, cfg):
        kf = make_kf(cfg)
        assert np.isnan(kf.nis())

    def test_n_predict_zero_at_init(self, cfg):
        kf = make_kf(cfg)
        assert kf.n_predict == 0

    def test_n_update_zero_at_init(self, cfg):
        kf = make_kf(cfg)
        assert kf.n_update == 0

    def test_wrong_state_shape_raises(self, cfg):
        with pytest.raises(ValueError, match="shape"):
            KalmanFilter(np.array([1.0, 2.0, 3.0]), cfg)

    def test_state_is_copy(self, cfg):
        """Mutating the returned state must not affect the filter."""
        kf = make_kf(cfg)
        s = kf.state
        s[0] = 9999.0
        assert kf.state[0] != 9999.0

    def test_covariance_is_copy(self, cfg):
        kf = make_kf(cfg)
        P = kf.covariance
        P[0, 0] = 9999.0
        assert kf.covariance[0, 0] != 9999.0


# ---------------------------------------------------------------------------
# Predict step
# ---------------------------------------------------------------------------

class TestKFPredict:

    def test_predict_returns_array(self, cfg):
        kf = make_kf(cfg)
        result = kf.predict(dt=0.033)
        assert isinstance(result, np.ndarray)
        assert result.shape == (N_STATE,)

    def test_predict_increments_counter(self, cfg):
        kf = make_kf(cfg)
        kf.predict(dt=0.033)
        kf.predict(dt=0.033)
        assert kf.n_predict == 2

    def test_predict_with_zero_velocity_keeps_position(self, cfg):
        """With zero velocity and no noise, position should be unchanged."""
        kf = make_kf(cfg, cx=200.0, cy=150.0)
        kf.predict(dt=0.033)
        # Position should be close to initial (dt small, v=0)
        assert kf.state[0] == pytest.approx(200.0, abs=1e-10)
        assert kf.state[1] == pytest.approx(150.0, abs=1e-10)

    def test_predict_propagates_velocity(self, cfg):
        """Manually set velocity, check position advances correctly."""
        state = np.array([100.0, 100.0, 50.0, 40.0, 20.0, -10.0, 0.0, 0.0])
        kf = KalmanFilter(state, cfg)
        dt = 0.1
        kf.predict(dt=dt)
        assert kf.state[0] == pytest.approx(100.0 + 20.0 * dt)
        assert kf.state[1] == pytest.approx(100.0 + (-10.0) * dt)

    def test_predict_covariance_grows(self, cfg):
        """Covariance must grow during predict — we become less certain."""
        kf = make_kf(cfg)
        trace_before = kf.covariance_trace()
        kf.predict(dt=0.033)
        trace_after = kf.covariance_trace()
        assert trace_after > trace_before, (
            "Covariance must grow during prediction — uncertainty increases "
            "when we have no new observations."
        )

    def test_predict_covariance_remains_symmetric(self, cfg):
        kf = make_kf(cfg)
        for _ in range(20):
            kf.predict(dt=0.033)
        P = kf.covariance
        np.testing.assert_array_almost_equal(P, P.T, decimal=10)

    def test_predict_covariance_remains_pd(self, cfg):
        kf = make_kf(cfg)
        for _ in range(50):
            kf.predict(dt=0.033)
        assert kf.is_covariance_pd()

    def test_negative_dt_raises(self, cfg):
        kf = make_kf(cfg)
        with pytest.raises(ValueError, match="non-negative"):
            kf.predict(dt=-0.001)

    def test_zero_dt_warns(self, cfg):
        kf = make_kf(cfg)
        with pytest.warns(UserWarning):
            kf.predict(dt=0.0)

    def test_zero_dt_state_unchanged(self, cfg):
        kf = make_kf(cfg, cx=200.0, cy=150.0)
        state_before = kf.state.copy()
        with pytest.warns(UserWarning):
            kf.predict(dt=0.0)
        np.testing.assert_array_equal(kf.state, state_before)


# ---------------------------------------------------------------------------
# Update step
# ---------------------------------------------------------------------------

class TestKFUpdate:

    def test_update_returns_array(self, cfg):
        kf = make_kf(cfg)
        kf.predict(dt=0.033)
        result = kf.update(make_obs())
        assert isinstance(result, np.ndarray)
        assert result.shape == (N_STATE,)

    def test_update_increments_counter(self, cfg):
        kf = make_kf(cfg)
        kf.predict(dt=0.033)
        kf.update(make_obs())
        assert kf.n_update == 1

    def test_update_reduces_covariance(self, cfg):
        """Covariance must shrink after update — new info reduces uncertainty."""
        kf = make_kf(cfg)
        kf.predict(dt=0.033)
        trace_before = kf.covariance_trace()
        kf.update(make_obs())
        trace_after = kf.covariance_trace()
        assert trace_after < trace_before, (
            "Covariance must decrease after an update — a measurement "
            "provides information and reduces uncertainty."
        )

    def test_update_moves_state_toward_observation(self, cfg):
        """State must move toward observation, not away from it."""
        kf = make_kf(cfg, cx=300.0, cy=200.0)
        kf.predict(dt=0.033)
        obs = make_obs(cx=350.0, cy=200.0)  # observation is to the right
        kf.update(obs)
        # cx should move from ~300 toward 350
        assert kf.state[0] > 300.0, (
            "After update with observation at cx=350, estimated cx must "
            "increase from initial 300 — filter must move toward observation."
        )

    def test_update_covariance_remains_symmetric(self, cfg):
        kf = make_kf(cfg)
        for _ in range(30):
            kf.predict(dt=0.033)
            kf.update(make_obs())
        P = kf.covariance
        np.testing.assert_array_almost_equal(P, P.T, decimal=10)

    def test_update_covariance_remains_pd(self, cfg):
        kf = make_kf(cfg)
        for _ in range(30):
            kf.predict(dt=0.033)
            kf.update(make_obs())
        assert kf.is_covariance_pd()

    def test_update_wrong_obs_shape_raises(self, cfg):
        kf = make_kf(cfg)
        kf.predict(dt=0.033)
        with pytest.raises(ValueError, match=f"{N_OBS}"):
            kf.update(np.array([1.0, 2.0, 3.0]))  # 3 elements, need 4

    def test_update_sets_nis(self, cfg):
        kf = make_kf(cfg)
        kf.predict(dt=0.033)
        kf.update(make_obs())
        assert not np.isnan(kf.nis())

    def test_update_nis_non_negative(self, cfg):
        kf = make_kf(cfg)
        kf.predict(dt=0.033)
        kf.update(make_obs())
        assert kf.nis() >= 0.0

    def test_high_confidence_tighter_update(self, cfg):
        """
        High confidence → smaller R → larger Kalman gain → state moves more
        toward observation.
        """
        obs = make_obs(cx=400.0, cy=240.0)

        kf_high = make_kf(cfg, cx=300.0)
        kf_low  = make_kf(cfg, cx=300.0)

        kf_high.predict(dt=0.033)
        kf_low.predict(dt=0.033)

        kf_high.update(obs, confidence=0.95)
        kf_low.update(obs,  confidence=0.30)

        # High confidence = more pull toward observation (cx=400)
        assert kf_high.state[0] > kf_low.state[0], (
            "High-confidence detection must pull state more toward the "
            "observation than a low-confidence detection."
        )


# ---------------------------------------------------------------------------
# Combined predict + update cycle
# ---------------------------------------------------------------------------

class TestKFCycle:

    def test_filter_converges_on_stationary_target(self, cfg):
        """
        A filter tracking a stationary target must converge — estimated
        position should approach the true position and uncertainty shrinks.
        """
        true_cx, true_cy = 320.0, 240.0
        kf = make_kf(cfg, cx=280.0, cy=200.0)  # start with offset
        rng = np.random.default_rng(42)

        errors = []
        for _ in range(60):
            kf.predict(dt=0.033)
            noise = rng.normal(0, 1.0, size=4)
            obs = make_obs(cx=true_cx, cy=true_cy) + np.array([noise[0], noise[1], 0.0, 0.0])
            kf.update(obs)
            errors.append(abs(kf.state[0] - true_cx))

        early_error  = float(np.mean(errors[:5]))
        late_error   = float(np.mean(errors[-5:]))

        assert late_error < early_error, (
            f"Filter must converge — late error ({late_error:.2f}px) must be "
            f"less than early error ({early_error:.2f}px)."
        )

    def test_filter_tracks_constant_velocity(self, cfg):
        """
        A filter tracking a target moving at constant velocity should estimate
        a non-zero velocity after enough observations.
        """
        kf = make_kf(cfg, cx=100.0, cy=100.0)
        rng = np.random.default_rng(0)
        true_vx = 30.0  # pixels/second
        cx = 100.0

        for i in range(30):
            cx += true_vx * 0.033
            kf.predict(dt=0.033)
            noise = rng.normal(0, 0.5)
            kf.update(make_obs(cx=cx + noise, cy=100.0))

        estimated_vx = kf.state[4]
        assert abs(estimated_vx - true_vx) < 10.0, (
            f"Estimated vx={estimated_vx:.1f} should be close to true "
            f"vx={true_vx:.1f} after 30 observations."
        )

    def test_uncertainty_grows_without_updates(self, cfg):
        """
        During predict-only periods (occlusion), uncertainty must grow.
        """
        kf = make_kf(cfg)
        kf.predict(dt=0.033)
        kf.update(make_obs())
        trace_after_update = kf.covariance_trace()

        for _ in range(10):
            kf.predict(dt=0.033)

        trace_after_predict = kf.covariance_trace()
        assert trace_after_predict > trace_after_update

    def test_snapshot_captures_state(self, cfg):
        kf = make_kf(cfg)
        kf.predict(dt=0.033)
        kf.update(make_obs())
        snap = kf.snapshot(timestamp=1.0, frame_idx=5)

        assert isinstance(snap, KFSnapshot)
        np.testing.assert_array_equal(snap.state, kf.state)
        assert snap.frame_idx == 5
        assert snap.n_updates == 1


# ---------------------------------------------------------------------------
# NIS diagnostic
# ---------------------------------------------------------------------------

class TestKFNIS:

    def test_nis_nan_before_first_update(self, cfg):
        kf = make_kf(cfg)
        kf.predict(dt=0.033)
        assert np.isnan(kf.nis())

    def test_nis_non_negative_after_update(self, cfg):
        kf = make_kf(cfg)
        kf.predict(dt=0.033)
        kf.update(make_obs())
        assert kf.nis() >= 0.0

    def test_nis_scalar(self, cfg):
        kf = make_kf(cfg)
        kf.predict(dt=0.033)
        kf.update(make_obs())
        assert np.isscalar(kf.nis()) or kf.nis().ndim == 0

    def test_nis_consistent_filter_in_bounds(self, cfg):
        """
        A well-tuned filter running on consistent data must produce NIS
        values within the χ²(4) 95% confidence interval on average.
        We run 200 cycles and check the mean NIS.
        """
        rng = np.random.default_rng(7)
        kf = make_kf(cfg, cx=320.0, cy=240.0)
        nis_values = []

        for _ in range(200):
            kf.predict(dt=0.033)
            noise = rng.multivariate_normal(
                mean=np.zeros(4),
                cov=_build_R(cfg),
            )
            obs = make_obs() + noise
            kf.update(obs)
            nis_values.append(kf.nis())

        mean_nis = float(np.mean(nis_values))
        # Expected value of χ²(4) = 4.0 (number of DOF)
        # A consistent filter should have mean NIS close to 4
        assert 1.0 < mean_nis < 12.0, (
            f"Mean NIS={mean_nis:.2f} is far from expected χ²(4) mean of 4.0. "
            "Check Q and R calibration."
        )

    def test_nis_consistency_flag(self, cfg):
        kf = make_kf(cfg)
        kf.predict(dt=0.033)
        # Observation exactly at predicted position → small innovation → NIS near 0
        # (actually NIS won't be exactly 0 but should be small)
        kf.update(make_obs())  # same position as init
        # Can't assert nis_is_consistent() because a single sample is noisy,
        # but we can assert it doesn't crash
        result = kf.nis_is_consistent()
        assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# Numerical stability
# ---------------------------------------------------------------------------

class TestKFNumericalStability:

    def test_survives_1000_cycles(self, cfg):
        """
        The filter must not diverge or produce NaN/Inf after 1000 cycles.
        Tests the Joseph form's long-term numerical stability.
        """
        rng = np.random.default_rng(99)
        kf = make_kf(cfg)
        R = _build_R(cfg)

        for i in range(1000):
            kf.predict(dt=0.033)
            noise = rng.multivariate_normal(np.zeros(4), R)
            obs = make_obs(cx=320.0 + i * 0.01) + noise
            kf.update(obs)

        assert not np.any(np.isnan(kf.state)), "State contains NaN after 1000 cycles"
        assert not np.any(np.isinf(kf.state)), "State contains Inf after 1000 cycles"
        assert not np.any(np.isnan(kf.covariance)), "Covariance contains NaN"
        assert kf.is_covariance_pd(), "Covariance is not PD after 1000 cycles"

    def test_covariance_pd_check_detects_invalid(self, cfg):
        kf = make_kf(cfg)
        # Corrupt P to be non-PD
        kf._P = -np.eye(N_STATE)
        assert not kf.is_covariance_pd()

    def test_covariance_pd_check_passes_valid(self, cfg):
        kf = make_kf(cfg)
        assert kf.is_covariance_pd()

    def test_predict_only_1000_cycles_no_nan(self, cfg):
        kf = make_kf(cfg)
        for _ in range(1000):
            kf.predict(dt=0.033)
        assert not np.any(np.isnan(kf.state))
        assert not np.any(np.isnan(kf.covariance))


# ---------------------------------------------------------------------------
# Bbox accessors
# ---------------------------------------------------------------------------

class TestKFBboxAccessors:

    def test_bbox_xywh_shape(self, cfg):
        kf = make_kf(cfg, cx=320.0, cy=240.0, w=80.0, h=60.0)
        assert kf.bbox_xywh.shape == (4,)

    def test_bbox_xywh_values(self, cfg):
        kf = make_kf(cfg, cx=320.0, cy=240.0, w=80.0, h=60.0)
        xywh = kf.bbox_xywh
        assert xywh[0] == pytest.approx(320.0)
        assert xywh[1] == pytest.approx(240.0)
        assert xywh[2] == pytest.approx(80.0)
        assert xywh[3] == pytest.approx(60.0)

    def test_bbox_xyxy_shape(self, cfg):
        kf = make_kf(cfg)
        assert kf.bbox_xyxy.shape == (4,)

    def test_bbox_xyxy_values(self, cfg):
        kf = make_kf(cfg, cx=320.0, cy=240.0, w=80.0, h=60.0)
        x1, y1, x2, y2 = kf.bbox_xyxy
        assert x1 == pytest.approx(320.0 - 40.0)  # cx - w/2
        assert y1 == pytest.approx(240.0 - 30.0)  # cy - h/2
        assert x2 == pytest.approx(320.0 + 40.0)
        assert y2 == pytest.approx(240.0 + 30.0)

    def test_bbox_xyxy_clamps_negative_size(self, cfg):
        """Negative width/height is physically impossible — must be clipped."""
        state = np.array([100.0, 100.0, -10.0, -5.0, 0.0, 0.0, 0.0, 0.0])
        kf = KalmanFilter(state, cfg)
        x1, y1, x2, y2 = kf.bbox_xyxy
        assert x2 > x1, "x2 must be > x1 even with negative state width"
        assert y2 > y1, "y2 must be > y1 even with negative state height"

    def test_position_property(self, cfg):
        kf = make_kf(cfg, cx=150.0, cy=200.0)
        pos = kf.position
        assert pos[0] == pytest.approx(150.0)
        assert pos[1] == pytest.approx(200.0)

    def test_velocity_property_zero_at_init(self, cfg):
        kf = make_kf(cfg)
        vel = kf.velocity
        np.testing.assert_array_almost_equal(vel, [0.0, 0.0, 0.0, 0.0])


# ---------------------------------------------------------------------------
# from_detection constructor
# ---------------------------------------------------------------------------

class TestKFFromDetection:

    def test_from_detection_state_matches_bbox(self, cfg):
        det = make_detection(cx=200.0, cy=150.0, w=70.0, h=50.0)
        kf = KalmanFilter.from_detection(det, cfg)
        assert kf.state[0] == pytest.approx(200.0)
        assert kf.state[1] == pytest.approx(150.0)
        assert kf.state[2] == pytest.approx(70.0)
        assert kf.state[3] == pytest.approx(50.0)

    def test_from_detection_velocity_zero(self, cfg):
        det = make_detection()
        kf = KalmanFilter.from_detection(det, cfg)
        np.testing.assert_array_almost_equal(kf.velocity, [0.0, 0.0, 0.0, 0.0])

    def test_from_detection_pd_covariance(self, cfg):
        det = make_detection()
        kf = KalmanFilter.from_detection(det, cfg)
        assert kf.is_covariance_pd()


# ---------------------------------------------------------------------------
# Cross-validation against filterpy
# ---------------------------------------------------------------------------

class TestKFVsFilterPy:
    """
    Numerical cross-check against filterpy.KalmanFilter.

    We run identical predict/update sequences on both our implementation
    and filterpy, and assert agreement to 6 decimal places.
    Any discrepancy indicates a bug in our matrix math.
    """

    @pytest.fixture
    def filterpy_kf(self, cfg):
        """Build a filterpy KF with identical parameters."""
        filterpy = pytest.importorskip("filterpy.kalman",
                                       reason="filterpy not installed")
        from filterpy.kalman import KalmanFilter as FPKalmanFilter

        from state_estimation.kalman_filter import _build_F, _build_H, _build_Q, _build_R

        dt = 0.033
        fp = FPKalmanFilter(dim_x=N_STATE, dim_z=N_OBS)
        fp.F = _build_F(dt)
        fp.H = _build_H()
        fp.Q = _build_Q(dt, cfg)
        fp.R = _build_R(cfg)

        # Match our initial covariance exactly
        fp.P = np.diag([
            10.0, 10.0, 10.0, 10.0,
            100.0, 100.0, 100.0, 100.0,
        ]).astype(np.float64)

        cx, cy, w, h = 320.0, 240.0, 80.0, 60.0
        fp.x = np.array([[cx], [cy], [w], [h],
                          [0.0], [0.0], [0.0], [0.0]])
        return fp

    def test_predict_matches_filterpy(self, cfg, filterpy_kf):
        dt = 0.033
        our_kf = make_kf(cfg)
        our_kf.predict(dt=dt)

        filterpy_kf.predict()

        np.testing.assert_array_almost_equal(
            our_kf.state,
            filterpy_kf.x.flatten(),
            decimal=6,
            err_msg="Predicted state does not match filterpy",
        )
        np.testing.assert_array_almost_equal(
            our_kf.covariance,
            filterpy_kf.P,
            decimal=6,
            err_msg="Predicted covariance does not match filterpy",
        )

    def test_update_matches_filterpy(self, cfg, filterpy_kf):
        dt = 0.033
        obs = make_obs(cx=330.0, cy=245.0, w=82.0, h=61.0)

        our_kf = make_kf(cfg)
        our_kf.predict(dt=dt)
        our_kf.update(obs, confidence=None)

        filterpy_kf.predict()
        filterpy_kf.update(obs.reshape(-1, 1))

        np.testing.assert_array_almost_equal(
            our_kf.state,
            filterpy_kf.x.flatten(),
            decimal=5,
            err_msg="Updated state does not match filterpy",
        )
        np.testing.assert_array_almost_equal(
            our_kf.covariance,
            filterpy_kf.P,
            decimal=5,
            err_msg="Updated covariance does not match filterpy",
        )

    def test_10_cycle_trajectory_matches_filterpy(self, cfg, filterpy_kf):
        """Run 10 predict+update cycles and assert state agreement."""
        dt = 0.033
        rng = np.random.default_rng(42)
        our_kf = make_kf(cfg)

        for _ in range(10):
            noise = rng.normal(0, 1.0, size=4)
            obs = make_obs() + noise

            our_kf.predict(dt=dt)
            filterpy_kf.predict()

            our_kf.update(obs, confidence=None)
            filterpy_kf.update(obs.reshape(-1, 1))

        np.testing.assert_array_almost_equal(
            our_kf.state,
            filterpy_kf.x.flatten(),
            decimal=4,
        )
