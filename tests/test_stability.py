"""
Tests for the stability classification machinery.

TestStabilityLookup        : class-prior table + overrides + unknown class
TestPerStabilityTimeouts   : per-class lost-timeout in SceneGraph._prune
TestMotionOverride         : sustained motion demotes; sustained stillness
                             promotes; hysteresis behaviour
TestStaticPersistence      : STATIC objects survive arbitrarily long timeouts
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from perception.detector import Detection
from tracking.track import Track, TrackState
from world_model.scene_graph import SceneGraph
from world_model.stability import (
    DEFAULT_COCO_STABILITY,
    DEFAULT_TIMEOUTS_S,
    StabilityClass,
    stability_for_class,
    timeout_for_stability,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_ids():
    Track.reset_id_counter()
    yield
    Track.reset_id_counter()


def _base_cfg():
    return {
        "world_model": {"max_history": 10},
        "tracker": {
            "high_thresh": 0.5, "low_thresh": 0.1,
            "new_track_thresh": 0.5, "iou_threshold": 0.3,
            "max_age": 5, "min_hits": 1, "use_ekf": False,
        },
        "kalman_filter": {
            "initial_covariance": {
                "p_position": 10.0, "p_size": 10.0, "p_velocity": 100.0,
            },
            "process_noise": {
                "q_position": 1.0, "q_size": 1.0,
                "q_velocity": 0.1, "q_vel_size": 0.02,
            },
            "measurement_noise": {"r_center": 1.0, "r_size": 1.0},
        },
    }


def _det(cx=150, cy=150, cls_name="person", cls_id=0) -> Detection:
    return Detection(
        bbox_xyxy=np.array([cx-30, cy-40, cx+30, cy+40], dtype=np.float32),
        confidence=0.9, class_id=cls_id, class_name=cls_name,
        frame_idx=0, timestamp=time.monotonic(),
    )


def _make_track(cfg, cx=150, cy=150, cls_name="person", cls_id=0) -> Track:
    t = Track(_det(cx, cy, cls_name, cls_id), cfg)
    t.state = TrackState.CONFIRMED
    return t


# ---------------------------------------------------------------------------
# Stability lookup
# ---------------------------------------------------------------------------

class TestStabilityLookup:

    def test_person_is_dynamic_by_default(self):
        assert stability_for_class("person") == StabilityClass.DYNAMIC

    def test_chair_is_static_by_default(self):
        assert stability_for_class("chair") == StabilityClass.STATIC

    def test_car_is_semi_static_by_default(self):
        assert stability_for_class("car") == StabilityClass.SEMI_STATIC

    def test_unknown_class_defaults_to_semi_static(self):
        assert (
            stability_for_class("ufo") == StabilityClass.SEMI_STATIC
        ), "unknown classes should land in the conservative middle"

    def test_overrides_take_precedence(self):
        ovr = {"person": StabilityClass.STATIC}
        assert stability_for_class("person", ovr) == StabilityClass.STATIC

    def test_every_coco_class_covered(self):
        # The DETAULT_COCO_STABILITY table must have the same 80 names
        # YOLO emits. Spot-check a few categories rather than enumerate.
        for cls in ["person", "chair", "car", "refrigerator",
                    "bottle", "stop sign", "teddy bear"]:
            assert cls in DEFAULT_COCO_STABILITY, f"missing: {cls}"

    def test_timeout_lookup_uses_defaults(self):
        assert timeout_for_stability(StabilityClass.DYNAMIC) == 1.5
        assert timeout_for_stability(StabilityClass.SEMI_STATIC) == 60.0
        assert timeout_for_stability(StabilityClass.STATIC) == float("inf")

    def test_timeout_overrides(self):
        ovr = {StabilityClass.DYNAMIC: 5.0}
        assert timeout_for_stability(StabilityClass.DYNAMIC, ovr) == 5.0
        # Untouched class still uses default
        assert timeout_for_stability(StabilityClass.STATIC, ovr) == float("inf")


# ---------------------------------------------------------------------------
# Per-class lost-timeout in pruning
# ---------------------------------------------------------------------------

class TestPerStabilityTimeouts:

    def test_static_object_not_pruned_after_long_lost(self):
        sg = SceneGraph(_base_cfg())
        # Chair = STATIC by default
        tr = _make_track(_base_cfg(), cls_name="chair", cls_id=56)
        ts = time.monotonic()
        sg.update([tr], [], timestamp=ts)
        tr.state = TrackState.LOST
        # Mark lost and jump time forward by 1 hour.
        sg.update([], [tr], timestamp=ts + 3600.0)
        # STATIC has inf timeout → must survive.
        assert sg.n_objects == 1
        assert sg.get_state(tr.track_id).stability == StabilityClass.STATIC

    def test_dynamic_object_pruned_per_default_timeout(self):
        sg = SceneGraph(_base_cfg())
        tr = _make_track(_base_cfg(), cls_name="person", cls_id=0)
        ts = time.monotonic()
        sg.update([tr], [], timestamp=ts)
        tr.state = TrackState.LOST
        # 2 s > default DYNAMIC timeout of 1.5 s
        sg.update([], [tr], timestamp=ts + 2.0)
        assert sg.n_objects == 0

    def test_semi_static_kept_longer_than_dynamic(self):
        sg = SceneGraph(_base_cfg())
        # car = SEMI_STATIC, default timeout 60 s
        tr = _make_track(_base_cfg(), cls_name="car", cls_id=2)
        ts = time.monotonic()
        sg.update([tr], [], timestamp=ts)
        tr.state = TrackState.LOST
        sg.update([], [tr], timestamp=ts + 30.0)
        assert sg.n_objects == 1   # still under 60 s
        sg.update([], [tr], timestamp=ts + 70.0)
        assert sg.n_objects == 0   # past 60 s

    def test_per_class_override_via_config(self):
        cfg = _base_cfg()
        cfg["stability"] = {"timeouts_s": {"DYNAMIC": 0.05}}
        sg = SceneGraph(cfg)
        tr = _make_track(cfg, cls_name="person")
        ts = time.monotonic()
        sg.update([tr], [], timestamp=ts)
        tr.state = TrackState.LOST
        sg.update([], [tr], timestamp=ts + 0.1)
        assert sg.n_objects == 0

    def test_class_stability_override(self):
        cfg = _base_cfg()
        # Force "person" to STATIC in this deployment.
        cfg["stability"] = {"class_overrides": {"person": "STATIC"}}
        sg = SceneGraph(cfg)
        tr = _make_track(cfg, cls_name="person")
        sg.update([tr], [], timestamp=time.monotonic())
        assert sg.get_state(tr.track_id).stability == StabilityClass.STATIC


# ---------------------------------------------------------------------------
# Motion-based override
# ---------------------------------------------------------------------------

class TestMotionOverride:

    def _config_with_fast_thresholds(self):
        cfg = _base_cfg()
        # Make the override fire quickly enough to test in unit-test time.
        cfg["stability"] = {
            "demote_speed_px_s":  50.0,
            "demote_frames":      3,
            "promote_speed_px_s": 5.0,
            "promote_frames":     3,
        }
        return cfg

    def test_static_demotes_after_sustained_motion(self):
        cfg = self._config_with_fast_thresholds()
        sg = SceneGraph(cfg)
        tr = _make_track(cfg, cls_name="chair", cls_id=56)
        ts = time.monotonic()

        # First update: chair becomes a STATIC ObjectState.
        sg.update([tr], [], timestamp=ts)
        assert sg.get_state(tr.track_id).stability == StabilityClass.STATIC

        # Now force the KF velocity to look like motion. After
        # `demote_frames` updates with speed above threshold, demote.
        for i in range(5):
            tr.kf._x[4] = 100.0   # vx in px/s; above 50 threshold (private write)
            tr.kf._x[5] = 0.0
            sg.update([tr], [], timestamp=ts + (i + 1) * 0.1)

        assert sg.get_state(tr.track_id).stability == StabilityClass.DYNAMIC

    def test_dynamic_promotes_after_sustained_stillness(self):
        cfg = self._config_with_fast_thresholds()
        sg = SceneGraph(cfg)
        tr = _make_track(cfg, cls_name="person")
        ts = time.monotonic()
        sg.update([tr], [], timestamp=ts)
        assert sg.get_state(tr.track_id).stability == StabilityClass.DYNAMIC

        # Force essentially zero velocity for enough frames to trigger.
        for i in range(5):
            tr.kf._x[4] = 0.0
            tr.kf._x[5] = 0.0
            sg.update([tr], [], timestamp=ts + (i + 1) * 0.1)

        assert sg.get_state(tr.track_id).stability == StabilityClass.SEMI_STATIC

    def test_hysteresis_resets_on_single_lapse(self):
        cfg = self._config_with_fast_thresholds()
        sg = SceneGraph(cfg)
        tr = _make_track(cfg, cls_name="chair", cls_id=56)
        ts = time.monotonic()
        sg.update([tr], [], timestamp=ts)

        # 2 frames moving (one short of demote_frames=3), then 1 still.
        # Counter should reset; subsequent 2 more moving frames must not
        # be enough to trip the transition.
        tr.kf._x[4] = 100.0
        sg.update([tr], [], timestamp=ts + 0.1)
        sg.update([tr], [], timestamp=ts + 0.2)
        tr.kf._x[4] = 0.0
        sg.update([tr], [], timestamp=ts + 0.3)
        tr.kf._x[4] = 100.0
        sg.update([tr], [], timestamp=ts + 0.4)
        sg.update([tr], [], timestamp=ts + 0.5)

        # Need 3 *consecutive* moving frames → still STATIC.
        assert sg.get_state(tr.track_id).stability == StabilityClass.STATIC

    def test_semi_static_can_demote_to_dynamic(self):
        cfg = self._config_with_fast_thresholds()
        sg = SceneGraph(cfg)
        # car = SEMI_STATIC by default
        tr = _make_track(cfg, cls_name="car", cls_id=2)
        ts = time.monotonic()
        sg.update([tr], [], timestamp=ts)
        assert sg.get_state(tr.track_id).stability == StabilityClass.SEMI_STATIC
        for i in range(5):
            tr.kf._x[4] = 100.0
            sg.update([tr], [], timestamp=ts + (i + 1) * 0.1)
        assert sg.get_state(tr.track_id).stability == StabilityClass.DYNAMIC


# ---------------------------------------------------------------------------
# STATIC persistence is the headline behavioural contract
# ---------------------------------------------------------------------------

class TestStaticPersistence:

    def test_chair_remembered_long_after_person_forgotten(self):
        """
        Headline scenario for spatial memory: a STATIC object (chair)
        and a DYNAMIC object (person) both go LOST. After several
        seconds, the chair must still be remembered while the person
        is gone — the asymmetry Pass A is built to capture.
        """
        sg = SceneGraph(_base_cfg())
        chair  = _make_track(_base_cfg(), cx=100, cy=100,
                             cls_name="chair", cls_id=56)
        person = _make_track(_base_cfg(), cx=300, cy=300,
                             cls_name="person", cls_id=0)
        ts = time.monotonic()

        sg.update([chair, person], [], timestamp=ts)
        chair.state = TrackState.LOST
        person.state = TrackState.LOST
        sg.update([], [chair, person], timestamp=ts + 5.0)

        ids = sg.object_ids
        assert chair.track_id in ids,  "STATIC chair must persist"
        assert person.track_id not in ids, "DYNAMIC person must be pruned"
