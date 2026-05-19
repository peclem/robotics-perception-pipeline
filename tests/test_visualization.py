"""
Unit tests for visualization — rerun-sdk 0.31.4.

TestEllipsePoints   : covariance ellipse geometry
TestColour          : colour determinism and range
TestDebugVisDraw    : OpenCV annotator contracts
TestRerunLogger     : graceful degradation
TestWSLHostDetect   : host IP resolution
"""

from __future__ import annotations

import math
import textwrap
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from perception.camera_interface import CameraFrame, CameraIntrinsics
from perception.config_loader import load_config
from perception.detector import Detection
from tracking.track import Track, TrackState
from visualization.debug_vis import DebugVisualizer, _bgr
from visualization.rerun_logger import (
    RerunLogger,
    _ellipse_points,
    _resolve_rerun_host,
    _track_colour_rgba,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

MINIMAL_CFG = textwrap.dedent("""
    pipeline:
      log_level: "INFO"
      target_hz: 30
    camera:
      width: 640
      height: 480
      fps: 30
      intrinsics_path: null
    detector:
      confidence_threshold: 0.25
      iou_threshold: 0.45
      max_detections: 100
      img_size: 640
      device: "cpu"
      half_precision: false
    tracker:
      high_thresh: 0.50
      low_thresh: 0.10
      new_track_thresh: 0.50
      iou_threshold: 0.30
      max_age: 30
      min_hits: 1
    kalman_filter:
      initial_covariance:
        p_position: 10.0
        p_size: 10.0
        p_velocity: 100.0
      process_noise:
        q_position: 1.0
        q_size: 1.0
        q_velocity: 0.1
        q_vel_size: 0.02
      measurement_noise:
        r_center: 1.0
        r_size: 1.0
    visualization:
      rerun_enabled: false
      rerun_host: "auto"
      rerun_port: 9876
      rerun_save_path: null
      show_bboxes: true
      show_track_ids: true
      show_velocity: true
      show_covariance_ellipse: true
      show_nis: false
      show_stats_overlay: true
      bbox_thickness: 2
      velocity_arrow_scale: 0.5
""")


@pytest.fixture
def cfg(tmp_path):
    p = tmp_path / "cfg.yaml"
    p.write_text(MINIMAL_CFG)
    return load_config(p)


@pytest.fixture(autouse=True)
def reset_ids():
    Track.reset_id_counter()
    yield
    Track.reset_id_counter()


def make_frame(w=640, h=480) -> CameraFrame:
    intr = CameraIntrinsics(fx=500, fy=500, cx=320, cy=240, width=w, height=h)
    return CameraFrame(
        image=np.zeros((h, w, 3), dtype=np.uint8),
        timestamp=time.monotonic(),
        frame_idx=0,
        intrinsics=intr,
        source_id="test",
    )


def make_detection(
    x1=100, y1=100, x2=200, y2=200, conf=0.9
) -> Detection:
    return Detection(
        bbox_xyxy=np.array([x1, y1, x2, y2], dtype=np.float32),
        confidence=conf, class_id=0, class_name="person",
        frame_idx=0, timestamp=time.monotonic(),
    )


def make_track(cfg, cx=200, cy=200) -> Track:
    det = make_detection(x1=cx-30, y1=cy-40, x2=cx+30, y2=cy+40)
    t   = Track(det, cfg.as_dict())
    t.state = TrackState.CONFIRMED
    return t


# ---------------------------------------------------------------------------
# TestEllipsePoints
# ---------------------------------------------------------------------------

class TestEllipsePoints:

    def test_shape_closed_loop(self):
        pts = _ellipse_points(100, 100, np.eye(2) * 4.0, n_points=36)
        assert pts.shape == (37, 2)

    def test_first_equals_last(self):
        pts = _ellipse_points(100, 100, np.eye(2) * 4.0)
        np.testing.assert_array_almost_equal(pts[0], pts[-1])

    def test_centred_on_cx_cy(self):
        pts = _ellipse_points(320, 240, np.eye(2) * 9.0, n_points=72)
        assert abs(pts[:-1, 0].mean() - 320) < 1.0
        assert abs(pts[:-1, 1].mean() - 240) < 1.0

    def test_isotropic_is_circle(self):
        sigma_sq = 16.0
        pts = _ellipse_points(0, 0, np.eye(2) * sigma_sq,
                               n_points=360, sigma=1.0)
        radii = np.sqrt(pts[:-1, 0]**2 + pts[:-1, 1]**2)
        np.testing.assert_allclose(radii, math.sqrt(sigma_sq), atol=0.1)

    def test_2sigma_larger_than_1sigma(self):
        P   = np.eye(2) * 4.0
        r1  = np.max(np.abs(_ellipse_points(0, 0, P, sigma=1.0)[:-1]))
        r2  = np.max(np.abs(_ellipse_points(0, 0, P, sigma=2.0)[:-1]))
        assert r2 > r1

    def test_singular_P_no_crash(self):
        pts = _ellipse_points(100, 100, np.zeros((2, 2)))
        assert pts.shape[0] > 0
        assert not np.any(np.isnan(pts))

    def test_output_dtype_float32(self):
        pts = _ellipse_points(100, 100, np.eye(2))
        assert pts.dtype == np.float32


# ---------------------------------------------------------------------------
# TestColour
# ---------------------------------------------------------------------------

class TestColour:

    def test_rgba_length(self):
        assert len(_track_colour_rgba(1)) == 4

    def test_rgba_range(self):
        for tid in range(20):
            for c in _track_colour_rgba(tid):
                assert 0 <= c <= 255

    def test_bgr_length(self):
        assert len(_bgr(1)) == 3

    def test_deterministic(self):
        assert _track_colour_rgba(5) == _track_colour_rgba(5)
        assert _bgr(7) == _bgr(7)

    def test_distinct_colours(self):
        colours = {_bgr(i) for i in range(10)}
        assert len(colours) > 5


# ---------------------------------------------------------------------------
# TestDebugVisDraw
# ---------------------------------------------------------------------------

class TestDebugVisDraw:

    def test_returns_ndarray(self, cfg):
        vis = DebugVisualizer(cfg)
        assert isinstance(vis.draw(make_frame(), [], []), np.ndarray)

    def test_output_shape(self, cfg):
        vis = DebugVisualizer(cfg)
        assert vis.draw(make_frame(), [], []).shape == (480, 640, 3)

        assert vis.draw(make_frame(), [], []).dtype == np.uint8

    def test_output_dtype_uint8(self, cfg):
        vis = DebugVisualizer(cfg)
        assert vis.draw(make_frame(), [], []).dtype == np.uint8

    def test_does_not_modify_input(self, cfg):
        vis    = DebugVisualizer(cfg)
        frame  = make_frame()
        before = frame.image.copy()
        vis.draw(frame, [], [])
        np.testing.assert_array_equal(frame.image, before)

    def test_detection_box_changes_output(self, cfg):
        vis    = DebugVisualizer(cfg)
        empty  = vis.draw(make_frame(), [], [])
        with_d = vis.draw(make_frame(), [make_detection()], [])
        assert not np.array_equal(empty, with_d)

    def test_track_box_changes_output(self, cfg):
        vis    = DebugVisualizer(cfg)
        empty  = vis.draw(make_frame(), [], [])
        with_t = vis.draw(make_frame(), [], [make_track(cfg)])
        assert not np.array_equal(empty, with_t)

    def test_multiple_tracks_no_crash(self, cfg):
        vis    = DebugVisualizer(cfg)
        tracks = [make_track(cfg, cx=100 + i * 150, cy=200) for i in range(3)]
        result = vis.draw(make_frame(), [], tracks)
        assert result.shape == (480, 640, 3)

    def test_stats_overlay_writes_pixels(self, cfg):
        vis    = DebugVisualizer(cfg)
        result = vis.draw(make_frame(), [make_detection()], [])
        # Top-left region should be non-black (text was drawn)
        assert result[:50, :300].sum() > 0

    def test_close_no_crash(self, cfg):
        DebugVisualizer(cfg).close()


# ---------------------------------------------------------------------------
# TestRerunLogger — graceful degradation
# ---------------------------------------------------------------------------

class TestRerunLogger:

    def test_disabled_connect_returns_false(self, cfg):
        cfg.visualization.rerun_enabled = False
        logger = RerunLogger(cfg)
        assert logger.connect() is False
        assert not logger.is_ready

    def test_not_ready_before_connect(self, cfg):
        assert not RerunLogger(cfg).is_ready

    def test_log_frame_silent_when_not_ready(self, cfg):
        cfg.visualization.rerun_enabled = False
        logger = RerunLogger(cfg)
        logger.connect()
        logger.log_frame(make_frame(), [], [])   # must not raise

    def test_log_metrics_silent_when_not_ready(self, cfg):
        cfg.visualization.rerun_enabled = False
        logger = RerunLogger(cfg)
        logger.connect()
        logger.log_metrics(5.0, 1.0, 120.0, 0)  # must not raise

    def test_connect_no_raise_when_viewer_absent(self, cfg):
        cfg.visualization.rerun_enabled = True
        cfg.visualization.rerun_host    = "127.0.0.1"
        cfg.visualization.rerun_port    = 19877   # unlikely to be in use
        logger = RerunLogger(cfg)
        try:
            result = logger.connect()
            assert isinstance(result, bool)
        except Exception as exc:
            pytest.fail(f"connect() raised unexpectedly: {exc}")

    def test_close_always_safe(self, cfg):
        logger = RerunLogger(cfg)
        logger.close()   # before connect — must not raise


# ---------------------------------------------------------------------------
# TestRerunShowcaseLayers — voxels / rooms / semantic mask
# ---------------------------------------------------------------------------

class _FakeRR:
    """
    Minimal stand-in for the `rerun` module. Records every log call so
    tests can assert which entity paths were written + what primitives
    were used. Each Rerun primitive constructor (Image, Points3D,
    LineStrips3D, SegmentationImage, AnnotationContext, Clear) returns
    a tagged dict so the test can inspect the payload.
    """
    def __init__(self) -> None:
        self.calls: list = []

    # Timeline
    def set_time(self, *a, **k):  # noqa: D401
        return None

    # Primitive constructors — return tagged dicts the test can inspect.
    def Image(self, arr):                  return {"type": "Image", "shape": arr.shape}
    def DepthImage(self, arr, meter=1.0):  return {"type": "DepthImage",
                                                    "shape": arr.shape,
                                                    "meter": meter,
                                                    "dtype": str(arr.dtype)}
    def Boxes2D(self, **kw):               return {"type": "Boxes2D", **kw}
    def Points3D(self, **kw):              return {"type": "Points3D", **kw}
    def LineStrips3D(self, strips, **kw):  return {"type": "LineStrips3D",
                                                    "n": len(strips),
                                                    "strips": list(strips), **kw}
    def LineStrips2D(self, strips, **kw):  return {"type": "LineStrips2D",
                                                    "n": len(strips),
                                                    "strips": list(strips), **kw}
    def Arrows2D(self, **kw):              return {"type": "Arrows2D", **kw}
    def SegmentationImage(self, arr):      return {"type": "SegmentationImage",
                                                    "shape": arr.shape}
    def AnnotationContext(self, ctx):      return {"type": "AnnotationContext",
                                                    "ctx": ctx}
    def Clear(self, recursive=False):      return {"type": "Clear",
                                                    "recursive": recursive}
    def Scalars(self, *a, **k):            return {"type": "Scalars"}
    def Pinhole(self, **kw):               return {"type": "Pinhole", **kw}
    def Transform3D(self, **kw):           return {"type": "Transform3D", **kw}

    class _BoxFormat:
        XYXY = "XYXY"
    Box2DFormat = _BoxFormat()

    def log(self, path, primitive, **kwargs):
        self.calls.append({"path": path, "primitive": primitive, **kwargs})


def _semantic_mask(mask=None, names=None):
    from perception.semantic_segmenter import SemanticMask
    if mask is None:
        mask = np.array([[0, 1], [1, 0]], dtype=np.int32)
    if names is None:
        names = {0: "road", 1: "person"}
    return SemanticMask(
        mask=mask, class_names=names, dataset="cityscapes",
        timestamp=0.0, frame_idx=0,
    )


def _occupancy_3d(centres=None):
    from world_model.occupancy_3d import Occupancy3D, Occupancy3DParams
    voxels = {}
    if centres is not None:
        for (i, j, k) in centres:
            voxels[(i, j, k)] = np.int8(100)
    return Occupancy3D(occupied_voxels=voxels, params=Occupancy3DParams())


def _rooms(n: int = 2):
    from world_model.room_layer import Room
    out = []
    for r_id in range(1, n + 1):
        poly = np.array([[0, 0], [1 + r_id, 0],
                          [1 + r_id, 1 + r_id], [0, 1 + r_id]], dtype=np.float64)
        out.append(Room(
            room_id=r_id,
            polygon_world=poly,
            centroid=poly.mean(axis=0),
            area_m2=float((1 + r_id) ** 2),
            bbox_world=(0.0, 0.0, float(1 + r_id), float(1 + r_id)),
        ))
    return out


class TestRerunShowcaseLayers:
    """
    Asserts that RerunLogger.log_frame emits the expected entity paths
    and Rerun primitives when occupancy_3d / rooms / semantic_mask are
    provided. Uses _FakeRR to intercept log calls.
    """

    def _ready_logger(self, cfg) -> RerunLogger:
        logger = RerunLogger(cfg)
        logger._rr = _FakeRR()      # type: ignore[attr-defined]
        logger._ready = True        # type: ignore[attr-defined]
        return logger

    def _paths(self, fake):
        return {c["path"] for c in fake.calls}

    def _by_path(self, fake, path):
        return [c for c in fake.calls if c["path"] == path]

    def test_voxels_logged_as_points3d(self, cfg):
        logger = self._ready_logger(cfg)
        sm_occ = _occupancy_3d(centres=[(10, 10, 5), (10, 11, 5)])
        logger.log_frame(make_frame(), [], [], occupancy_3d=sm_occ)
        fake = logger._rr
        calls = self._by_path(fake, "world/occupancy_3d")
        assert calls, "expected world/occupancy_3d to be logged"
        prim = calls[-1]["primitive"]
        assert prim["type"] == "Points3D"
        assert prim["positions"].shape == (2, 3)

    def test_voxels_empty_clears_entity(self, cfg):
        logger = self._ready_logger(cfg)
        logger.log_frame(make_frame(), [], [], occupancy_3d=_occupancy_3d())
        clears = [c for c in logger._rr.calls
                  if c["path"] == "world/occupancy_3d"
                  and c["primitive"]["type"] == "Clear"]
        assert clears, "expected Clear on empty occupancy_3d"

    def test_rooms_logged_as_linestrips3d_plus_labels(self, cfg):
        logger = self._ready_logger(cfg)
        rooms = _rooms(n=2)
        logger.log_frame(make_frame(), [], [], rooms=rooms)
        fake = logger._rr
        # Polygons.
        poly_calls = self._by_path(fake, "world/rooms/polygons")
        assert poly_calls
        assert poly_calls[-1]["primitive"]["type"] == "LineStrips3D"
        assert poly_calls[-1]["primitive"]["n"] == 2
        # Labels as Points3D.
        label_calls = self._by_path(fake, "world/rooms/labels")
        assert label_calls
        assert label_calls[-1]["primitive"]["type"] == "Points3D"
        assert label_calls[-1]["primitive"]["labels"] == ["room_1", "room_2"]

    def test_rooms_empty_clears_entity(self, cfg):
        logger = self._ready_logger(cfg)
        logger.log_frame(make_frame(), [], [], rooms=[])
        clears = [c for c in logger._rr.calls
                  if c["path"] == "world/rooms"
                  and c["primitive"]["type"] == "Clear"]
        assert clears

    def test_semantic_mask_logs_annotation_context_once(self, cfg):
        logger = self._ready_logger(cfg)
        sm = _semantic_mask()
        logger.log_frame(make_frame(), [], [], semantic_mask=sm)
        logger.log_frame(make_frame(), [], [], semantic_mask=sm)
        ctx_calls = [
            c for c in logger._rr.calls
            if c["primitive"]["type"] == "AnnotationContext"
        ]
        # Logged once, not twice.
        assert len(ctx_calls) == 1
        # The id→label table matches.
        assert (0, "road") in ctx_calls[0]["primitive"]["ctx"]
        assert (1, "person") in ctx_calls[0]["primitive"]["ctx"]

    def test_semantic_mask_logs_segmentation_image_each_frame(self, cfg):
        logger = self._ready_logger(cfg)
        sm = _semantic_mask()
        logger.log_frame(make_frame(), [], [], semantic_mask=sm)
        logger.log_frame(make_frame(), [], [], semantic_mask=sm)
        seg_calls = [
            c for c in logger._rr.calls
            if c["primitive"]["type"] == "SegmentationImage"
        ]
        assert len(seg_calls) == 2

    def test_class_signature_change_relogs_context(self, cfg):
        logger = self._ready_logger(cfg)
        sm1 = _semantic_mask(names={0: "road"})
        sm2 = _semantic_mask(names={0: "road", 1: "person"})
        logger.log_frame(make_frame(), [], [], semantic_mask=sm1)
        logger.log_frame(make_frame(), [], [], semantic_mask=sm2)
        ctx_calls = [
            c for c in logger._rr.calls
            if c["primitive"]["type"] == "AnnotationContext"
        ]
        # Different class signatures → two context logs.
        assert len(ctx_calls) == 2

    def test_no_extra_calls_when_all_layers_none(self, cfg):
        logger = self._ready_logger(cfg)
        baseline = len(logger._rr.calls)
        logger.log_frame(make_frame(), [], [])
        after = logger._rr.calls
        # No occupancy / rooms / semantic entities should appear when
        # the caller passed nothing for them.
        paths = self._paths(logger._rr)
        assert "world/occupancy_3d"      not in paths
        assert "world/rooms/polygons"    not in paths
        assert "world/rooms/labels"      not in paths
        assert "world/camera/semantic"   not in paths
        # Same goes for the camera-pose layer when no pose is provided.
        assert "world/ego_trajectory"    not in paths


# ---------------------------------------------------------------------------
# TestRerunCameraPose — Pinhole + Transform3D + ego trajectory
# ---------------------------------------------------------------------------

class TestRerunCameraPose:

    def _pose(self, t=(0.0, 0.0, 0.0)):
        from perception.pose_estimator import CameraPose
        return CameraPose(
            R=np.eye(3),
            t=np.asarray(t, dtype=np.float64),
            timestamp=0.0, frame_idx=0, source="test",
        )

    def _ready_logger(self, cfg) -> RerunLogger:
        logger = RerunLogger(cfg)
        logger._rr = _FakeRR()      # type: ignore[attr-defined]
        logger._ready = True        # type: ignore[attr-defined]
        return logger

    def _by_path(self, fake, path):
        return [c for c in fake.calls if c["path"] == path]

    def _by_type(self, fake, ptype):
        return [c for c in fake.calls if c["primitive"]["type"] == ptype]

    def test_pinhole_logged_once_per_signature(self, cfg):
        logger = self._ready_logger(cfg)
        logger.log_frame(make_frame(), [], [], camera_pose=self._pose())
        logger.log_frame(make_frame(), [], [], camera_pose=self._pose((0.1, 0, 0)))
        pinholes = self._by_type(logger._rr, "Pinhole")
        # Logged once — intrinsics didn't change between frames.
        assert len(pinholes) == 1
        # Mounted under world/camera (so the image entity inherits the
        # 3D positioning Rerun derives from Pinhole + Transform3D).
        assert pinholes[0]["path"] == "world/camera"

    def test_resolution_change_relogs_pinhole(self, cfg):
        logger = self._ready_logger(cfg)
        logger.log_frame(make_frame(w=640, h=480), [], [], camera_pose=self._pose())
        logger.log_frame(make_frame(w=320, h=240), [], [], camera_pose=self._pose())
        pinholes = self._by_type(logger._rr, "Pinhole")
        assert len(pinholes) == 2

    def test_transform3d_logged_each_frame(self, cfg):
        logger = self._ready_logger(cfg)
        logger.log_frame(make_frame(), [], [], camera_pose=self._pose((0.0, 0, 0)))
        logger.log_frame(make_frame(), [], [], camera_pose=self._pose((0.1, 0, 0)))
        logger.log_frame(make_frame(), [], [], camera_pose=self._pose((0.2, 0, 0)))
        transforms = self._by_type(logger._rr, "Transform3D")
        assert len(transforms) == 3
        # Translations match what we passed in (within float32 rounding).
        ts = np.array(
            [t["primitive"]["translation"] for t in transforms],
            dtype=np.float64,
        )
        np.testing.assert_allclose(
            ts, [[0.0, 0, 0], [0.1, 0, 0], [0.2, 0, 0]], atol=1e-6,
        )

    def test_trajectory_logged_after_two_frames(self, cfg):
        logger = self._ready_logger(cfg)
        # First frame: only one point in the buffer → no polyline yet.
        logger.log_frame(make_frame(), [], [], camera_pose=self._pose((0, 0, 0)))
        assert not self._by_path(logger._rr, "world/ego_trajectory")
        # Second frame: two points → polyline emitted.
        logger.log_frame(make_frame(), [], [], camera_pose=self._pose((1, 0, 0)))
        traj_calls = self._by_path(logger._rr, "world/ego_trajectory")
        assert traj_calls
        prim = traj_calls[-1]["primitive"]
        assert prim["type"] == "LineStrips3D"
        assert prim["n"] == 1   # one polyline

    def test_trajectory_deque_bounded(self, cfg):
        logger = self._ready_logger(cfg)
        # 5 calls — deque should hold all 5 (maxlen=2000).
        for k in range(5):
            logger.log_frame(make_frame(), [], [],
                             camera_pose=self._pose((float(k), 0, 0)))
        assert len(logger._ego_trajectory) == 5

    def test_no_pose_means_no_camera_entities(self, cfg):
        logger = self._ready_logger(cfg)
        logger.log_frame(make_frame(), [], [])  # no camera_pose
        assert not self._by_type(logger._rr, "Pinhole")
        assert not self._by_type(logger._rr, "Transform3D")
        assert not [c for c in logger._rr.calls
                    if c["path"] == "world/ego_trajectory"]


# ---------------------------------------------------------------------------
# TestRerunWorldTracks — world-frame markers for tracked objects
# ---------------------------------------------------------------------------

class TestRerunWorldTracks:

    def _obj(self, track_id=1, position_world=None, persistent_id=None,
             class_name="person"):
        from world_model.object_state import ObjectState
        return ObjectState(
            track_id=track_id, class_id=0, class_name=class_name,
            position=np.array([0.0, 0.0]),
            covariance=np.eye(8),
            velocity=np.zeros(4),
            score=0.9, last_seen=0.0, n_updates=1,
            position_world=(
                np.asarray(position_world, dtype=np.float64)
                if position_world is not None else None
            ),
            persistent_id=persistent_id,
            max_history=1,
        )

    def _ready_logger(self, cfg) -> RerunLogger:
        logger = RerunLogger(cfg)
        logger._rr = _FakeRR()      # type: ignore[attr-defined]
        logger._ready = True        # type: ignore[attr-defined]
        return logger

    def _by_path(self, fake, path):
        return [c for c in fake.calls if c["path"] == path]

    def test_emits_points3d_at_world_positions(self, cfg):
        logger = self._ready_logger(cfg)
        objs = [
            self._obj(track_id=1, position_world=[1.0, 2.0, 0.5]),
            self._obj(track_id=2, position_world=[-1.0, 0.0, 0.5],
                       class_name="car"),
        ]
        logger.log_frame(make_frame(), [], [], scene_objects=objs)
        calls = self._by_path(logger._rr, "world/scene/objects")
        assert calls and calls[-1]["primitive"]["type"] == "Points3D"
        positions = calls[-1]["primitive"]["positions"]
        np.testing.assert_allclose(positions, [[1, 2, 0.5], [-1, 0, 0.5]],
                                   atol=1e-6)

    def test_persistent_id_appears_in_label(self, cfg):
        logger = self._ready_logger(cfg)
        objs = [
            self._obj(track_id=7, position_world=[0, 0, 0],
                       persistent_id=42),
            self._obj(track_id=8, position_world=[1, 0, 0]),  # no p_id
        ]
        logger.log_frame(make_frame(), [], [], scene_objects=objs)
        labels = self._by_path(logger._rr, "world/scene/objects")[-1] \
            ["primitive"]["labels"]
        assert "#7" in labels[0] and "p:42" in labels[0]
        assert "#8" in labels[1] and "p:" not in labels[1]

    def test_objects_without_position_world_skipped(self, cfg):
        logger = self._ready_logger(cfg)
        objs = [
            self._obj(track_id=1, position_world=None),         # no world pos
            self._obj(track_id=2, position_world=[5.0, 0, 0]),  # has it
        ]
        logger.log_frame(make_frame(), [], [], scene_objects=objs)
        calls = self._by_path(logger._rr, "world/scene/objects")
        assert calls
        prim = calls[-1]["primitive"]
        assert prim["type"] == "Points3D"
        assert prim["positions"].shape == (1, 3)   # only obj 2
        assert prim["labels"] == ["#2 person"]

    def test_no_objects_emits_clear(self, cfg):
        logger = self._ready_logger(cfg)
        logger.log_frame(make_frame(), [], [], scene_objects=[])
        clears = [c for c in self._by_path(logger._rr, "world/scene/objects")
                  if c["primitive"]["type"] == "Clear"]
        assert clears

    def test_all_objects_lack_position_world_emits_clear(self, cfg):
        logger = self._ready_logger(cfg)
        objs = [
            self._obj(track_id=1, position_world=None),
            self._obj(track_id=2, position_world=None),
        ]
        logger.log_frame(make_frame(), [], [], scene_objects=objs)
        clears = [c for c in self._by_path(logger._rr, "world/scene/objects")
                  if c["primitive"]["type"] == "Clear"]
        assert clears

    def test_no_scene_objects_arg_means_no_call(self, cfg):
        logger = self._ready_logger(cfg)
        logger.log_frame(make_frame(), [], [])
        assert not self._by_path(logger._rr, "world/scene/objects")


# ---------------------------------------------------------------------------
# TestRerunWorldMap — "remembered vs visible" long-term spatial memory layer
# ---------------------------------------------------------------------------

class TestRerunWorldMap:

    def _world_map(self, entries):
        """Build a WorldMap pre-populated with the given entries."""
        from world_model.world_map import WorldMap
        from world_model.stability import StabilityClass
        wm = WorldMap()
        for class_name, pos, emb in entries:
            wm.insert(
                class_name=class_name, class_id=0,
                position_world=np.asarray(pos, dtype=np.float64),
                embedding=emb, stability=StabilityClass.STATIC,
                timestamp=0.0,
            )
        return wm

    def _obj(self, track_id=1, position_world=None, persistent_id=None,
             class_name="chair"):
        from world_model.object_state import ObjectState
        return ObjectState(
            track_id=track_id, class_id=0, class_name=class_name,
            position=np.array([0.0, 0.0]),
            covariance=np.eye(8),
            velocity=np.zeros(4),
            score=0.9, last_seen=0.0, n_updates=1,
            position_world=(
                np.asarray(position_world, dtype=np.float64)
                if position_world is not None else None
            ),
            persistent_id=persistent_id,
            max_history=1,
        )

    def _ready_logger(self, cfg) -> RerunLogger:
        logger = RerunLogger(cfg)
        logger._rr = _FakeRR()      # type: ignore[attr-defined]
        logger._ready = True        # type: ignore[attr-defined]
        return logger

    def _by_path(self, fake, path):
        return [c for c in fake.calls if c["path"] == path]

    def test_visible_and_remembered_split(self, cfg):
        logger = self._ready_logger(cfg)
        wm = self._world_map([
            ("chair", [1.0, 0.0, 0.5], None),   # persistent_id=1
            ("chair", [3.0, 0.0, 0.5], None),   # persistent_id=2
            ("chair", [5.0, 0.0, 0.5], None),   # persistent_id=3
        ])
        # Only persistent_id=2 is "visible" this frame.
        objs = [self._obj(track_id=10, position_world=[3.0, 0.0, 0.5],
                          persistent_id=2)]
        logger.log_frame(make_frame(), [], [],
                         scene_objects=objs, world_map=wm)

        vis = self._by_path(logger._rr, "world/world_map/visible")[-1]
        rem = self._by_path(logger._rr, "world/world_map/remembered")[-1]
        assert vis["primitive"]["type"] == "Points3D"
        assert rem["primitive"]["type"] == "Points3D"
        # 1 visible, 2 remembered.
        assert vis["primitive"]["positions"].shape == (1, 3)
        assert rem["primitive"]["positions"].shape == (2, 3)
        np.testing.assert_allclose(
            vis["primitive"]["positions"], [[3.0, 0.0, 0.5]], atol=1e-6,
        )

    def test_labels_carry_persistent_id_and_obs_count(self, cfg):
        logger = self._ready_logger(cfg)
        wm = self._world_map([("chair", [0.0, 0.0, 0.0], None)])
        logger.log_frame(make_frame(), [], [], world_map=wm)
        # No scene_objects → everything is "remembered".
        rem = self._by_path(logger._rr, "world/world_map/remembered")[-1]
        labels = rem["primitive"]["labels"]
        assert labels == ["p:1 chair n=1"]

    def test_empty_world_map_emits_clears(self, cfg):
        logger = self._ready_logger(cfg)
        wm = self._world_map([])
        logger.log_frame(make_frame(), [], [], world_map=wm)
        for path in ("world/world_map/visible", "world/world_map/remembered"):
            clears = [c for c in self._by_path(logger._rr, path)
                      if c["primitive"]["type"] == "Clear"]
            assert clears, f"expected Clear on {path}"

    def test_all_visible_clears_remembered_layer(self, cfg):
        logger = self._ready_logger(cfg)
        wm = self._world_map([("chair", [0.0, 0.0, 0.0], None)])
        objs = [self._obj(track_id=10, position_world=[0.0, 0.0, 0.0],
                          persistent_id=1)]
        logger.log_frame(make_frame(), [], [],
                         scene_objects=objs, world_map=wm)
        rem = self._by_path(logger._rr, "world/world_map/remembered")[-1]
        # Nothing remembered → Clear.
        assert rem["primitive"]["type"] == "Clear"
        vis = self._by_path(logger._rr, "world/world_map/visible")[-1]
        assert vis["primitive"]["type"] == "Points3D"

    def test_objects_without_persistent_id_count_as_not_visible(self, cfg):
        logger = self._ready_logger(cfg)
        wm = self._world_map([("chair", [0.0, 0.0, 0.0], None)])
        # Scene has an object but with no persistent_id — can't match.
        objs = [self._obj(track_id=10, position_world=[0.0, 0.0, 0.0],
                          persistent_id=None)]
        logger.log_frame(make_frame(), [], [],
                         scene_objects=objs, world_map=wm)
        rem = self._by_path(logger._rr, "world/world_map/remembered")[-1]
        assert rem["primitive"]["type"] == "Points3D"
        assert rem["primitive"]["positions"].shape == (1, 3)

    def test_no_world_map_arg_means_no_call(self, cfg):
        logger = self._ready_logger(cfg)
        logger.log_frame(make_frame(), [], [])
        paths = {c["path"] for c in logger._rr.calls}
        assert "world/world_map/visible"    not in paths
        assert "world/world_map/remembered" not in paths


# ---------------------------------------------------------------------------
# TestRerunTrackHistory — past-position polylines per track
# ---------------------------------------------------------------------------

class TestRerunTrackHistory:

    def _ready_logger(self, cfg) -> RerunLogger:
        logger = RerunLogger(cfg)
        logger._rr = _FakeRR()      # type: ignore[attr-defined]
        logger._ready = True        # type: ignore[attr-defined]
        return logger

    def _by_path(self, fake, path):
        return [c for c in fake.calls if c["path"] == path]

    def _seed_history(self, track, points):
        """Push synthetic KFSnapshot entries onto track.history."""
        from state_estimation.kalman_filter import KFSnapshot
        for k, (x, y) in enumerate(points):
            state = np.zeros(8, dtype=np.float64)
            state[0], state[1] = float(x), float(y)
            track.history.append(KFSnapshot(
                timestamp=float(k), frame_idx=k,
                state=state, covariance=np.eye(8),
                nis=float("nan"), n_updates=k + 1,
            ))

    def test_polyline_emitted_for_history_ge_2(self, cfg):
        logger = self._ready_logger(cfg)
        track = make_track(cfg)
        self._seed_history(track, [(10, 10), (20, 20), (30, 25)])
        logger.log_frame(make_frame(), [], [track])
        hist = self._by_path(logger._rr, "world/tracks/history")[-1]
        prim = hist["primitive"]
        assert prim["type"] == "LineStrips2D"
        assert prim["n"] == 1

    def test_single_point_history_skipped(self, cfg):
        logger = self._ready_logger(cfg)
        track = make_track(cfg)
        self._seed_history(track, [(10, 10)])
        logger.log_frame(make_frame(), [], [track])
        hist = self._by_path(logger._rr, "world/tracks/history")[-1]
        # Single-point trail is not meaningful → Clear.
        assert hist["primitive"]["type"] == "Clear"

    def test_history_capped_at_max_points(self, cfg):
        # Default cap is 32 — feed 50, expect the polyline to carry the
        # most-recent 32 points.
        logger = self._ready_logger(cfg)
        track = make_track(cfg)
        self._seed_history(track, [(k, k) for k in range(50)])
        logger.log_frame(make_frame(), [], [track])
        hist = self._by_path(logger._rr, "world/tracks/history")[-1]
        prim = hist["primitive"]
        assert prim["type"] == "LineStrips2D"
        assert prim["n"] == 1
        strip = prim["strips"][0]
        assert strip.shape == (32, 2)
        # First point in the strip is index 18 (50 - 32) — confirms we
        # kept the most recent, not the oldest.
        np.testing.assert_allclose(strip[0],  [18.0, 18.0], atol=1e-6)
        np.testing.assert_allclose(strip[-1], [49.0, 49.0], atol=1e-6)

    def test_disabled_flag_skips_layer(self, cfg):
        # Disable show_track_history; layer should not be emitted at all
        # — and no Clear either, since the gating short-circuits before
        # reaching _log_track_history.
        cfg.visualization.show_track_history = False
        logger = self._ready_logger(cfg)
        track = make_track(cfg)
        self._seed_history(track, [(10, 10), (20, 20)])
        logger.log_frame(make_frame(), [], [track])
        # Should be no history-path calls at all.
        assert not self._by_path(logger._rr, "world/tracks/history")

    def test_no_tracks_clears_history_entity(self, cfg):
        logger = self._ready_logger(cfg)
        logger.log_frame(make_frame(), [], [])
        clears = [c for c in self._by_path(logger._rr, "world/tracks/history")
                  if c["primitive"]["type"] == "Clear"]
        assert clears, "expected Clear on world/tracks/history when no tracks"


# ---------------------------------------------------------------------------
# TestRerunDepthMap — dense depth map heatmap at world/camera/depth
# ---------------------------------------------------------------------------

class TestRerunDepthMap:

    def _ready_logger(self, cfg) -> RerunLogger:
        logger = RerunLogger(cfg)
        logger._rr = _FakeRR()      # type: ignore[attr-defined]
        logger._ready = True        # type: ignore[attr-defined]
        return logger

    def _by_path(self, fake, path):
        return [c for c in fake.calls if c["path"] == path]

    def test_depth_map_logged_as_depthimage_in_metres(self, cfg):
        logger = self._ready_logger(cfg)
        dm = np.full((4, 6), 2.5, dtype=np.float32)
        logger.log_frame(make_frame(), [], [], depth_map=dm)
        calls = self._by_path(logger._rr, "world/camera/depth")
        assert calls
        prim = calls[-1]["primitive"]
        assert prim["type"] == "DepthImage"
        assert prim["shape"] == (4, 6)
        assert prim["meter"] == 1.0
        assert prim["dtype"] == "float32"

    def test_non_float32_input_is_coerced(self, cfg):
        logger = self._ready_logger(cfg)
        dm = (np.ones((3, 3)) * 1500).astype(np.uint16)   # mm-style ints
        logger.log_frame(make_frame(), [], [], depth_map=dm)
        prim = self._by_path(logger._rr, "world/camera/depth")[-1]["primitive"]
        assert prim["dtype"] == "float32"

    def test_no_depth_map_arg_means_no_call(self, cfg):
        logger = self._ready_logger(cfg)
        logger.log_frame(make_frame(), [], [])
        assert not self._by_path(logger._rr, "world/camera/depth")


# ---------------------------------------------------------------------------
# TestRerunOccupancyGrid2D — top-down 2D occupancy grid layer
# ---------------------------------------------------------------------------

class TestRerunOccupancyGrid2D:

    def _ready_logger(self, cfg) -> RerunLogger:
        logger = RerunLogger(cfg)
        logger._rr = _FakeRR()      # type: ignore[attr-defined]
        logger._ready = True        # type: ignore[attr-defined]
        return logger

    def _by_path(self, fake, path):
        return [c for c in fake.calls if c["path"] == path]

    def _grid(self, occupied_cells, h=8, w=10, value=100):
        """Build a (H, W) int8 grid with the given (r, c) cells set."""
        g = np.zeros((h, w), dtype=np.int8)
        for r, c in occupied_cells:
            g[r, c] = value
        return g

    def _params(self, resolution_m=0.5, origin_x_m=-2.5, origin_y_m=-2.0):
        from world_model.occupancy_grid import OccupancyGridParams
        return OccupancyGridParams(
            resolution_m=resolution_m,
            size_x_m=resolution_m * 10, size_y_m=resolution_m * 8,
            origin_x_m=origin_x_m, origin_y_m=origin_y_m,
        )

    def test_occupied_cells_become_points3d_at_z0(self, cfg):
        logger = self._ready_logger(cfg)
        # One cell at (row=2, col=3); origin (-2.5, -2.0), res=0.5 →
        # world (x = -2.5 + 3.5*0.5 = -0.75, y = -2.0 + 2.5*0.5 = -0.75, z = 0)
        grid = self._grid([(2, 3)])
        logger.log_frame(make_frame(), [], [],
                         occupancy_grid_2d=(grid, self._params()))
        calls = self._by_path(logger._rr, "world/occupancy_grid")
        assert calls
        prim = calls[-1]["primitive"]
        assert prim["type"] == "Points3D"
        positions = prim["positions"]
        assert positions.shape == (1, 3)
        np.testing.assert_allclose(positions[0], [-0.75, -0.75, 0.0], atol=1e-6)

    def test_below_threshold_cells_dropped(self, cfg):
        logger = self._ready_logger(cfg)
        # Cells at 30 (free-ish) shouldn't render — only ≥50.
        grid = self._grid([(2, 3)], value=30)
        logger.log_frame(make_frame(), [], [],
                         occupancy_grid_2d=(grid, self._params()))
        clears = [c for c in self._by_path(logger._rr, "world/occupancy_grid")
                  if c["primitive"]["type"] == "Clear"]
        assert clears

    def test_empty_grid_emits_clear(self, cfg):
        logger = self._ready_logger(cfg)
        logger.log_frame(make_frame(), [], [],
                         occupancy_grid_2d=(self._grid([]),
                                            self._params()))
        clears = [c for c in self._by_path(logger._rr, "world/occupancy_grid")
                  if c["primitive"]["type"] == "Clear"]
        assert clears

    def test_radius_matches_half_resolution(self, cfg):
        logger = self._ready_logger(cfg)
        grid = self._grid([(0, 0)])
        logger.log_frame(make_frame(), [], [],
                         occupancy_grid_2d=(grid,
                                            self._params(resolution_m=0.5)))
        prim = self._by_path(logger._rr, "world/occupancy_grid")[-1]["primitive"]
        assert prim["radii"] == 0.25

    def test_multiple_cells_each_emit_point(self, cfg):
        logger = self._ready_logger(cfg)
        grid = self._grid([(0, 0), (1, 1), (2, 2)])
        logger.log_frame(make_frame(), [], [],
                         occupancy_grid_2d=(grid, self._params()))
        prim = self._by_path(logger._rr, "world/occupancy_grid")[-1]["primitive"]
        assert prim["positions"].shape == (3, 3)

    def test_no_occupancy_grid_arg_means_no_call(self, cfg):
        logger = self._ready_logger(cfg)
        logger.log_frame(make_frame(), [], [])
        assert not self._by_path(logger._rr, "world/occupancy_grid")


class TestWSLHostDetect:

    def test_explicit_host_unchanged(self):
        assert _resolve_rerun_host("192.168.1.5") == "192.168.1.5"

    def test_localhost_unchanged(self):
        assert _resolve_rerun_host("127.0.0.1") == "127.0.0.1"

    def test_auto_returns_string(self):
        result = _resolve_rerun_host("auto")
        assert isinstance(result, str) and len(result) > 0

    def test_auto_fallback_on_missing_file(self):
        with patch("builtins.open", side_effect=FileNotFoundError):
            assert _resolve_rerun_host("auto") == "127.0.0.1"

    def test_auto_parses_nameserver(self, tmp_path):
        rc = tmp_path / "resolv.conf"
        rc.write_text("# comment\nnameserver 172.22.160.1\n")
        with patch("builtins.open", return_value=open(rc)):
            assert _resolve_rerun_host("auto") == "172.22.160.1"
