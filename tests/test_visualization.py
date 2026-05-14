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

    def test_save_path_creates_directory(self, cfg, tmp_path):
        """connect() with rerun_save_path must create parent dirs."""
        save_path = tmp_path / "nested" / "dir" / "rec.rrd"
        cfg.visualization.rerun_enabled  = True
        cfg.visualization.rerun_save_path = str(save_path)
        logger = RerunLogger(cfg)
        try:
            logger.connect()
        except Exception:
            pass  # viewer may not be available in CI
        # Directory must have been created regardless
        assert save_path.parent.exists()


# ---------------------------------------------------------------------------
# TestWSLHostDetect
# ---------------------------------------------------------------------------

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
