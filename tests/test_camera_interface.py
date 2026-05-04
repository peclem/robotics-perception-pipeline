"""
Unit tests for the CameraInterface module.

Tests use SyntheticCamera exclusively — no hardware required.
All tests are deterministic and run in CI.

What we're validating:
  - Output type contract (always CameraFrame, never raw array)
  - Timestamp monotonicity (critical for downstream KF dt computation)
  - Frame index monotonicity
  - EOF signal (get_frame returns None after num_frames)
  - Intrinsics correctness (camera matrix, bearing vector math)
  - Context manager protocol
"""

import time
import numpy as np
import pytest

from perception.camera_interface import (
    SyntheticCamera,
    VideoFileCamera,
    CameraFrame,
    CameraIntrinsics,
    load_intrinsics,
    _default_intrinsics,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def config():
    return {}


@pytest.fixture
def synthetic_cam(config):
    cam = SyntheticCamera(config, width=640, height=480, num_frames=30, fps=30.0)
    cam.open()
    yield cam
    cam.release()


# ---------------------------------------------------------------------------
# Interface contract
# ---------------------------------------------------------------------------

class TestCameraFrame:
    def test_get_frame_returns_camera_frame(self, synthetic_cam):
        frame = synthetic_cam.get_frame()
        assert isinstance(frame, CameraFrame), (
            "get_frame() must return CameraFrame, not a raw array. "
            "Downstream modules depend on the typed contract."
        )

    def test_image_shape(self, synthetic_cam):
        frame = synthetic_cam.get_frame()
        assert frame.image.ndim == 3
        assert frame.image.shape[2] == 3, "Expected BGR — 3 channels"
        assert frame.image.dtype == np.uint8

    def test_image_dimensions_match_config(self, config):
        cam = SyntheticCamera(config, width=320, height=240, num_frames=5)
        cam.open()
        frame = cam.get_frame()
        cam.release()
        assert frame.height == 240
        assert frame.width == 320

    def test_source_id_set(self, synthetic_cam):
        frame = synthetic_cam.get_frame()
        assert frame.source_id == "synthetic"

    def test_intrinsics_present(self, synthetic_cam):
        frame = synthetic_cam.get_frame()
        assert frame.intrinsics is not None
        assert isinstance(frame.intrinsics, CameraIntrinsics)


# ---------------------------------------------------------------------------
# Timestamp guarantees — critical for Kalman filter dt
# ---------------------------------------------------------------------------

class TestTimestamps:
    def test_timestamps_monotonically_increasing(self, synthetic_cam):
        """
        Timestamps must never go backwards.
        A non-monotonic timestamp corrupts the Kalman filter's dt term,
        causing the prediction step to use a negative time delta.
        """
        timestamps = []
        for _ in range(20):
            frame = synthetic_cam.get_frame()
            if frame is None:
                break
            timestamps.append(frame.timestamp)

        for i in range(1, len(timestamps)):
            assert timestamps[i] > timestamps[i - 1], (
                f"Timestamp went backwards at frame {i}: "
                f"{timestamps[i-1]:.6f} -> {timestamps[i]:.6f}"
            )

    def test_timestamps_are_positive(self, synthetic_cam):
        frame = synthetic_cam.get_frame()
        assert frame.timestamp > 0.0

    def test_dt_consistent_with_fps(self, config):
        """
        For a synthetic camera at 30 fps, frame dt should be ~33.3ms.
        This validates that the timestamp spacing is usable for KF prediction.
        """
        fps = 30.0
        cam = SyntheticCamera(config, fps=fps, num_frames=10)
        cam.open()

        frames = []
        for _ in range(10):
            f = cam.get_frame()
            if f:
                frames.append(f)
        cam.release()

        dts = [frames[i+1].timestamp - frames[i].timestamp
               for i in range(len(frames) - 1)]
        expected_dt = 1.0 / fps
        for dt in dts:
            assert abs(dt - expected_dt) < 1e-6, (
                f"dt={dt:.6f}s expected {expected_dt:.6f}s — "
                "synthetic timestamps must be exact"
            )


# ---------------------------------------------------------------------------
# Frame index
# ---------------------------------------------------------------------------

class TestFrameIndex:
    def test_frame_index_starts_at_zero(self, synthetic_cam):
        frame = synthetic_cam.get_frame()
        assert frame.frame_idx == 0

    def test_frame_index_increments(self, synthetic_cam):
        indices = [synthetic_cam.get_frame().frame_idx for _ in range(5)]
        assert indices == list(range(5))


# ---------------------------------------------------------------------------
# EOF / termination
# ---------------------------------------------------------------------------

class TestEOF:
    def test_returns_none_after_exhaustion(self, config):
        """
        get_frame() must return None when the source is exhausted.
        Callers use this as the loop-termination signal.
        """
        cam = SyntheticCamera(config, num_frames=5)
        cam.open()
        frames = []
        for _ in range(10):  # ask for more than available
            f = cam.get_frame()
            if f is not None:
                frames.append(f)
        cam.release()

        assert len(frames) == 5, (
            f"Expected exactly 5 frames, got {len(frames)}. "
            "get_frame() must return None after num_frames."
        )

    def test_no_exception_after_release(self, config):
        cam = SyntheticCamera(config, num_frames=3)
        cam.open()
        cam.release()
        # Must not raise — return None gracefully
        result = cam.get_frame()
        assert result is None


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------

class TestContextManager:
    def test_context_manager_opens_and_closes(self, config):
        with SyntheticCamera(config, num_frames=3) as cam:
            assert cam.is_open
            frame = cam.get_frame()
            assert frame is not None
        assert not cam.is_open

    def test_context_manager_releases_on_exception(self, config):
        try:
            with SyntheticCamera(config, num_frames=3) as cam:
                _ = cam.get_frame()
                raise ValueError("Simulated pipeline error")
        except ValueError:
            pass
        assert not cam.is_open, (
            "Camera must be released even if the pipeline raises. "
            "Resource leaks on a robot are unacceptable."
        )


# ---------------------------------------------------------------------------
# Intrinsics
# ---------------------------------------------------------------------------

class TestCameraIntrinsics:
    def test_camera_matrix_shape(self):
        intr = _default_intrinsics(640, 480)
        K = intr.camera_matrix()
        assert K.shape == (3, 3)
        assert K[2, 2] == 1.0

    def test_camera_matrix_values(self):
        intr = CameraIntrinsics(fx=800.0, fy=800.0, cx=320.0, cy=240.0,
                                 width=640, height=480)
        K = intr.camera_matrix()
        assert K[0, 0] == 800.0   # fx
        assert K[1, 1] == 800.0   # fy
        assert K[0, 2] == 320.0   # cx
        assert K[1, 2] == 240.0   # cy
        assert K[0, 1] == 0.0     # skew — zero for modern cameras

    def test_bearing_vector_unit_length(self):
        intr = CameraIntrinsics(fx=800.0, fy=800.0, cx=320.0, cy=240.0,
                                 width=640, height=480)
        bearing = intr.pixel_to_bearing(320.0, 240.0)  # principal point
        assert abs(np.linalg.norm(bearing) - 1.0) < 1e-10, (
            "Bearing vector must be unit length — it is a direction, not a position"
        )

    def test_bearing_at_principal_point_is_optical_axis(self):
        intr = CameraIntrinsics(fx=800.0, fy=800.0, cx=320.0, cy=240.0,
                                 width=640, height=480)
        bearing = intr.pixel_to_bearing(320.0, 240.0)
        # At the principal point, bearing should be along [0, 0, 1]
        assert abs(bearing[0]) < 1e-10
        assert abs(bearing[1]) < 1e-10
        assert abs(bearing[2] - 1.0) < 1e-10

    def test_load_intrinsics_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_intrinsics(tmp_path / "nonexistent.yaml")

    def test_load_intrinsics_from_yaml(self, tmp_path):
        yaml_content = """
camera_matrix:
  fx: 923.5
  fy: 923.1
  cx: 637.2
  cy: 361.4
image_size:
  width: 1280
  height: 720
dist_coeffs: [-0.042, 0.018, 0.0003, -0.0002, 0.0]
"""
        p = tmp_path / "intrinsics.yaml"
        p.write_text(yaml_content)
        intr = load_intrinsics(p)

        assert intr.fx == pytest.approx(923.5)
        assert intr.fy == pytest.approx(923.1)
        assert intr.cx == pytest.approx(637.2)
        assert intr.cy == pytest.approx(361.4)
        assert intr.width == 1280
        assert intr.height == 720
        assert len(intr.dist_coeffs) == 5


# ---------------------------------------------------------------------------
# Synthetic camera — ground truth access
# ---------------------------------------------------------------------------

class TestSyntheticGroundTruth:
    def test_ground_truth_bbox_count(self, config):
        cam = SyntheticCamera(config, num_objects=3, num_frames=10)
        cam.open()
        _ = cam.get_frame()
        boxes = cam.get_ground_truth_bboxes()
        cam.release()
        assert len(boxes) == 3

    def test_ground_truth_boxes_within_frame(self, config):
        cam = SyntheticCamera(config, width=640, height=480, num_objects=2, num_frames=10)
        cam.open()
        for _ in range(5):
            cam.get_frame()
        boxes = cam.get_ground_truth_bboxes()
        cam.release()

        for (x1, y1, x2, y2) in boxes:
            assert 0 <= x1 < x2 <= 640, f"Box x out of bounds: {x1}, {x2}"
            assert 0 <= y1 < y2 <= 480, f"Box y out of bounds: {y1}, {y2}"

    def test_deterministic_with_same_seed(self, config):
        """Same seed must produce identical frame sequences — required for reproducible tests."""
        frames_a, frames_b = [], []
        for frames in [frames_a, frames_b]:
            cam = SyntheticCamera(config, seed=42, num_frames=5)
            cam.open()
            for _ in range(5):
                f = cam.get_frame()
                if f:
                    frames.append(f.image.copy())
            cam.release()

        for a, b in zip(frames_a, frames_b):
            assert np.array_equal(a, b), "Same seed must produce identical frames"
