"""
Unit tests for TUMDatasetCamera.

A tiny synthetic TUM RGB-D sequence is written to a tmp directory — no
real dataset download, hardware-free, deterministic. The TUM on-disk
format is just text index files + PNGs, so it is cheap to fabricate.

What we validate:
  - CameraFrame contract (typed output, BGR uint8, intrinsics present)
  - Timestamp offset: starts at 0.0, monotonic, preserves true dt
  - EOF signal after the sequence is exhausted
  - Ground-truth accessors: depth_gt() metres + NaN invalids, pose_gt() 4x4
  - RGB<->depth/pose association window
  - Per-Freiburg intrinsics selection
  - Error handling for missing directories / index files
"""

import warnings

import cv2
import numpy as np
import pytest

from perception.camera_interface import CameraFrame, CameraIntrinsics
from perception.tum_dataset_camera import (
    TUMDatasetCamera,
    read_tum_trajectory,
    tum_intrinsics_for,
    _nearest,
    _quat_to_rot,
)


# ---------------------------------------------------------------------------
# Synthetic TUM sequence fixture
# ---------------------------------------------------------------------------

def _make_tum_sequence(
    root,
    name: str = "rgbd_dataset_freiburg1_test",
    n: int = 5,
    depth_factor: float = 5000.0,
    with_depth: bool = True,
    with_gt: bool = True,
    t0: float = 1000.0,
    dt: float = 0.0333,
):
    """Write a minimal but format-correct TUM sequence directory."""
    seq = root / name
    (seq / "rgb").mkdir(parents=True)
    stamps = [t0 + i * dt for i in range(n)]

    rgb_lines = ["# color images\n", "# timestamp filename\n"]
    for i, ts in enumerate(stamps):
        img = np.full((480, 640, 3), (i * 10) % 256, dtype=np.uint8)
        rel = f"rgb/{ts:.6f}.png"
        cv2.imwrite(str(seq / rel), img)
        rgb_lines.append(f"{ts:.6f} {rel}\n")
    (seq / "rgb.txt").write_text("".join(rgb_lines))

    if with_depth:
        (seq / "depth").mkdir()
        d_lines = ["# depth images\n", "# timestamp filename\n"]
        for ts in stamps:
            depth = np.full((480, 640), int(2.0 * depth_factor), dtype=np.uint16)
            depth[0, 0] = 0  # invalid / no-return pixel
            rel = f"depth/{ts:.6f}.png"
            cv2.imwrite(str(seq / rel), depth)
            d_lines.append(f"{ts:.6f} {rel}\n")
        (seq / "depth.txt").write_text("".join(d_lines))

    if with_gt:
        g_lines = ["# ground truth\n", "# timestamp tx ty tz qx qy qz qw\n"]
        for i, ts in enumerate(stamps):
            g_lines.append(f"{ts:.6f} {float(i)} 0.0 0.0 0.0 0.0 0.0 1.0\n")
        (seq / "groundtruth.txt").write_text("".join(g_lines))

    return seq


@pytest.fixture
def tum_seq(tmp_path):
    return _make_tum_sequence(tmp_path)


@pytest.fixture
def cam(tum_seq):
    c = TUMDatasetCamera({}, tum_seq)
    c.open()
    yield c
    c.release()


# ---------------------------------------------------------------------------
# CameraFrame contract
# ---------------------------------------------------------------------------

class TestFrameContract:
    def test_returns_camera_frame(self, cam):
        assert isinstance(cam.get_frame(), CameraFrame)

    def test_image_is_bgr_uint8(self, cam):
        frame = cam.get_frame()
        assert frame.image.dtype == np.uint8
        assert frame.image.shape == (480, 640, 3)

    def test_intrinsics_present(self, cam):
        frame = cam.get_frame()
        assert isinstance(frame.intrinsics, CameraIntrinsics)

    def test_source_id_tags_sequence(self, cam):
        assert cam.get_frame().source_id.startswith("tum:")

    def test_frame_idx_increments(self, cam):
        assert [cam.get_frame().frame_idx for _ in range(3)] == [0, 1, 2]

    def test_context_manager(self, tum_seq):
        with TUMDatasetCamera({}, tum_seq) as c:
            assert c.is_open
            assert c.get_frame() is not None
        assert not c.is_open


# ---------------------------------------------------------------------------
# Timestamps
# ---------------------------------------------------------------------------

class TestTimestamps:
    def test_first_frame_starts_at_zero(self, cam):
        assert cam.get_frame().timestamp == pytest.approx(0.0)

    def test_monotonic_increasing(self, cam):
        ts = [cam.get_frame().timestamp for _ in range(5)]
        assert all(b > a for a, b in zip(ts, ts[1:]))

    def test_preserves_true_dt(self, cam):
        f0, f1 = cam.get_frame(), cam.get_frame()
        assert f1.timestamp - f0.timestamp == pytest.approx(0.0333, abs=1e-4)

    def test_current_timestamp_is_absolute(self, cam):
        cam.get_frame()
        assert cam.current_timestamp == pytest.approx(1000.0)


# ---------------------------------------------------------------------------
# EOF
# ---------------------------------------------------------------------------

class TestEOF:
    def test_returns_none_after_exhaustion(self, cam):
        for _ in range(cam.total_frames):
            assert cam.get_frame() is not None
        assert cam.get_frame() is None

    def test_total_frames(self, cam):
        assert cam.total_frames == 5

    def test_max_frames_truncates(self, tum_seq):
        cam = TUMDatasetCamera({}, tum_seq, max_frames=2)
        cam.open()
        assert cam.total_frames == 2
        assert cam.get_frame() is not None and cam.get_frame() is not None
        assert cam.get_frame() is None


# ---------------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------------

class TestGroundTruth:
    def test_depth_gt_metric_value(self, cam):
        cam.get_frame()
        depth = cam.depth_gt()
        assert depth is not None
        # 2.0 m written; centre pixel is valid.
        assert depth[240, 320] == pytest.approx(2.0)

    def test_depth_gt_invalid_is_nan(self, cam):
        cam.get_frame()
        assert np.isnan(cam.depth_gt()[0, 0])

    def test_pose_gt_is_4x4(self, cam):
        cam.get_frame()
        pose = cam.pose_gt()
        assert pose.shape == (4, 4)
        np.testing.assert_allclose(pose[3], [0, 0, 0, 1])

    def test_pose_gt_translation_tracks_frame(self, cam):
        cam.get_frame()
        cam.get_frame()  # frame 1 -> tx == 1.0
        np.testing.assert_allclose(cam.pose_gt()[:3, 3], [1.0, 0.0, 0.0])

    def test_depth_gt_none_when_no_depth(self, tmp_path):
        seq = _make_tum_sequence(tmp_path, name="rgbd_dataset_freiburg1_nd",
                                 with_depth=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cam = TUMDatasetCamera({}, seq)
            cam.open()
        cam.get_frame()
        assert cam.depth_gt() is None

    def test_pose_gt_none_outside_window(self, tmp_path):
        # Ground truth stamped far from the RGB frames -> no association.
        seq = _make_tum_sequence(tmp_path, name="rgbd_dataset_freiburg1_off",
                                 with_gt=False)
        (seq / "groundtruth.txt").write_text(
            "# gt\n9999.0 0 0 0 0 0 0 1\n"
        )
        cam = TUMDatasetCamera({}, seq)
        cam.open()
        cam.get_frame()
        assert cam.pose_gt() is None


# ---------------------------------------------------------------------------
# Intrinsics
# ---------------------------------------------------------------------------

class TestIntrinsics:
    def test_freiburg1_selected(self, cam):
        assert cam.intrinsics.fx == pytest.approx(517.306408)

    def test_freiburg2_selected(self, tmp_path):
        intr = tum_intrinsics_for(tmp_path / "rgbd_dataset_freiburg2_desk")
        assert intr.fx == pytest.approx(520.908620)

    def test_unknown_name_warns_and_defaults(self, tmp_path):
        with pytest.warns(UserWarning):
            intr = tum_intrinsics_for(tmp_path / "some_random_sequence")
        assert intr.fx == pytest.approx(517.306408)  # freiburg1 fallback


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestErrors:
    def test_missing_directory_raises(self):
        with pytest.raises(FileNotFoundError):
            TUMDatasetCamera({}, "/nonexistent/tum/seq").open()

    def test_missing_rgb_index_raises(self, tmp_path):
        bare = tmp_path / "rgbd_dataset_freiburg1_bare"
        bare.mkdir()
        with pytest.raises(FileNotFoundError):
            TUMDatasetCamera({}, bare).open()

    def test_get_frame_before_open_returns_none(self, tum_seq):
        assert TUMDatasetCamera({}, tum_seq).get_frame() is None


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_read_trajectory(self, tum_seq):
        traj = read_tum_trajectory(tum_seq / "groundtruth.txt")
        assert len(traj) == 5
        ts, pose = traj[0]
        assert ts == pytest.approx(1000.0)
        assert pose.shape == (4, 4)

    def test_quat_identity(self):
        np.testing.assert_allclose(_quat_to_rot(0, 0, 0, 1), np.eye(3))

    def test_quat_90deg_z(self):
        # 90 deg about +z: x-axis -> y-axis.
        R = _quat_to_rot(0, 0, np.sin(np.pi / 4), np.cos(np.pi / 4))
        np.testing.assert_allclose(R @ [1, 0, 0], [0, 1, 0], atol=1e-9)

    def test_nearest_within_window(self):
        stamps = np.array([0.0, 1.0, 2.0])
        assert _nearest(1.01, stamps, 0.02) == 1

    def test_nearest_outside_window(self):
        stamps = np.array([0.0, 1.0, 2.0])
        assert _nearest(1.5, stamps, 0.02) is None

    def test_nearest_empty(self):
        assert _nearest(1.0, np.empty(0), 0.02) is None
