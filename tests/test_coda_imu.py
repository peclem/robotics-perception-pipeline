"""
Unit tests for CODaIMU.

A tiny synthetic CODa sequence (poses/imu + poses/dense_global text
files) is written to a tmp directory — hardware-free, deterministic,
no dataset download.

What we validate:
  - IMUSample contract (accel/gyro shape, values parsed correctly)
  - Timestamps are monotonic offsets sharing the camera's t0
  - get_samples_since() returns exactly the (t, next_frame] window
  - rate_hz, exhaustion, error handling
"""

import warnings

import numpy as np
import pytest

from perception.imu_interface import IMUSample
from perception.coda_imu import CODaIMU, _read_coda_imu


# ---------------------------------------------------------------------------
# Synthetic CODa IMU + pose fixture
# ---------------------------------------------------------------------------

def _make_coda_imu_sequence(
    root,
    name: str = "seq_imu",
    seq_id: str = "0",
    n_frames: int = 5,
    frame_dt: float = 0.1,
    imu_dt: float = 0.05,
    t0: float = 1000.0,
    with_poses: bool = True,
    with_imu: bool = True,
):
    """Write a minimal CODa sequence with poses/imu + poses/dense_global."""
    seq = root / name

    if with_poses:
        pose_dir = seq / "poses" / "dense_global"
        pose_dir.mkdir(parents=True)
        lines = []
        for i in range(n_frames):
            ts = t0 + i * frame_dt
            lines.append(f"{ts:.6f} {float(i)} 0.0 0.0 1.0 0.0 0.0 0.0\n")
        (pose_dir / f"{seq_id}.txt").write_text("".join(lines))

    if with_imu:
        imu_dir = seq / "poses" / "imu"
        imu_dir.mkdir(parents=True)
        # IMU spans the whole frame range at a finer rate.
        n_imu = int((n_frames - 1) * frame_dt / imu_dt) + 2
        lines = []
        for i in range(n_imu):
            ts = t0 + i * imu_dt
            # Distinct, checkable accel/gyro per sample.
            lines.append(
                f"{ts:.6f} {float(i)} 0.0 9.8 0.0 0.0 {0.01 * i:.4f} "
                "0.0 0.0 0.0 1.0\n"
            )
        (imu_dir / f"{seq_id}.txt").write_text("".join(lines))

    return seq


@pytest.fixture
def imu_seq(tmp_path):
    return _make_coda_imu_sequence(tmp_path)


@pytest.fixture
def imu(imu_seq):
    m = CODaIMU(imu_seq)
    m.open()
    yield m
    m.release()


# ---------------------------------------------------------------------------
# Open / load
# ---------------------------------------------------------------------------

class TestOpen:
    def test_loads_samples(self, imu):
        # 5 frames over 0.4 s, IMU at 0.05 s -> 10 samples (0.0 .. 0.45).
        assert imu.total_samples == 10

    def test_rate_hz(self, imu):
        assert imu.rate_hz == pytest.approx(20.0, abs=1e-6)

    def test_repr(self, imu):
        assert "CODaIMU" in repr(imu) and "20.0Hz" in repr(imu)


# ---------------------------------------------------------------------------
# Sample contract
# ---------------------------------------------------------------------------

class TestSamples:
    def test_samples_are_imusamples(self, imu):
        s = imu.get_samples_since(0.0)
        assert all(isinstance(x, IMUSample) for x in s)

    def test_accel_gyro_shape(self, imu):
        s = imu.get_samples_since(0.0)[0]
        assert s.accel.shape == (3,) and s.gyro.shape == (3,)

    def test_values_parsed(self, imu):
        # First window (0, 0.1] holds IMU samples i=1,2 (ts 0.05, 0.10).
        s = imu.get_samples_since(0.0)
        np.testing.assert_allclose(s[0].accel, [1.0, 0.0, 9.8])
        np.testing.assert_allclose(s[1].accel, [2.0, 0.0, 9.8])
        np.testing.assert_allclose(s[1].gyro, [0.0, 0.0, 0.02])

    def test_timestamps_are_offsets(self, imu):
        # t0 subtracted -> first sample sits at 0.0.
        s = imu._samples
        assert s[0].timestamp == pytest.approx(0.0)
        assert s[1].timestamp == pytest.approx(0.05)


# ---------------------------------------------------------------------------
# get_samples_since windowing
# ---------------------------------------------------------------------------

class TestWindow:
    def test_first_window(self, imu):
        # (0.0, 0.1] -> samples at 0.05, 0.10.
        s = imu.get_samples_since(0.0)
        assert [round(x.timestamp, 2) for x in s] == [0.05, 0.10]

    def test_mid_window(self, imu):
        # (0.1, 0.2] -> samples at 0.15, 0.20.
        s = imu.get_samples_since(0.1)
        assert [round(x.timestamp, 2) for x in s] == [0.15, 0.20]

    def test_last_window_takes_remainder(self, imu):
        # t past the final frame -> no upper bound, all remaining samples.
        s = imu.get_samples_since(0.4)
        assert [round(x.timestamp, 2) for x in s] == [0.45]

    def test_window_is_two_samples_for_20hz_imu(self, imu):
        # 20 Hz IMU, 10 Hz frames -> ~2 samples per inter-frame window.
        for t in (0.0, 0.1, 0.2):
            assert len(imu.get_samples_since(t)) == 2

    def test_returns_empty_before_open(self, imu_seq):
        assert CODaIMU(imu_seq).get_samples_since(0.0) == []


class TestInitialSamples:
    def test_window_by_duration(self, imu):
        # IMU at 0.05 s spacing; 0.22 s window -> samples at 0.0..0.20.
        assert [round(x.timestamp, 2) for x in imu.initial_samples(0.22)] == [
            0.0, 0.05, 0.10, 0.15, 0.20
        ]

    def test_shorter_window_excludes_later_samples(self, imu):
        assert [round(x.timestamp, 2) for x in imu.initial_samples(0.12)] == [
            0.0, 0.05, 0.10
        ]

    def test_empty_before_open(self, imu_seq):
        assert CODaIMU(imu_seq).initial_samples(2.0) == []


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestErrors:
    def test_missing_directory_raises(self):
        with pytest.raises(FileNotFoundError):
            CODaIMU("/nonexistent/coda/seq").open()

    def test_missing_imu_dir_raises(self, tmp_path):
        seq = _make_coda_imu_sequence(tmp_path, name="seq_noimu", with_imu=False)
        with pytest.raises(FileNotFoundError):
            CODaIMU(seq).open()

    def test_missing_pose_grid_raises(self, tmp_path):
        seq = _make_coda_imu_sequence(tmp_path, name="seq_nopose", with_poses=False)
        with pytest.raises(FileNotFoundError):
            CODaIMU(seq).open()


# ---------------------------------------------------------------------------
# Parsing helper
# ---------------------------------------------------------------------------

class TestParseHelper:
    def test_read_coda_imu(self, imu_seq):
        rows = _read_coda_imu(imu_seq / "poses" / "imu" / "0.txt")
        assert len(rows) == 10
        ts, accel, gyro = rows[0]
        assert ts == pytest.approx(1000.0)
        np.testing.assert_allclose(accel, [0.0, 0.0, 9.8])
