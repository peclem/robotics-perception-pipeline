"""
Unit tests for the benchmark script.

Tests use synthetic data — no MOT17 dataset required.
Validates: MOTChallenge output format, seqinfo parsing,
frame wrapper, result file writing.
"""

from __future__ import annotations

import sys
import textwrap
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.benchmark import (
    make_frame_from_image,
    write_results_markdown,
)
from perception.camera_interface import CameraFrame


# ---------------------------------------------------------------------------
# TestFrameWrapper
# ---------------------------------------------------------------------------

class TestFrameWrapper:

    def test_returns_camera_frame(self):
        img    = np.zeros((480, 640, 3), dtype=np.uint8)
        frame  = make_frame_from_image(img, 0, 1.0, 640, 480)
        assert isinstance(frame, CameraFrame)

    def test_image_shape_preserved(self):
        img   = np.zeros((480, 640, 3), dtype=np.uint8)
        frame = make_frame_from_image(img, 0, 1.0, 640, 480)
        assert frame.image.shape == (480, 640, 3)

    def test_frame_idx_set(self):
        img   = np.zeros((480, 640, 3), dtype=np.uint8)
        frame = make_frame_from_image(img, 42, 1.0, 640, 480)
        assert frame.frame_idx == 42

    def test_timestamp_set(self):
        img   = np.zeros((480, 640, 3), dtype=np.uint8)
        frame = make_frame_from_image(img, 0, 3.14, 640, 480)
        assert frame.timestamp == pytest.approx(3.14)

    def test_intrinsics_dimensions(self):
        img   = np.zeros((720, 1280, 3), dtype=np.uint8)
        frame = make_frame_from_image(img, 0, 0.0, 1280, 720)
        assert frame.intrinsics.width  == 1280
        assert frame.intrinsics.height == 720

    def test_source_id_is_mot17(self):
        img   = np.zeros((480, 640, 3), dtype=np.uint8)
        frame = make_frame_from_image(img, 0, 0.0, 640, 480)
        assert frame.source_id == "mot17"


# ---------------------------------------------------------------------------
# TestResultFormat
# ---------------------------------------------------------------------------

class TestResultFormat:

    def test_results_file_format(self, tmp_path):
        """
        MOTChallenge result format:
        frame_id, track_id, x1, y1, w, h, conf, -1, -1, -1
        """
        out = tmp_path / "MOT17-02-DPM.txt"

        # Simulate what run_sequence writes
        results = [
            (1, 1, 100.0, 200.0, 50.0, 80.0, 0.9),
            (1, 2, 300.0, 150.0, 60.0, 90.0, 0.8),
            (2, 1, 105.0, 202.0, 50.0, 80.0, 0.85),
        ]
        with open(out, "w") as f:
            for frame_id, track_id, x1, y1, w, h, conf in results:
                f.write(
                    f"{frame_id},{track_id},"
                    f"{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},"
                    f"{conf:.4f},-1,-1,-1\n"
                )

        lines = out.read_text().strip().split("\n")
        assert len(lines) == 3

        # Parse first line
        parts = lines[0].split(",")
        assert len(parts) == 10
        assert int(parts[0]) == 1       # frame_id
        assert int(parts[1]) == 1       # track_id
        assert float(parts[6]) == pytest.approx(0.9, abs=1e-4)  # conf
        assert parts[7] == "-1"         # unused fields
        assert parts[8] == "-1"
        assert parts[9] == "-1"

    def test_results_file_1_indexed_frames(self, tmp_path):
        """MOT17 is 1-indexed — frame 0 must never appear."""
        out = tmp_path / "test.txt"
        with open(out, "w") as f:
            # frame_id=1 is the first frame
            f.write("1,1,100.00,100.00,50.00,80.00,0.9000,-1,-1,-1\n")

        lines = out.read_text().strip().split("\n")
        frame_id = int(lines[0].split(",")[0])
        assert frame_id >= 1, "Frame IDs must be 1-indexed in MOTChallenge format"


# ---------------------------------------------------------------------------
# TestSeqinfo
# ---------------------------------------------------------------------------

class TestSeqinfoParser:

    def test_parse_seqinfo(self, tmp_path):
        """Parse a synthetic seqinfo.ini file."""
        from scripts.benchmark import read_seqinfo

        ini = textwrap.dedent("""
            [Sequence]
            name=MOT17-02-DPM
            imDir=img1
            frameRate=30
            seqLength=600
            imWidth=1920
            imHeight=1080
            imExt=.jpg
            motChallenge=1
        """)
        seq_dir = tmp_path / "MOT17-02-DPM"
        seq_dir.mkdir()
        (seq_dir / "seqinfo.ini").write_text(ini)

        info = read_seqinfo(seq_dir)
        assert info["name"]     == "MOT17-02-DPM"
        assert info["fps"]      == pytest.approx(30.0)
        assert info["n_frames"] == 600
        assert info["width"]    == 1920
        assert info["height"]   == 1080
        assert info["img_dir"]  == "img1"

    def test_seqinfo_missing_raises(self, tmp_path):
        from scripts.benchmark import read_seqinfo
        with pytest.raises(FileNotFoundError, match="seqinfo.ini"):
            read_seqinfo(tmp_path / "nonexistent_seq")


# ---------------------------------------------------------------------------
# TestMarkdownOutput
# ---------------------------------------------------------------------------

class TestMarkdownOutput:

    def test_write_results_markdown_creates_file(self, tmp_path):
        out = tmp_path / "docs" / "benchmark_results.md"
        seq_stats = [
            {"seq_name": "MOT17-02-DPM", "n_frames": 600,
             "hz": 45.2, "n_tracks": 12},
        ]
        write_results_markdown(
            metrics={"MOTA": 52.3, "MOTP": 76.4, "IDF1": 58.1, "HOTA": 48.2},
            seq_stats=seq_stats,
            out_path=out,
            config_name="config/default.yaml",
        )
        assert out.exists()

    def test_write_results_contains_metrics(self, tmp_path):
        out = tmp_path / "results.md"
        write_results_markdown(
            metrics={"MOTA": 52.3, "MOTP": 76.4, "IDF1": 58.1, "HOTA": 48.2},
            seq_stats=[{"seq_name": "MOT17-02-DPM", "n_frames": 600,
                         "hz": 45.2, "n_tracks": 12}],
            out_path=out,
            config_name="config/default.yaml",
        )
        content = out.read_text()
        assert "52.3" in content
        assert "IDF1" in content
        assert "MOTA" in content

    def test_write_results_no_metrics(self, tmp_path):
        """write_results_markdown must work even with no TrackEval metrics."""
        out = tmp_path / "results.md"
        write_results_markdown(
            metrics=None,
            seq_stats=[{"seq_name": "MOT17-02-DPM", "n_frames": 600,
                         "hz": 45.2, "n_tracks": 12}],
            out_path=out,
            config_name="config/default.yaml",
        )
        assert out.exists()
        content = out.read_text()
        assert "MOT17-02-DPM" in content

    def test_write_results_contains_seq_name(self, tmp_path):
        out = tmp_path / "results.md"
        write_results_markdown(
            metrics=None,
            seq_stats=[{"seq_name": "MOT17-04-FRCNN", "n_frames": 1050,
                         "hz": 38.7, "n_tracks": 25}],
            out_path=out,
            config_name="config/default.yaml",
        )
        assert "MOT17-04-FRCNN" in out.read_text()

    def test_write_results_creates_parent_dirs(self, tmp_path):
        out = tmp_path / "nested" / "deep" / "results.md"
        write_results_markdown(
            metrics=None,
            seq_stats=[],
            out_path=out,
            config_name="config/default.yaml",
        )
        assert out.exists()
