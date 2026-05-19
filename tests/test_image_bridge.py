"""
Tests for the ROS2 image_bridge — sensor_msgs/Image ↔ numpy ndarray.

These tests need `sensor_msgs` on PYTHONPATH (so they're skipped when
the host hasn't sourced ROS2). Cover round-trip equality, per-encoding
dtype enforcement, step (bytes-per-row) calculation for both uint8 and
float32 encodings, and rejection of unknown encodings.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("sensor_msgs")

# image_bridge lives under ros2_ws/ — add it to the import path so the
# test runs without needing colcon-installed bindings.
_ROS2_PKG = (
    Path(__file__).resolve().parent.parent
    / "ros2_ws" / "src" / "robotics_perception_ros2"
)
if str(_ROS2_PKG) not in sys.path:
    sys.path.insert(0, str(_ROS2_PKG))

from robotics_perception_ros2.image_bridge import (  # noqa: E402
    imgmsg_to_numpy, numpy_to_imgmsg,
)


class TestRoundTrip:

    def test_bgr8_round_trip_preserves_pixels(self):
        arr = (np.arange(480 * 640 * 3, dtype=np.uint8)
               .reshape(480, 640, 3))
        msg = numpy_to_imgmsg(arr, encoding="bgr8")
        out = imgmsg_to_numpy(msg)
        assert out.dtype == np.uint8
        assert out.shape == arr.shape
        np.testing.assert_array_equal(out, arr)

    def test_mono8_round_trip_preserves_pixels(self):
        arr = (np.arange(480 * 640, dtype=np.uint8).reshape(480, 640))
        msg = numpy_to_imgmsg(arr, encoding="mono8")
        out = imgmsg_to_numpy(msg)
        assert out.dtype == np.uint8
        assert out.shape == arr.shape
        np.testing.assert_array_equal(out, arr)

    def test_32fc1_round_trip_preserves_pixels(self):
        arr = (np.linspace(0.5, 5.0, 480 * 640, dtype=np.float32)
               .reshape(480, 640))
        msg = numpy_to_imgmsg(arr, encoding="32FC1")
        out = imgmsg_to_numpy(msg)
        assert out.dtype == np.float32
        assert out.shape == arr.shape
        # Float round-trip should be exact — we copy bytes, not values.
        np.testing.assert_array_equal(out, arr)


class TestStepCalculation:

    def test_bgr8_step_equals_width_times_three(self):
        arr = np.zeros((10, 20, 3), dtype=np.uint8)
        msg = numpy_to_imgmsg(arr, encoding="bgr8")
        assert msg.step == 20 * 3 * 1   # width * channels * 1-byte dtype

    def test_mono8_step_equals_width(self):
        arr = np.zeros((10, 20), dtype=np.uint8)
        msg = numpy_to_imgmsg(arr, encoding="mono8")
        assert msg.step == 20 * 1 * 1

    def test_32fc1_step_equals_width_times_four(self):
        arr = np.zeros((10, 20), dtype=np.float32)
        msg = numpy_to_imgmsg(arr, encoding="32FC1")
        # Single-channel float32 → 4 bytes per pixel.
        assert msg.step == 20 * 1 * 4


class TestDtypeEnforcement:

    def test_32fc1_rejects_float64_input(self):
        arr = np.zeros((10, 20), dtype=np.float64)
        with pytest.raises(ValueError, match="expects dtype float32"):
            numpy_to_imgmsg(arr, encoding="32FC1")

    def test_bgr8_rejects_float32_input(self):
        arr = np.zeros((10, 20, 3), dtype=np.float32)
        with pytest.raises(ValueError, match="expects dtype uint8"):
            numpy_to_imgmsg(arr, encoding="bgr8")

    def test_unknown_encoding_rejected_on_write(self):
        arr = np.zeros((10, 20), dtype=np.uint8)
        with pytest.raises(ValueError, match="unsupported encoding"):
            numpy_to_imgmsg(arr, encoding="rgba8")

    def test_unknown_encoding_rejected_on_read(self):
        from sensor_msgs.msg import Image
        msg = Image()
        msg.encoding = "rgba8"
        msg.width = 1
        msg.height = 1
        msg.data = b"\x00\x00\x00\x00"
        with pytest.raises(ValueError, match="unsupported encoding"):
            imgmsg_to_numpy(msg)
