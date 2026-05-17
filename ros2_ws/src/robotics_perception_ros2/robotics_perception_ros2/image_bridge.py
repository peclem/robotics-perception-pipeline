"""
Minimal sensor_msgs/Image ↔ numpy ndarray conversion.

Why not cv_bridge
-----------------
cv_bridge ships as a binary wheel against system numpy (1.x on Ubuntu
22.04). The project venv runs numpy 2.x for torch / kornia / pandas
compatibility. Loading cv_bridge from the venv raises a NumPy ABI
warning and may segfault on real conversions. We need ~10 lines of
straight buffer manipulation, so we write it ourselves and avoid the
ABI mismatch entirely.

Supports BGR8 (OpenCV native) and MONO8. Extend as needed.
"""

from __future__ import annotations

import numpy as np
from sensor_msgs.msg import Image


_CHANNELS = {
    "bgr8":  3,
    "rgb8":  3,
    "mono8": 1,
}


def imgmsg_to_numpy(msg: Image) -> np.ndarray:
    """sensor_msgs/Image → (H, W, C) uint8 ndarray (or (H, W) for mono8)."""
    channels = _CHANNELS.get(msg.encoding)
    if channels is None:
        raise ValueError(
            f"image_bridge: unsupported encoding '{msg.encoding}'. "
            f"Supported: {sorted(_CHANNELS)}"
        )
    arr = np.frombuffer(msg.data, dtype=np.uint8)
    if channels == 1:
        return arr.reshape(msg.height, msg.width)
    return arr.reshape(msg.height, msg.width, channels)


def numpy_to_imgmsg(
    arr: np.ndarray,
    encoding: str = "bgr8",
    stamp=None,
    frame_id: str = "camera_frame",
) -> Image:
    """(H, W, C) uint8 ndarray → sensor_msgs/Image."""
    if arr.dtype != np.uint8:
        raise ValueError(f"image_bridge: expected uint8, got {arr.dtype}")
    channels = _CHANNELS.get(encoding)
    if channels is None:
        raise ValueError(f"image_bridge: unsupported encoding '{encoding}'")

    msg = Image()
    if stamp is not None:
        msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.height = arr.shape[0]
    msg.width = arr.shape[1]
    msg.encoding = encoding
    msg.is_bigendian = 0
    msg.step = arr.shape[1] * channels
    msg.data = arr.tobytes()
    return msg
