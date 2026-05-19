"""
Minimal sensor_msgs/Image ↔ numpy ndarray conversion.

Why not cv_bridge
-----------------
cv_bridge ships as a binary wheel against system numpy (1.x on Ubuntu
22.04). The project venv runs numpy 2.x for torch / kornia / pandas
compatibility. Loading cv_bridge from the venv raises a NumPy ABI
warning and may segfault on real conversions. We need a few dozen
lines of straight buffer manipulation, so we write it ourselves and
avoid the ABI mismatch entirely.

Supported encodings: bgr8 / rgb8 (uint8, 3-channel), mono8 (uint8,
1-channel), 32FC1 (float32, 1-channel — metric depth).
Extend the tables below when a new encoding lands.
"""

from __future__ import annotations

import numpy as np
from sensor_msgs.msg import Image


# Channels per pixel and numpy dtype for each supported encoding.
# Step (bytes per row) = width * channels * dtype.itemsize.
_CHANNELS = {
    "bgr8":  3,
    "rgb8":  3,
    "mono8": 1,
    "32FC1": 1,
}

_DTYPE_FOR = {
    "bgr8":  np.uint8,
    "rgb8":  np.uint8,
    "mono8": np.uint8,
    "32FC1": np.float32,
}


def imgmsg_to_numpy(msg: Image) -> np.ndarray:
    """sensor_msgs/Image → numpy ndarray.

    Shape is (H, W, C) for multi-channel encodings, (H, W) for single-
    channel encodings. Dtype matches the encoding: uint8 for *8 codes,
    float32 for 32FC1 (metric depth).
    """
    channels = _CHANNELS.get(msg.encoding)
    dtype    = _DTYPE_FOR.get(msg.encoding)
    if channels is None or dtype is None:
        raise ValueError(
            f"image_bridge: unsupported encoding '{msg.encoding}'. "
            f"Supported: {sorted(_CHANNELS)}"
        )
    arr = np.frombuffer(msg.data, dtype=dtype)
    if channels == 1:
        return arr.reshape(msg.height, msg.width)
    return arr.reshape(msg.height, msg.width, channels)


def numpy_to_imgmsg(
    arr: np.ndarray,
    encoding: str = "bgr8",
    stamp=None,
    frame_id: str = "camera_frame",
) -> Image:
    """numpy ndarray → sensor_msgs/Image. dtype must match encoding."""
    channels = _CHANNELS.get(encoding)
    dtype    = _DTYPE_FOR.get(encoding)
    if channels is None or dtype is None:
        raise ValueError(f"image_bridge: unsupported encoding '{encoding}'")
    if arr.dtype != dtype:
        raise ValueError(
            f"image_bridge: encoding '{encoding}' expects dtype {dtype.__name__}, "
            f"got {arr.dtype}"
        )

    msg = Image()
    if stamp is not None:
        msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.height = arr.shape[0]
    msg.width  = arr.shape[1]
    msg.encoding = encoding
    msg.is_bigendian = 0
    msg.step = arr.shape[1] * channels * np.dtype(dtype).itemsize
    msg.data = arr.tobytes()
    return msg
