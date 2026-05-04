"""
Camera abstraction layer for the robotics perception pipeline.

Design contract
---------------
All camera backends expose the same interface:
    get_frame() -> Optional[CameraFrame]

CameraFrame is the typed data transfer object passed to every downstream
module (detector, tracker, state estimator). It carries:
    - the raw image array
    - a monotonic timestamp (not wall-clock — critical for sensor fusion)
    - the camera intrinsics needed for metric-space projection
    - the frame index for sequence tracking

Robotics upgrade path
---------------------
In a ROS2 deployment, WebcamCamera is replaced by a ROS2ImageSubscriber that
converts sensor_msgs/Image + sensor_msgs/CameraInfo into CameraFrame objects.
The rest of the pipeline is unchanged — the interface contract is identical.
"""

from __future__ import annotations

import time
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import yaml


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CameraIntrinsics:
    """
    Pinhole camera intrinsics (OpenCV convention).

    Parameters
    ----------
    fx, fy : focal lengths in pixels
    cx, cy : principal point in pixels
    width, height : sensor resolution
    dist_coeffs : distortion coefficients [k1, k2, p1, p2, k3]
                  Use np.zeros(5) for an undistorted or pre-rectified stream.

    Robotics note
    -------------
    These parameters are what converts a 2D pixel coordinate (u, v) into a
    3D bearing vector. Combined with a depth estimate (Phase 2+), they give
    you metric-space object positions — the input to the state estimator.
    """
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    dist_coeffs: np.ndarray = field(
        default_factory=lambda: np.zeros(5, dtype=np.float64)
    )

    def camera_matrix(self) -> np.ndarray:
        """Return the 3x3 intrinsic matrix K."""
        return np.array(
            [[self.fx, 0.0, self.cx],
             [0.0, self.fy, self.cy],
             [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    def pixel_to_bearing(self, u: float, v: float) -> np.ndarray:
        """
        Convert a pixel coordinate to a unit bearing vector.

        Returns a 3D unit vector [x, y, z] in the camera frame.
        This is the geometric primitive underlying monocular depth estimation
        and visual-inertial odometry.
        """
        x = (u - self.cx) / self.fx
        y = (v - self.cy) / self.fy
        bearing = np.array([x, y, 1.0], dtype=np.float64)
        return bearing / np.linalg.norm(bearing)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CameraIntrinsics):
            return NotImplemented
        return (
            self.fx == other.fx and self.fy == other.fy
            and self.cx == other.cx and self.cy == other.cy
            and self.width == other.width and self.height == other.height
            and np.allclose(self.dist_coeffs, other.dist_coeffs)
        )

    def __hash__(self) -> int:
        return hash((self.fx, self.fy, self.cx, self.cy, self.width, self.height))


@dataclass
class CameraFrame:
    """
    Typed output of any CameraInterface backend.

    Attributes
    ----------
    image : (H, W, 3) uint8 BGR array — OpenCV convention
    timestamp : monotonic seconds since an arbitrary epoch.
                DO NOT use for wall-clock time. Use for dt computation only.
    frame_idx : zero-based frame counter within the current session
    intrinsics : camera model — always present, never None
    source_id : human-readable label ('webcam:0', 'video:clip.mp4', 'synthetic')

    Why monotonic?
    --------------
    time.monotonic() is guaranteed never to decrease even across NTP
    corrections. On a real robot, mixing wall-clock and monotonic timestamps
    corrupts your Kalman filter's dt — this is a real and common bug.
    """
    image: np.ndarray
    timestamp: float
    frame_idx: int
    intrinsics: CameraIntrinsics
    source_id: str

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.image.shape  # (H, W, C)

    @property
    def height(self) -> int:
        return self.image.shape[0]

    @property
    def width(self) -> int:
        return self.image.shape[1]


# ---------------------------------------------------------------------------
# Intrinsics I/O
# ---------------------------------------------------------------------------

def load_intrinsics(path: str | Path) -> CameraIntrinsics:
    """
    Load camera intrinsics from a YAML file.

    Expected schema (OpenCV calibration output format):
        camera_matrix:
          fx: 800.0
          fy: 800.0
          cx: 320.0
          cy: 240.0
        image_size:
          width: 640
          height: 480
        dist_coeffs: [0.0, 0.0, 0.0, 0.0, 0.0]

    Raises
    ------
    FileNotFoundError if the file does not exist.
    KeyError if required fields are missing.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Intrinsics file not found: {path}")

    with open(path) as f:
        data = yaml.safe_load(f)

    cm = data["camera_matrix"]
    sz = data["image_size"]
    dist = np.array(data.get("dist_coeffs", [0.0, 0.0, 0.0, 0.0, 0.0]),
                    dtype=np.float64)

    return CameraIntrinsics(
        fx=float(cm["fx"]),
        fy=float(cm["fy"]),
        cx=float(cm["cx"]),
        cy=float(cm["cy"]),
        width=int(sz["width"]),
        height=int(sz["height"]),
        dist_coeffs=dist,
    )


def _default_intrinsics(width: int, height: int) -> CameraIntrinsics:
    """
    Synthetic intrinsics assuming a ~60 degree horizontal FoV.
    Use only when no calibration file is available. Always calibrate for
    metric-space use (Phase 2+).
    """
    fx = fy = width / (2.0 * np.tan(np.radians(30.0)))
    return CameraIntrinsics(
        fx=fx, fy=fy,
        cx=width / 2.0, cy=height / 2.0,
        width=width, height=height,
    )


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------

class CameraInterface(ABC):
    """
    Abstract base for all camera backends.

    Subclasses implement get_frame() and release().
    All other pipeline code depends only on this interface.

    Usage (context manager — preferred):
        with WebcamCamera(config) as cam:
            while True:
                frame = cam.get_frame()
                if frame is None:
                    break
                process(frame)

    Usage (manual):
        cam = VideoFileCamera(config, path='clip.mp4')
        cam.open()
        frame = cam.get_frame()
        cam.release()
    """

    def __init__(self, config: dict):
        self._config = config
        self._frame_idx: int = 0
        self._is_open: bool = False

    @abstractmethod
    def open(self) -> None:
        """Initialise the underlying capture source."""
        ...

    @abstractmethod
    def get_frame(self) -> Optional[CameraFrame]:
        """
        Retrieve the next frame.

        Returns None when the source is exhausted (EOF for video files)
        or unavailable (hardware disconnect). Callers must handle None.
        Never raises — failures are expressed as None returns + warnings.
        """
        ...

    @abstractmethod
    def release(self) -> None:
        """Release the underlying capture resource."""
        ...

    @property
    def is_open(self) -> bool:
        return self._is_open

    def __enter__(self) -> "CameraInterface":
        self.open()
        return self

    def __exit__(self, *_) -> None:
        self.release()


# ---------------------------------------------------------------------------
# Backend 1: Live webcam
# ---------------------------------------------------------------------------

class WebcamCamera(CameraInterface):
    """
    Live camera via cv2.VideoCapture.

    In a ROS2 deployment this class is replaced by a ROS2ImageSubscriber.
    The rest of the pipeline never changes.
    """

    def __init__(
        self,
        config: dict,
        device_index: int = 0,
        intrinsics_path: Optional[str] = None,
    ):
        super().__init__(config)
        self._device_index = device_index
        self._intrinsics_path = intrinsics_path
        self._cap: Optional[cv2.VideoCapture] = None
        self._intrinsics: Optional[CameraIntrinsics] = None

    def open(self) -> None:
        cam_cfg = self._config.get("camera", {})
        self._cap = cv2.VideoCapture(self._device_index)

        if not self._cap.isOpened():
            raise RuntimeError(
                f"Could not open camera device {self._device_index}. "
                "Check that the camera is connected and not in use."
            )

        # Apply config settings
        width = cam_cfg.get("width", 1280)
        height = cam_cfg.get("height", 720)
        fps = cam_cfg.get("fps", 30)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._cap.set(cv2.CAP_PROP_FPS, fps)

        # Read back actual values (hardware may round)
        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if self._intrinsics_path:
            self._intrinsics = load_intrinsics(self._intrinsics_path)
        else:
            warnings.warn(
                "No intrinsics file provided — using synthetic defaults. "
                "Metric-space estimates will be inaccurate. "
                "Run scripts/calibrate_camera.py to generate intrinsics.",
                stacklevel=2,
            )
            self._intrinsics = _default_intrinsics(actual_w, actual_h)

        self._is_open = True

    def get_frame(self) -> Optional[CameraFrame]:
        if not self._is_open or self._cap is None:
            return None

        ret, image = self._cap.read()
        if not ret:
            warnings.warn("WebcamCamera: failed to read frame.", stacklevel=2)
            return None

        frame = CameraFrame(
            image=image,
            timestamp=time.monotonic(),
            frame_idx=self._frame_idx,
            intrinsics=self._intrinsics,
            source_id=f"webcam:{self._device_index}",
        )
        self._frame_idx += 1
        return frame

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._is_open = False


# ---------------------------------------------------------------------------
# Backend 2: Video file replay (deterministic mock)
# ---------------------------------------------------------------------------

class VideoFileCamera(CameraInterface):
    """
    Replay a video file as a camera source.

    Key property: deterministic — the same file always produces the same
    sequence of frames. This makes unit tests and benchmarks reproducible.

    Parameters
    ----------
    loop : if True, restart from the beginning on EOF.
           Set loop=False in tests to get a definite end signal.
    playback_fps : if set, enforce inter-frame timing to simulate
                   real-time capture. If None, returns frames as fast
                   as the consumer can read them.
    """

    def __init__(
        self,
        config: dict,
        video_path: str | Path,
        intrinsics_path: Optional[str] = None,
        loop: bool = False,
        playback_fps: Optional[float] = None,
    ):
        super().__init__(config)
        self._video_path = Path(video_path)
        self._intrinsics_path = intrinsics_path
        self._loop = loop
        self._playback_fps = playback_fps
        self._cap: Optional[cv2.VideoCapture] = None
        self._intrinsics: Optional[CameraIntrinsics] = None
        self._last_frame_time: Optional[float] = None

    def open(self) -> None:
        if not self._video_path.exists():
            raise FileNotFoundError(f"Video file not found: {self._video_path}")

        self._cap = cv2.VideoCapture(str(self._video_path))
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open video file: {self._video_path}")

        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if self._intrinsics_path:
            self._intrinsics = load_intrinsics(self._intrinsics_path)
        else:
            self._intrinsics = _default_intrinsics(actual_w, actual_h)

        self._is_open = True

    def get_frame(self) -> Optional[CameraFrame]:
        if not self._is_open or self._cap is None:
            return None

        # Enforce playback rate if requested
        if self._playback_fps is not None and self._last_frame_time is not None:
            target_dt = 1.0 / self._playback_fps
            elapsed = time.monotonic() - self._last_frame_time
            if elapsed < target_dt:
                time.sleep(target_dt - elapsed)

        ret, image = self._cap.read()

        if not ret:
            if self._loop:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, image = self._cap.read()
                if not ret:
                    return None  # Empty file
            else:
                return None  # EOF — caller should stop the loop

        ts = time.monotonic()
        self._last_frame_time = ts

        frame = CameraFrame(
            image=image,
            timestamp=ts,
            frame_idx=self._frame_idx,
            intrinsics=self._intrinsics,
            source_id=f"video:{self._video_path.name}",
        )
        self._frame_idx += 1
        return frame

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._is_open = False

    @property
    def total_frames(self) -> int:
        """Total frame count in the video file."""
        if self._cap is None:
            return 0
        return int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

    @property
    def native_fps(self) -> float:
        """FPS as encoded in the video file."""
        if self._cap is None:
            return 0.0
        return self._cap.get(cv2.CAP_PROP_FPS)


# ---------------------------------------------------------------------------
# Backend 3: Synthetic camera (pure programmatic — no files needed)
# ---------------------------------------------------------------------------

class SyntheticCamera(CameraInterface):
    """
    Generates frames programmatically for unit testing and CI.

    Each frame contains a moving rectangle whose position follows a
    known linear trajectory. Tests can use this ground truth to
    validate tracker accuracy and filter convergence without any
    real video data.

    This is the correct robotics engineering practice: test your
    algorithms against known ground truth before touching real data.

    Parameters
    ----------
    width, height : frame dimensions
    num_frames : how many frames to produce before returning None
    fps : simulated frame rate (used for timestamp spacing)
    num_objects : number of independently moving rectangles to generate
    """

    def __init__(
        self,
        config: dict,
        width: int = 640,
        height: int = 480,
        num_frames: int = 300,
        fps: float = 30.0,
        num_objects: int = 2,
        seed: int = 42,
    ):
        super().__init__(config)
        self._width = width
        self._height = height
        self._num_frames = num_frames
        self._fps = fps
        self._num_objects = num_objects
        self._rng = np.random.default_rng(seed)
        self._intrinsics: Optional[CameraIntrinsics] = None
        self._t0: float = 0.0

        # Each object: [x, y, w, h, vx, vy] — constant velocity
        self._objects: list[np.ndarray] = []

    def open(self) -> None:
        self._intrinsics = _default_intrinsics(self._width, self._height)
        self._t0 = time.monotonic()

        # Initialise objects with random positions and velocities
        box_w, box_h = 60, 45
        for _ in range(self._num_objects):
            x = float(self._rng.integers(box_w, self._width - box_w * 2))
            y = float(self._rng.integers(box_h, self._height - box_h * 2))
            vx = float(self._rng.uniform(1.5, 4.0) * self._rng.choice([-1, 1]))
            vy = float(self._rng.uniform(1.0, 3.0) * self._rng.choice([-1, 1]))
            self._objects.append(np.array([x, y, float(box_w), float(box_h), vx, vy]))

        self._is_open = True

    def get_frame(self) -> Optional[CameraFrame]:
        if not self._is_open:
            return None
        if self._frame_idx >= self._num_frames:
            return None  # Sequence exhausted

        image = np.zeros((self._height, self._width, 3), dtype=np.uint8)
        dt = 1.0 / self._fps

        for obj in self._objects:
            x, y, w, h, vx, vy = obj

            # Bounce off walls
            if x < 0 or x + w > self._width:
                obj[4] *= -1
            if y < 0 or y + h > self._height:
                obj[5] *= -1

            # Advance position
            obj[0] += obj[4] * dt * 30  # scale to pixel-space
            obj[1] += obj[5] * dt * 30
            obj[0] = float(np.clip(obj[0], 0, self._width - w))
            obj[1] = float(np.clip(obj[1], 0, self._height - h))

            x1, y1 = int(obj[0]), int(obj[1])
            x2, y2 = int(obj[0] + w), int(obj[1] + h)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 220, 120), -1)

        # Add mild Gaussian noise (simulates sensor noise)
        noise = self._rng.integers(0, 18, image.shape, dtype=np.uint8)
        image = np.clip(image.astype(np.int16) + noise - 9, 0, 255).astype(np.uint8)

        ts = self._t0 + self._frame_idx * dt

        frame = CameraFrame(
            image=image,
            timestamp=ts,
            frame_idx=self._frame_idx,
            intrinsics=self._intrinsics,
            source_id="synthetic",
        )
        self._frame_idx += 1
        return frame

    def get_ground_truth_bboxes(self) -> list[tuple[int, int, int, int]]:
        """
        Return current ground-truth bounding boxes as (x1, y1, x2, y2).
        Use in tests to compute detection error and validate tracker accuracy.
        """
        boxes = []
        for obj in self._objects:
            x1, y1 = int(obj[0]), int(obj[1])
            x2, y2 = int(obj[0] + obj[2]), int(obj[1] + obj[3])
            boxes.append((x1, y1, x2, y2))
        return boxes

    def release(self) -> None:
        self._is_open = False
