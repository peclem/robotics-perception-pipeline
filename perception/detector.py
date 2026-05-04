"""
Detection module for the robotics perception pipeline.

Design contract
---------------
All detector backends expose:
    detect(frame: CameraFrame) -> List[Detection]

Detection is the typed DTO passed to the tracker. It carries:
    - bbox_xyxy: pixel-space bounding box [x1, y1, x2, y2] as float32 array
    - confidence: model confidence in [0, 1]
    - class_id / class_name: semantic label
    - timestamp + frame_idx: traceability back to the source frame

Robotics rationale
------------------
Confidence is used downstream as an observation quality signal.
In the ByteTrack integration (Step 5), detections are split into
high-confidence (used for primary assignment) and low-confidence
(used for occlusion recovery). In the KF update (Step 4), confidence
scales the measurement noise matrix R — a high-confidence detection
produces a tighter update.

Upgrade path
------------
Replace YOLOv8Detector with TensorRTDetector (same interface) for
deployment on Jetson Orin or any NVIDIA embedded platform.
"""

from __future__ import annotations

import time
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from perception.camera_interface import CameraFrame


# ---------------------------------------------------------------------------
# Detection dataclass — the core DTO
# ---------------------------------------------------------------------------

@dataclass
class Detection:
    """
    A single object detection output.

    All spatial quantities are in pixel coordinates of the source image.
    The bbox_xyxy format [x1, y1, x2, y2] is the standard used by:
      - IoU computation (tracking association)
      - Kalman filter observation vector (state_estimation/)
      - Rerun.io bounding box logger (visualization/)

    Attributes
    ----------
    bbox_xyxy : float32 array [x1, y1, x2, y2]
    confidence : float in [0.0, 1.0] — detector's certainty
    class_id   : integer COCO class index
    class_name : human-readable label string
    frame_idx  : index of the source frame (for traceability)
    timestamp  : monotonic timestamp from the source CameraFrame
    """
    bbox_xyxy: np.ndarray        # shape (4,), dtype float32
    confidence: float
    class_id: int
    class_name: str
    frame_idx: int
    timestamp: float

    def __post_init__(self):
        self.bbox_xyxy = np.asarray(self.bbox_xyxy, dtype=np.float32)
        if self.bbox_xyxy.shape != (4,):
            raise ValueError(
                f"bbox_xyxy must have shape (4,), got {self.bbox_xyxy.shape}"
            )
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(
                f"confidence must be in [0, 1], got {self.confidence:.4f}"
            )

    # ------------------------------------------------------------------
    # Derived spatial properties — computed on access, never stored
    # ------------------------------------------------------------------

    @property
    def x1(self) -> float: return float(self.bbox_xyxy[0])
    @property
    def y1(self) -> float: return float(self.bbox_xyxy[1])
    @property
    def x2(self) -> float: return float(self.bbox_xyxy[2])
    @property
    def y2(self) -> float: return float(self.bbox_xyxy[3])

    @property
    def center_x(self) -> float:
        return float((self.bbox_xyxy[0] + self.bbox_xyxy[2]) / 2.0)

    @property
    def center_y(self) -> float:
        return float((self.bbox_xyxy[1] + self.bbox_xyxy[3]) / 2.0)

    @property
    def width(self) -> float:
        return float(self.bbox_xyxy[2] - self.bbox_xyxy[0])

    @property
    def height(self) -> float:
        return float(self.bbox_xyxy[3] - self.bbox_xyxy[1])

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> np.ndarray:
        """Center as [cx, cy] float32 array."""
        return np.array([self.center_x, self.center_y], dtype=np.float32)

    @property
    def bbox_xywh(self) -> np.ndarray:
        """Bounding box as [cx, cy, w, h] — Kalman filter observation format."""
        return np.array(
            [self.center_x, self.center_y, self.width, self.height],
            dtype=np.float32,
        )

    @property
    def bbox_tlwh(self) -> np.ndarray:
        """Bounding box as [x1, y1, w, h] — top-left + size format."""
        return np.array(
            [self.x1, self.y1, self.width, self.height],
            dtype=np.float32,
        )

    def is_valid(self) -> bool:
        """
        Sanity check: box has positive area and coordinates are ordered.
        Invalid detections should be filtered before entering the tracker.
        """
        return (
            self.x2 > self.x1
            and self.y2 > self.y1
            and self.area > 0.0
            and self.confidence > 0.0
        )

    def __repr__(self) -> str:
        return (
            f"Detection({self.class_name!r} conf={self.confidence:.2f} "
            f"[{self.x1:.0f},{self.y1:.0f},{self.x2:.0f},{self.y2:.0f}] "
            f"frame={self.frame_idx})"
        )


# ---------------------------------------------------------------------------
# Detector abstract base class
# ---------------------------------------------------------------------------

class Detector(ABC):
    """
    Abstract base for all detection backends.

    The tracker and all downstream modules depend only on this interface.
    Concrete backends (YOLOv8, TensorRT, ONNX Runtime) are interchangeable.

    Contract
    --------
    - detect() always returns a List[Detection], never raises on empty frames
    - All detections pass Detection.is_valid() before being returned
    - Detections are sorted by confidence descending (highest first)
    """

    def __init__(self, config: dict):
        self._config = config
        self._det_config = config.get("detector", {})
        self._inference_times: List[float] = []

    @abstractmethod
    def detect(self, frame: CameraFrame) -> List[Detection]:
        """
        Run inference on a single frame.

        Parameters
        ----------
        frame : CameraFrame from any camera backend

        Returns
        -------
        List[Detection] sorted by confidence descending.
        Empty list if no objects detected — never None.
        """
        ...

    @abstractmethod
    def warmup(self) -> None:
        """
        Run one or more dummy inferences to initialise the GPU pipeline.

        Call once after construction, before the main loop.
        On a 4070Ti + YOLOv8n, the first real inference takes ~200ms
        (CUDA graph capture + kernel compilation). After warmup, it drops
        to ~5ms. Missing warmup causes a latency spike on frame 1.
        """
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Human-readable model identifier."""
        ...

    @property
    @abstractmethod
    def device(self) -> str:
        """Device string: 'cuda:0', 'cpu', etc."""
        ...

    # ------------------------------------------------------------------
    # Shared utilities
    # ------------------------------------------------------------------

    @property
    def mean_inference_ms(self) -> float:
        """Running mean of the last 100 inference times in milliseconds."""
        if not self._inference_times:
            return 0.0
        window = self._inference_times[-100:]
        return float(np.mean(window)) * 1000.0

    def _record_inference_time(self, elapsed_s: float) -> None:
        self._inference_times.append(elapsed_s)
        if len(self._inference_times) > 200:
            self._inference_times = self._inference_times[-100:]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model_name!r}, device={self.device!r})"


# ---------------------------------------------------------------------------
# Utility: IoU between two detections
# ---------------------------------------------------------------------------

def compute_iou(det_a: Detection, det_b: Detection) -> float:
    """
    Intersection-over-Union between two Detection bboxes.

    Used in the tracker's association step (Step 5). Defined here so
    it operates on the Detection type rather than raw arrays.
    """
    x1 = max(det_a.x1, det_b.x1)
    y1 = max(det_a.y1, det_b.y1)
    x2 = min(det_a.x2, det_b.x2)
    y2 = min(det_a.y2, det_b.y2)

    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter == 0.0:
        return 0.0

    union = det_a.area + det_b.area - inter
    return inter / union if union > 0.0 else 0.0


# ---------------------------------------------------------------------------
# YOLOv8 detector — concrete backend
# ---------------------------------------------------------------------------

class YOLOv8Detector(Detector):
    """
    YOLOv8 detector wrapping Ultralytics.

    Config keys (all under 'detector:' in default.yaml)
    ---------------------------------------------------
    model               : str   — model name or path.
                                  'yolov8n.pt' downloads ~6MB on first run.
                                  Use an absolute path for a local checkpoint.
    confidence_threshold: float — minimum confidence to return a detection.
                                  0.35 is a good starting point; tune per scene.
    iou_threshold       : float — NMS IoU threshold. 0.45 is Ultralytics default.
    device              : str   — 'cuda:0' for GPU, 'cpu' for CPU fallback.
    half_precision      : bool  — fp16 inference. True on 4070Ti gives ~2x speedup.
                                  Ignored on CPU.
    class_filter        : list  — COCO class IDs to keep, e.g. [0] for persons.
                                  null / empty list = all classes.
    max_detections      : int   — cap on returned detections per frame.
    img_size            : int   — inference resolution (square). 640 = YOLO default.

    4070Ti performance targets
    --------------------------
    YOLOv8n fp16 @ 640x640: ~4–6 ms/frame  (~200 FPS ceiling)
    YOLOv8s fp16 @ 640x640: ~8–10 ms/frame (~120 FPS ceiling)
    YOLOv8m fp16 @ 640x640: ~14–18 ms/frame (~65 FPS ceiling)
    Start with 'n', upgrade when you need accuracy.
    """

    def __init__(self, config: dict):
        super().__init__(config)

        cfg = self._det_config
        self._model_name: str = cfg.get("model", "yolov8n.pt")
        self._conf_thresh: float = float(cfg.get("confidence_threshold", 0.35))
        self._iou_thresh: float = float(cfg.get("iou_threshold", 0.45))
        self._device_str: str = cfg.get("device", "cuda:0")
        self._half: bool = bool(cfg.get("half_precision", True))
        self._img_size: int = int(cfg.get("img_size", 640))
        self._max_dets: int = int(cfg.get("max_detections", 100))

        raw_filter = cfg.get("class_filter", None)
        self._class_filter: Optional[set] = (
            set(int(c) for c in raw_filter)
            if raw_filter else None
        )

        self._model = None  # lazy: loaded in _build_model()
        self._class_names: Dict[int, str] = {}
        self._is_ready: bool = False

        self._build_model()

    def _build_model(self) -> None:
        """Load the YOLO model onto the target device.

        Do NOT manually call .half() here. Ultralytics' AutoBackend fuses
        Conv+BatchNorm lazily on the first forward pass. If you cast to FP16
        before that fusion, Conv weights become FP16 while BN weights are still
        FP32 — fuse_conv_and_bn then raises a dtype mismatch.

        FP16 is handled by passing half=self._half to every self._model() call.
        AutoBackend converts after fusion, in the correct order.
        """
        from ultralytics import YOLO

        if "cuda" in self._device_str and not torch.cuda.is_available():
            warnings.warn(
                f"CUDA requested ({self._device_str}) but not available. "
                "Falling back to CPU. Inference will be slow.",
                stacklevel=3,
            )
            self._device_str = "cpu"
            self._half = False

        if self._half and self._device_str == "cpu":
            self._half = False

        self._model = YOLO(self._model_name)
        self._model.to(self._device_str)
        # ← removed: self._model.model.half()

        self._class_names = self._model.names
        self._is_ready = True


    def detect(self, frame: CameraFrame) -> List[Detection]:
        """
        Run YOLOv8 inference on a single CameraFrame.

        The image is passed directly — no copy, no resize (YOLO handles
        letterboxing internally). Confidence and NMS thresholds are applied
        inside YOLO. We then apply class filtering and the max_detections cap.
        """
        if not self._is_ready or self._model is None:
            warnings.warn("YOLOv8Detector: model not ready.", stacklevel=2)
            return []

        t0 = time.monotonic()

        results = self._model(
            frame.image,
            conf=self._conf_thresh,
            iou=self._iou_thresh,
            imgsz=self._img_size,
            half=self._half,
            verbose=False,       # suppress Ultralytics console spam
            device=self._device_str,
        )

        elapsed = time.monotonic() - t0
        self._record_inference_time(elapsed)

        return self._parse_results(results, frame)

    def _parse_results(self, results, frame: CameraFrame) -> List[Detection]:
        """
        Convert Ultralytics result objects to Detection dataclasses.

        Ultralytics result format:
            results[0].boxes.xyxy  — tensor (N, 4)
            results[0].boxes.conf  — tensor (N,)
            results[0].boxes.cls   — tensor (N,)
        All tensors are on whatever device the model is on.
        We move them to CPU and convert to numpy here.
        """
        if not results or results[0].boxes is None:
            return []

        boxes = results[0].boxes

        if len(boxes) == 0:
            return []

        # Move to CPU + numpy in one call to minimise device transfers
        bboxes = boxes.xyxy.cpu().numpy().astype(np.float32)   # (N, 4)
        confs  = boxes.conf.cpu().numpy().astype(np.float32)   # (N,)
        clsids = boxes.cls.cpu().numpy().astype(np.int32)      # (N,)

        detections: List[Detection] = []

        for bbox, conf, cls_id in zip(bboxes, confs, clsids):
            # Apply class filter
            if self._class_filter is not None and int(cls_id) not in self._class_filter:
                continue

            cls_name = self._class_names.get(int(cls_id), f"class_{cls_id}")

            det = Detection(
                bbox_xyxy=bbox,
                confidence=float(conf),
                class_id=int(cls_id),
                class_name=cls_name,
                frame_idx=frame.frame_idx,
                timestamp=frame.timestamp,
            )

            if det.is_valid():
                detections.append(det)

        # Sort by confidence descending — tracker processes best detections first
        detections.sort(key=lambda d: d.confidence, reverse=True)

        # Apply max_detections cap
        return detections[: self._max_dets]

    def warmup(self) -> None:
        """
        Run 3 dummy inferences to prime the CUDA pipeline.

        Uses uint8 dummy frames — same dtype as real camera output.
        FP16 conversion happens inside Ultralytics on each call, after
        Conv+BN fusion completes on the first pass.
        """
        if not self._is_ready:
            return

        dummy = np.zeros((self._img_size, self._img_size, 3), dtype=np.uint8)

        for _ in range(3):
            self._model(
                dummy,
                conf=self._conf_thresh,
                iou=self._iou_thresh,
                imgsz=self._img_size,
                half=self._half,      # Ultralytics handles FP16 here, post-fusion
                verbose=False,
                device=self._device_str,
            )

        self._inference_times.clear()

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def device(self) -> str:
        return self._device_str

    @property
    def class_names(self) -> Dict[int, str]:
        return dict(self._class_names)

    @property
    def confidence_threshold(self) -> float:
        return self._conf_thresh
