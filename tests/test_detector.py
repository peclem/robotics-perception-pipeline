"""
Unit tests for the Detection module.

Test strategy
-------------
- TestDetection: pure dataclass tests — no model, no GPU required.
- TestDetectorInterface: validates the ABC contract using a minimal
  concrete stub — no model, no GPU, runs in CI.
- TestYOLOv8DetectorUnit: patches Ultralytics so no download occurs.
  Tests the parsing logic, class filtering, confidence sorting.
- TestYOLOv8DetectorIntegration: loads the real model on real hardware.
  Marked @pytest.mark.integration — skipped in CI, run locally.
- TestIoU: validates the IoU utility used by the tracker.

Why mock the YOLO model?
------------------------
Unit tests must not download models, require GPU, or have network access.
We mock the model's __call__ to return a controlled fake result object.
This lets us test the parsing logic (the part that can actually be wrong)
without the model being present.
"""

import time
import numpy as np
import pytest
from unittest.mock import MagicMock, patch, PropertyMock

from perception.camera_interface import CameraFrame, CameraIntrinsics, SyntheticCamera
from perception.detector import Detection, Detector, YOLOv8Detector, compute_iou


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_intrinsics():
    return CameraIntrinsics(fx=500.0, fy=500.0, cx=320.0, cy=240.0,
                             width=640, height=480)

def make_frame(frame_idx: int = 0) -> CameraFrame:
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    image[100:200, 100:200] = [0, 200, 100]  # a visible rectangle
    return CameraFrame(
        image=image,
        timestamp=time.monotonic(),
        frame_idx=frame_idx,
        intrinsics=make_intrinsics(),
        source_id="test",
    )

def make_detection(
    x1=100.0, y1=100.0, x2=200.0, y2=200.0,
    conf=0.85, cls_id=0, cls_name="person",
    frame_idx=0,
) -> Detection:
    return Detection(
        bbox_xyxy=np.array([x1, y1, x2, y2], dtype=np.float32),
        confidence=conf,
        class_id=cls_id,
        class_name=cls_name,
        frame_idx=frame_idx,
        timestamp=time.monotonic(),
    )

def make_fake_yolo_results(bboxes, confs, cls_ids):
    """
    Build a fake Ultralytics result object matching the structure
    that YOLOv8Detector._parse_results() expects.
    """
    import torch

    boxes_mock = MagicMock()
    boxes_mock.xyxy = torch.tensor(bboxes, dtype=torch.float32)
    boxes_mock.conf = torch.tensor(confs, dtype=torch.float32)
    boxes_mock.cls  = torch.tensor(cls_ids, dtype=torch.float32)
    boxes_mock.__len__ = lambda self: len(bboxes)

    result_mock = MagicMock()
    result_mock.boxes = boxes_mock

    return [result_mock]

def make_yolo_config(
    conf=0.35, iou=0.45, device="cpu",
    half=False, cls_filter=None, max_dets=100, img_size=640
):
    return {
        "detector": {
            "model": "yolov8n.pt",
            "confidence_threshold": conf,
            "iou_threshold": iou,
            "device": device,
            "half_precision": half,
            "class_filter": cls_filter,
            "max_detections": max_dets,
            "img_size": img_size,
        }
    }


# ---------------------------------------------------------------------------
# TestDetection — pure dataclass, no model
# ---------------------------------------------------------------------------

class TestDetection:

    def test_construction_valid(self):
        det = make_detection()
        assert det.class_name == "person"
        assert det.confidence == pytest.approx(0.85)

    def test_bbox_is_float32_array(self):
        det = make_detection()
        assert det.bbox_xyxy.dtype == np.float32
        assert det.bbox_xyxy.shape == (4,)

    def test_bbox_accepted_as_list(self):
        det = Detection(
            bbox_xyxy=[10.0, 20.0, 110.0, 120.0],
            confidence=0.9, class_id=0, class_name="person",
            frame_idx=0, timestamp=0.0,
        )
        assert isinstance(det.bbox_xyxy, np.ndarray)
        assert det.bbox_xyxy.dtype == np.float32

    def test_center_x(self):
        det = make_detection(x1=100.0, y1=100.0, x2=200.0, y2=200.0)
        assert det.center_x == pytest.approx(150.0)

    def test_center_y(self):
        det = make_detection(x1=100.0, y1=100.0, x2=200.0, y2=300.0)
        assert det.center_y == pytest.approx(200.0)

    def test_width_and_height(self):
        det = make_detection(x1=50.0, y1=80.0, x2=170.0, y2=220.0)
        assert det.width == pytest.approx(120.0)
        assert det.height == pytest.approx(140.0)

    def test_area(self):
        det = make_detection(x1=0.0, y1=0.0, x2=100.0, y2=50.0)
        assert det.area == pytest.approx(5000.0)

    def test_center_array(self):
        det = make_detection(x1=100.0, y1=100.0, x2=200.0, y2=200.0)
        c = det.center
        assert c.shape == (2,)
        assert c[0] == pytest.approx(150.0)
        assert c[1] == pytest.approx(150.0)

    def test_bbox_xywh(self):
        det = make_detection(x1=100.0, y1=100.0, x2=200.0, y2=200.0)
        xywh = det.bbox_xywh
        assert xywh.shape == (4,)
        assert xywh[0] == pytest.approx(150.0)  # cx
        assert xywh[1] == pytest.approx(150.0)  # cy
        assert xywh[2] == pytest.approx(100.0)  # w
        assert xywh[3] == pytest.approx(100.0)  # h

    def test_bbox_tlwh(self):
        det = make_detection(x1=50.0, y1=60.0, x2=150.0, y2=180.0)
        tlwh = det.bbox_tlwh
        assert tlwh[0] == pytest.approx(50.0)   # x1
        assert tlwh[1] == pytest.approx(60.0)   # y1
        assert tlwh[2] == pytest.approx(100.0)  # w
        assert tlwh[3] == pytest.approx(120.0)  # h

    def test_is_valid_normal(self):
        det = make_detection()
        assert det.is_valid()

    def test_is_valid_degenerate_zero_area(self):
        det = make_detection(x1=100.0, y1=100.0, x2=100.0, y2=200.0)
        assert not det.is_valid()

    def test_is_valid_inverted_coords(self):
        det = make_detection(x1=200.0, y1=200.0, x2=100.0, y2=100.0)
        assert not det.is_valid()

    def test_invalid_bbox_shape_raises(self):
        with pytest.raises(ValueError, match="shape"):
            Detection(
                bbox_xyxy=np.array([1.0, 2.0, 3.0], dtype=np.float32),
                confidence=0.9, class_id=0, class_name="person",
                frame_idx=0, timestamp=0.0,
            )

    def test_invalid_confidence_raises(self):
        with pytest.raises(ValueError, match="confidence"):
            make_detection(conf=1.5)

    def test_negative_confidence_raises(self):
        with pytest.raises(ValueError, match="confidence"):
            make_detection(conf=-0.1)

    def test_repr_contains_class_name(self):
        det = make_detection(cls_name="bicycle")
        assert "bicycle" in repr(det)

    def test_repr_contains_confidence(self):
        det = make_detection(conf=0.91)
        assert "0.91" in repr(det)


# ---------------------------------------------------------------------------
# TestDetectorInterface — ABC contract via stub
# ---------------------------------------------------------------------------

class MinimalDetector(Detector):
    """Minimal concrete implementation for testing the ABC."""
    def detect(self, frame: CameraFrame) -> list:
        return [make_detection(frame_idx=frame.frame_idx)]
    def warmup(self) -> None:
        pass
    @property
    def model_name(self) -> str:
        return "minimal"
    @property
    def device(self) -> str:
        return "cpu"


class TestDetectorInterface:

    def test_abc_cannot_be_instantiated(self):
        with pytest.raises(TypeError):
            Detector({})  # type: ignore

    def test_concrete_subclass_works(self):
        det = MinimalDetector({})
        frame = make_frame()
        result = det.detect(frame)
        assert isinstance(result, list)
        assert all(isinstance(d, Detection) for d in result)

    def test_detect_returns_list(self):
        det = MinimalDetector({})
        result = det.detect(make_frame())
        assert isinstance(result, list)

    def test_mean_inference_ms_zero_before_calls(self):
        det = MinimalDetector({})
        assert det.mean_inference_ms == 0.0

    def test_mean_inference_ms_after_recording(self):
        det = MinimalDetector({})
        det._record_inference_time(0.01)   # 10ms
        det._record_inference_time(0.02)   # 20ms
        assert det.mean_inference_ms == pytest.approx(15.0, abs=0.1)

    def test_timing_buffer_caps_at_200(self):
        det = MinimalDetector({})
        for _ in range(250):
            det._record_inference_time(0.005)
        assert len(det._inference_times) <= 200

    def test_repr_contains_model_name(self):
        det = MinimalDetector({})
        assert "minimal" in repr(det)


# ---------------------------------------------------------------------------
# TestYOLOv8DetectorUnit — patches Ultralytics, no download, no GPU
# ---------------------------------------------------------------------------

@pytest.fixture
def patched_yolo_detector():
    """
    YOLOv8Detector with the Ultralytics YOLO class mocked out.
    No model download, no GPU needed. Tests parsing logic only.
    """
    with patch("perception.detector.torch.cuda.is_available", return_value=False), \
         patch("perception.detector.YOLOv8Detector._build_model") as mock_build:

        def fake_build(self):
            self._model = MagicMock()
            self._model.names = {
                0: "person", 1: "bicycle", 2: "car",
                3: "motorcycle", 5: "bus", 7: "truck",
            }
            self._is_ready = True

        mock_build.side_effect = lambda self=None: fake_build(
            patched_yolo_detector._detector
        )

        cfg = make_yolo_config(device="cpu", half=False)
        detector = YOLOv8Detector(cfg)
        patched_yolo_detector._detector = detector
        yield detector


class TestYOLOv8DetectorUnit:

    def _make_detector_with_mock_model(self, class_filter=None, max_dets=100, conf=0.35):
        """Build a YOLOv8Detector with a fully controlled mock model."""
        with patch("perception.detector.YOLOv8Detector._build_model"):
            cfg = make_yolo_config(
                device="cpu", half=False,
                cls_filter=class_filter, max_dets=max_dets, conf=conf
            )
            det = YOLOv8Detector(cfg)

        # Inject mock model manually
        det._model = MagicMock()
        det._class_names = {
            0: "person", 1: "bicycle", 2: "car",
            3: "motorcycle", 5: "bus", 7: "truck",
        }
        det._is_ready = True
        return det

    def _inject_results(self, detector, bboxes, confs, cls_ids):
        """Set model to return specific fake results."""
        fake = make_fake_yolo_results(bboxes, confs, cls_ids)
        detector._model.return_value = fake

    # --- output type contract ---

    def test_returns_list(self):
        det = self._make_detector_with_mock_model()
        self._inject_results(det, [], [], [])
        result = det.detect(make_frame())
        assert isinstance(result, list)

    def test_empty_frame_returns_empty_list(self):
        det = self._make_detector_with_mock_model()
        self._inject_results(det, [], [], [])
        result = det.detect(make_frame())
        assert result == []

    def test_output_elements_are_detection_instances(self):
        det = self._make_detector_with_mock_model()
        self._inject_results(
            det,
            [[10, 20, 110, 120]],
            [0.9],
            [0],
        )
        results = det.detect(make_frame())
        assert len(results) == 1
        assert isinstance(results[0], Detection)

    # --- parsing correctness ---

    def test_bbox_parsed_correctly(self):
        det = self._make_detector_with_mock_model()
        self._inject_results(det, [[50, 60, 150, 200]], [0.8], [0])
        result = det.detect(make_frame())
        assert result[0].x1 == pytest.approx(50.0)
        assert result[0].y1 == pytest.approx(60.0)
        assert result[0].x2 == pytest.approx(150.0)
        assert result[0].y2 == pytest.approx(200.0)

    def test_confidence_parsed_correctly(self):
        det = self._make_detector_with_mock_model()
        self._inject_results(det, [[10, 10, 100, 100]], [0.73], [0])
        result = det.detect(make_frame())
        assert result[0].confidence == pytest.approx(0.73, abs=1e-4)

    def test_class_id_parsed(self):
        det = self._make_detector_with_mock_model()
        self._inject_results(det, [[10, 10, 100, 100]], [0.8], [2])
        result = det.detect(make_frame())
        assert result[0].class_id == 2

    def test_class_name_resolved(self):
        det = self._make_detector_with_mock_model()
        self._inject_results(det, [[10, 10, 100, 100]], [0.8], [2])
        result = det.detect(make_frame())
        assert result[0].class_name == "car"

    def test_frame_idx_propagated(self):
        det = self._make_detector_with_mock_model()
        self._inject_results(det, [[10, 10, 100, 100]], [0.8], [0])
        frame = make_frame(frame_idx=42)
        result = det.detect(frame)
        assert result[0].frame_idx == 42

    # --- sorting ---

    def test_sorted_by_confidence_descending(self):
        det = self._make_detector_with_mock_model()
        self._inject_results(
            det,
            [[10,10,100,100], [200,200,300,300], [400,400,500,500]],
            [0.5, 0.9, 0.7],
            [0, 0, 0],
        )
        results = det.detect(make_frame())
        confs = [d.confidence for d in results]
        assert confs == sorted(confs, reverse=True), (
            "Detections must be sorted confidence-descending for the tracker"
        )

    # --- class filtering ---

    def test_class_filter_keeps_matching(self):
        det = self._make_detector_with_mock_model(class_filter=[0])
        self._inject_results(
            det,
            [[10,10,100,100], [200,200,300,300]],
            [0.8, 0.9],
            [0, 2],  # person, car
        )
        results = det.detect(make_frame())
        assert len(results) == 1
        assert results[0].class_id == 0

    def test_class_filter_none_keeps_all(self):
        det = self._make_detector_with_mock_model(class_filter=None)
        self._inject_results(
            det,
            [[10,10,100,100], [200,200,300,300]],
            [0.8, 0.7],
            [0, 7],
        )
        results = det.detect(make_frame())
        assert len(results) == 2

    def test_class_filter_multiple_classes(self):
        det = self._make_detector_with_mock_model(class_filter=[0, 2, 7])
        self._inject_results(
            det,
            [[10,10,100,100],[200,200,300,300],[400,400,500,500],[600,600,700,700]],
            [0.8, 0.7, 0.6, 0.9],
            [0, 2, 5, 7],  # person car bus truck — filter keeps person car truck
        )
        results = det.detect(make_frame())
        ids = {d.class_id for d in results}
        assert ids == {0, 2, 7}
        assert 5 not in ids  # bus excluded

    # --- max detections cap ---

    def test_max_detections_respected(self):
        det = self._make_detector_with_mock_model(max_dets=2)
        self._inject_results(
            det,
            [[i*10,i*10,i*10+50,i*10+50] for i in range(5)],
            [0.9, 0.8, 0.7, 0.6, 0.5],
            [0] * 5,
        )
        results = det.detect(make_frame())
        assert len(results) <= 2, (
            "max_detections cap must be applied — the tracker has a fixed-size "
            "cost matrix and can silently break with too many detections"
        )

    def test_max_detections_keeps_highest_confidence(self):
        det = self._make_detector_with_mock_model(max_dets=2)
        self._inject_results(
            det,
            [[i*10,i*10,i*10+50,i*10+50] for i in range(5)],
            [0.9, 0.8, 0.7, 0.6, 0.5],
            [0] * 5,
        )
        results = det.detect(make_frame())
        # After sort + cap, must be the two highest-confidence detections
        assert results[0].confidence == pytest.approx(0.9, abs=1e-4)
        assert results[1].confidence == pytest.approx(0.8, abs=1e-4)

    # --- validity filtering ---

    def test_invalid_zero_area_detection_filtered(self):
        det = self._make_detector_with_mock_model()
        self._inject_results(
            det,
            [[100, 100, 100, 200]],  # zero width — invalid
            [0.9],
            [0],
        )
        results = det.detect(make_frame())
        assert len(results) == 0, (
            "Zero-area detections must be filtered before entering the tracker"
        )

    # --- properties ---

    def test_model_name_property(self):
        det = self._make_detector_with_mock_model()
        assert det.model_name == "yolov8n.pt"

    def test_device_property(self):
        det = self._make_detector_with_mock_model()
        assert det.device == "cpu"

    def test_confidence_threshold_property(self):
        det = self._make_detector_with_mock_model(conf=0.42)
        assert det.confidence_threshold == pytest.approx(0.42)

    def test_inference_time_tracked(self):
        det = self._make_detector_with_mock_model()
        self._inject_results(det, [[10,10,100,100]], [0.8], [0])
        det.detect(make_frame())
        assert det.mean_inference_ms >= 0.0

    # --- warmup ---

    def test_warmup_clears_timing_buffer(self):
        det = self._make_detector_with_mock_model()
        det._model.return_value = make_fake_yolo_results([], [], [])
        det._record_inference_time(0.05)
        assert len(det._inference_times) > 0
        det.warmup()
        assert len(det._inference_times) == 0, (
            "Warmup must flush timing buffer so warmup runs don't "
            "inflate the mean_inference_ms baseline"
        )

    def test_warmup_calls_model_three_times(self):
        det = self._make_detector_with_mock_model()
        det._model.return_value = make_fake_yolo_results([], [], [])
        det.warmup()
        assert det._model.call_count == 3

    def test_warmup_on_unready_detector_does_not_raise(self):
        with patch("perception.detector.YOLOv8Detector._build_model"):
            cfg = make_yolo_config(device="cpu")
            det = YOLOv8Detector(cfg)
        det._is_ready = False
        det.warmup()  # must not raise


# ---------------------------------------------------------------------------
# TestIoU — utility used by tracker
# ---------------------------------------------------------------------------

class TestIoU:

    def test_identical_boxes_iou_is_one(self):
        a = make_detection(x1=10, y1=10, x2=110, y2=110)
        b = make_detection(x1=10, y1=10, x2=110, y2=110)
        assert compute_iou(a, b) == pytest.approx(1.0)

    def test_non_overlapping_boxes_iou_is_zero(self):
        a = make_detection(x1=0, y1=0, x2=50, y2=50)
        b = make_detection(x1=100, y1=100, x2=200, y2=200)
        assert compute_iou(a, b) == pytest.approx(0.0)

    def test_partial_overlap(self):
        a = make_detection(x1=0, y1=0, x2=100, y2=100)
        b = make_detection(x1=50, y1=50, x2=150, y2=150)
        iou = compute_iou(a, b)
        # Intersection: 50x50=2500, Union: 10000+10000-2500=17500
        assert iou == pytest.approx(2500.0 / 17500.0, abs=1e-5)

    def test_contained_box(self):
        outer = make_detection(x1=0, y1=0, x2=100, y2=100)
        inner = make_detection(x1=25, y1=25, x2=75, y2=75)
        iou = compute_iou(outer, inner)
        # Intersection = inner area = 2500, Union = 10000
        assert iou == pytest.approx(2500.0 / 10000.0, abs=1e-5)

    def test_iou_is_symmetric(self):
        a = make_detection(x1=0, y1=0, x2=80, y2=80)
        b = make_detection(x1=40, y1=40, x2=120, y2=120)
        assert compute_iou(a, b) == pytest.approx(compute_iou(b, a))

    def test_iou_in_range_zero_to_one(self):
        a = make_detection(x1=10, y1=10, x2=90, y2=90)
        b = make_detection(x1=50, y1=50, x2=130, y2=130)
        iou = compute_iou(a, b)
        assert 0.0 <= iou <= 1.0


# ---------------------------------------------------------------------------
# Integration tests — real model, real GPU (run locally, not in CI)
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestYOLOv8DetectorIntegration:
    """
    These tests load the actual YOLOv8n model and run inference.
    Skip in CI — run manually to validate GPU pipeline end-to-end.

    Run with:
        pytest tests/test_detector.py -m integration -v
    """

    @pytest.fixture(scope="class")
    def detector(self):
        import torch
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available — skipping integration tests")

        cfg = {
            "detector": {
                "model": "yolov8n.pt",
                "confidence_threshold": 0.25,
                "iou_threshold": 0.45,
                "device": "cuda:0",
                "half_precision": True,
                "class_filter": None,
                "max_detections": 100,
                "img_size": 640,
            }
        }
        det = YOLOv8Detector(cfg)
        det.warmup()
        return det

    def test_returns_list_on_blank_frame(self, detector):
        frame = make_frame()
        result = detector.detect(frame)
        assert isinstance(result, list)

    def test_all_outputs_are_valid_detections(self, detector):
        frame = make_frame()
        result = detector.detect(frame)
        for d in result:
            assert d.is_valid(), f"Invalid detection returned: {d}"

    def test_inference_time_under_20ms(self, detector):
        frame = make_frame()
        for _ in range(10):
            detector.detect(frame)
        assert detector.mean_inference_ms < 20.0, (
            f"Mean inference {detector.mean_inference_ms:.1f}ms > 20ms on 4070Ti. "
            "Check CUDA is being used and half_precision=true."
        )

    def test_synthetic_camera_integration(self, detector):
        cam = SyntheticCamera({}, num_frames=5, num_objects=2)
        with cam:
            for _ in range(5):
                frame = cam.get_frame()
                if frame is None:
                    break
                dets = detector.detect(frame)
                assert isinstance(dets, list)
