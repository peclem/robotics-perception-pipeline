"""
detection_node — wraps perception.detector.YOLOv8Detector.

Subscribes
----------
  /perception/image_raw    sensor_msgs/Image
  /perception/camera_info  sensor_msgs/CameraInfo  (cached; intrinsics
                           change rarely)

Publishes
---------
  /perception/detections   vision_msgs/Detection2DArray

The published Detection2DArray preserves source frame's stamp so
downstream nodes can correlate detections with the originating image.
class_id and score are encoded per Detection2D.results[0]
(ObjectHypothesisWithPose). class_name is stored in id (string field
on the bbox.center.theta? No — we encode it via header.frame_id ?).
Actually we encode the class name in source_img header and the COCO
class index as the hypothesis id; the human label is reconstructed
downstream via a fixed COCO mapping.
"""

from __future__ import annotations

import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import CameraInfo, Image
from vision_msgs.msg import (
    Detection2D, Detection2DArray,
    BoundingBox2D, ObjectHypothesisWithPose, ObjectHypothesis,
)

from perception.camera_interface import CameraFrame, CameraIntrinsics
from perception.config_loader import load_config
from perception.detector import YOLOv8Detector
from robotics_perception_ros2.image_bridge import imgmsg_to_numpy


class DetectionNode(Node):

    def __init__(self) -> None:
        super().__init__("detection_node")

        self.declare_parameter("config_path", "config/default.yaml")
        cfg = load_config(self.get_parameter("config_path").value)
        raw = cfg.as_dict()

        self.get_logger().info(
            f"loading detector {cfg.detector.model} on {cfg.detector.device} ..."
        )
        self._detector = YOLOv8Detector(raw)
        self._detector.warmup()
        self.get_logger().info(
            f"detector ready (warmup latency {self._detector.mean_inference_ms:.1f} ms)"
        )

        # Cached intrinsics. Real CameraInfo overrides at first subscribe.
        self._intrinsics: CameraIntrinsics | None = None
        self._cinfo_warned = False

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.create_subscription(
            CameraInfo, "/perception/camera_info",
            self._on_camera_info, qos,
        )
        self.create_subscription(
            Image, "/perception/image_raw",
            self._on_image, qos,
        )
        self._pub = self.create_publisher(
            Detection2DArray, "/perception/detections", qos,
        )

    def _on_camera_info(self, msg: CameraInfo) -> None:
        self._intrinsics = CameraIntrinsics(
            fx=msg.k[0], fy=msg.k[4], cx=msg.k[2], cy=msg.k[5],
            width=msg.width, height=msg.height,
            dist_coeffs=np.asarray(msg.d, dtype=np.float64)
                if msg.d else np.zeros(5),
        )

    def _on_image(self, msg: Image) -> None:
        if self._intrinsics is None:
            if not self._cinfo_warned:
                self.get_logger().warn(
                    "no /perception/camera_info yet — dropping image"
                )
                self._cinfo_warned = True
            return

        img = imgmsg_to_numpy(msg)
        frame = CameraFrame(
            image=img,
            timestamp=time.monotonic(),
            frame_idx=0,           # downstream doesn't depend on this
            intrinsics=self._intrinsics,
            source_id="ros2",
        )
        detections = self._detector.detect(frame)

        out = Detection2DArray()
        out.header = msg.header   # preserve stamp + frame_id
        for d in detections:
            x1, y1, x2, y2 = d.bbox_xyxy
            bbox = BoundingBox2D()
            bbox.center.position.x = float((x1 + x2) / 2.0)
            bbox.center.position.y = float((y1 + y2) / 2.0)
            bbox.size_x = float(x2 - x1)
            bbox.size_y = float(y2 - y1)

            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis = ObjectHypothesis(
                class_id=str(d.class_id),
                score=float(d.confidence),
            )

            det = Detection2D()
            det.header = msg.header
            det.bbox = bbox
            det.results = [hyp]
            # Store class_name in id (free-form string).
            det.id = d.class_name
            out.detections.append(det)

        self._pub.publish(out)


def main():
    rclpy.init()
    node = DetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
