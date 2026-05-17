"""
tracking_node — wraps tracking.tracker.ByteTracker.

Subscribes
----------
  /perception/image_raw    sensor_msgs/Image
  /perception/camera_info  sensor_msgs/CameraInfo  (cached)
  /perception/detections   vision_msgs/Detection2DArray

Synchronisation strategy: cache the most recent image keyed by stamp,
then on each Detection2DArray arrival look up the matching image.
Both topics come from the same source and are stamped identically by
upstream nodes, so an exact-stamp lookup is sufficient — we don't
need an approximate-time synchroniser yet.

Publishes
---------
  /perception/tracks       vision_msgs/Detection2DArray
                           (Detection2D.id field encodes track ID)
"""

from __future__ import annotations

import time
from collections import OrderedDict

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
from perception.detector import Detection
from tracking.tracker import ByteTracker
from robotics_perception_ros2.image_bridge import imgmsg_to_numpy


class TrackingNode(Node):

    def __init__(self) -> None:
        super().__init__("tracking_node")

        self.declare_parameter("config_path", "config/default.yaml")
        cfg = load_config(self.get_parameter("config_path").value)
        raw = cfg.as_dict()

        self._tracker = ByteTracker(raw)
        self._intrinsics: CameraIntrinsics | None = None

        # Small ring of recent images keyed by (sec, nsec) so we can
        # pair each incoming detection array with its source frame.
        self._image_cache: "OrderedDict[tuple, np.ndarray]" = OrderedDict()
        self._cache_max = 8

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.create_subscription(
            CameraInfo, "/perception/camera_info", self._on_info, qos,
        )
        self.create_subscription(
            Image, "/perception/image_raw", self._on_image, qos,
        )
        self.create_subscription(
            Detection2DArray, "/perception/detections", self._on_detections, qos,
        )
        self._pub = self.create_publisher(
            Detection2DArray, "/perception/tracks", qos,
        )
        self.get_logger().info("tracking_node ready")

    def _on_info(self, msg: CameraInfo) -> None:
        self._intrinsics = CameraIntrinsics(
            fx=msg.k[0], fy=msg.k[4], cx=msg.k[2], cy=msg.k[5],
            width=msg.width, height=msg.height,
            dist_coeffs=np.asarray(msg.d, dtype=np.float64)
                if msg.d else np.zeros(5),
        )

    def _on_image(self, msg: Image) -> None:
        key = (msg.header.stamp.sec, msg.header.stamp.nanosec)
        self._image_cache[key] = imgmsg_to_numpy(msg)
        while len(self._image_cache) > self._cache_max:
            self._image_cache.popitem(last=False)

    def _on_detections(self, msg: Detection2DArray) -> None:
        if self._intrinsics is None:
            return
        key = (msg.header.stamp.sec, msg.header.stamp.nanosec)
        img = self._image_cache.get(key)
        if img is None:
            self.get_logger().debug(
                f"no cached image for stamp {key} — dropping detections"
            )
            return

        # Detection2DArray → List[Detection] for ByteTracker.
        dets: list[Detection] = []
        for d in msg.detections:
            hyp = d.results[0] if d.results else None
            if hyp is None:
                continue
            cx = d.bbox.center.position.x
            cy = d.bbox.center.position.y
            w  = d.bbox.size_x
            h  = d.bbox.size_y
            dets.append(Detection(
                bbox_xyxy=np.array(
                    [cx - w/2, cy - h/2, cx + w/2, cy + h/2],
                    dtype=np.float32,
                ),
                confidence=float(hyp.hypothesis.score),
                class_id=int(hyp.hypothesis.class_id)
                    if hyp.hypothesis.class_id.isdigit() else 0,
                class_name=d.id or "unknown",
                frame_idx=0,
                timestamp=time.monotonic(),
            ))
        # ByteTracker contract: detections sorted confidence-descending.
        dets.sort(key=lambda d: -d.confidence)

        frame = CameraFrame(
            image=img,
            timestamp=time.monotonic(),
            frame_idx=0,
            intrinsics=self._intrinsics,
            source_id="ros2",
        )
        confirmed = self._tracker.update(dets, frame)

        out = Detection2DArray()
        out.header = msg.header
        for tr in confirmed:
            cx, cy = tr.position
            w  = tr.size[0] if hasattr(tr, "size") else 0.0
            h  = tr.size[1] if hasattr(tr, "size") else 0.0
            # Track stores [cx, cy, w, h] inside its KF state; use bbox xyxy.
            x1, y1, x2, y2 = tr.bbox_xyxy
            bbox = BoundingBox2D()
            bbox.center.position.x = float((x1 + x2) / 2.0)
            bbox.center.position.y = float((y1 + y2) / 2.0)
            bbox.size_x = float(x2 - x1)
            bbox.size_y = float(y2 - y1)

            det = Detection2D()
            det.header = msg.header
            det.bbox = bbox
            det.id = f"{tr.track_id}:{tr.class_name}"  # track_id colon class
            det.results = [ObjectHypothesisWithPose(
                hypothesis=ObjectHypothesis(
                    class_id=str(tr.class_id), score=float(tr.score),
                )
            )]
            out.detections.append(det)
        self._pub.publish(out)


def main():
    rclpy.init()
    node = TrackingNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
