"""
depth_node — wraps a DepthEstimator and publishes the dense map.

Subscribes
----------
  /perception/image_raw    sensor_msgs/Image  (BGR8)
  /perception/camera_info  sensor_msgs/CameraInfo  (cached intrinsics)

Publishes
---------
  /perception/depth        sensor_msgs/Image  (32FC1, metres)
                           Dense per-pixel metric depth.

Backend selection is delegated to
`perception.depth_estimator_factory.build_depth_estimator`, the same
factory the standalone `launch.py` uses. `depth.enabled=false` (or
`depth.type='null'`) makes this node idle — no model loaded, no
messages published. Mirrors the gating pattern of drivable_mask_node.

Pipeline position
-----------------
This is the dense-depth publisher that drivable_costmap_node subscribes
to for its depth-aware projection. Without /perception/depth in the
graph, drivable_costmap_node falls back to flat-ground IPM.
"""

from __future__ import annotations

import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import CameraInfo, Image

from perception.camera_interface import CameraFrame, CameraIntrinsics
from perception.config_loader import load_config
from perception.depth_estimator import NullDepthEstimator
from perception.depth_estimator_factory import build_depth_estimator
from robotics_perception_ros2.image_bridge import imgmsg_to_numpy, numpy_to_imgmsg


class DepthNode(Node):

    def __init__(self, *, enable_intra_process: bool = False) -> None:
        super().__init__("depth_node")

        self.declare_parameter("config_path", "config/default.yaml")
        cfg = load_config(self.get_parameter("config_path").value)

        self._estimator = build_depth_estimator(cfg)
        self._idle = isinstance(self._estimator, NullDepthEstimator)
        if self._idle:
            self.get_logger().info(
                "depth disabled — depth_node idle (no model loaded, "
                "no /perception/depth messages will be published)"
            )
        else:
            self.get_logger().info(
                f"DepthEstimator: {type(self._estimator).__name__}"
            )

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
            Image, "/perception/depth", qos,
        )

    def _on_camera_info(self, msg: CameraInfo) -> None:
        self._intrinsics = CameraIntrinsics(
            fx=msg.k[0], fy=msg.k[4], cx=msg.k[2], cy=msg.k[5],
            width=msg.width, height=msg.height,
            dist_coeffs=np.asarray(msg.d, dtype=np.float64)
                if msg.d else np.zeros(5),
        )

    def _on_image(self, msg: Image) -> None:
        if self._idle:
            return
        if self._intrinsics is None:
            if not self._cinfo_warned:
                self.get_logger().warn(
                    "no /perception/camera_info yet — dropping image"
                )
                self._cinfo_warned = True
            return

        img = imgmsg_to_numpy(msg)
        frame = CameraFrame(
            image=img, timestamp=time.monotonic(),
            frame_idx=0, intrinsics=self._intrinsics, source_id="ros2",
        )
        dense = self._estimator.dense_depth_map(frame)
        if dense is None:
            return

        out = numpy_to_imgmsg(
            dense.astype(np.float32, copy=False),
            encoding="32FC1",
        )
        out.header = msg.header
        self._pub.publish(out)


def main():
    rclpy.init()
    node = DepthNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
