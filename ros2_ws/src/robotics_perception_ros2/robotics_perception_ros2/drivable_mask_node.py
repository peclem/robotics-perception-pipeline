"""
drivable_mask_node — semantic-segmenter-driven drivable-area publisher.

Subscribes
----------
  /perception/image_raw    sensor_msgs/Image       (BGR8)
  /perception/camera_info  sensor_msgs/CameraInfo  (cached intrinsics)

Publishes
---------
  /perception/drivable_mask  sensor_msgs/Image (mono8)
                             255 where the surface is drivable
                             (Cityscapes road / sidewalk / terrain, or
                              ADE20K floor / road / earth), 0 elsewhere.

Why an Image and not nav_msgs/OccupancyGrid
-------------------------------------------
The drivable_mask helper is per-pixel in IMAGE space — it tells you
"this pixel is drivable" but not "this 5 cm cell at world (x, y) is
drivable". Producing a top-down costmap from this requires depth +
camera pose + ground-plane projection, which is its own piece of work
(slated for a separate node when a planner needs it). For now we ship
the image-space mask: directly consumable for visual debugging,
image-based hazard masks (drone visual-servoing), or as the input to
a downstream projector.

Gating
------
When `semantic.enabled=false` (or `semantic.type='null'`), the node
starts but stays idle — no model is loaded, no messages are
published. Same pattern as the rest of the adapter nodes: ship the
node unconditionally so the launch file doesn't need to know which
features are on; let config gate the behaviour.
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
from perception.semantic_segmenter import (
    NullSemanticSegmenter, SemanticSegmenter, drivable_mask_mono8,
)
from robotics_perception_ros2.image_bridge import imgmsg_to_numpy, numpy_to_imgmsg


class DrivableMaskNode(Node):

    def __init__(self, *, enable_intra_process: bool = False) -> None:
        # enable_intra_process is a no-op at this layer in Python rclpy
        # (Humble) — kept on the signature for composite-launcher parity.
        super().__init__("drivable_mask_node")

        self.declare_parameter("config_path", "config/default.yaml")
        cfg = load_config(self.get_parameter("config_path").value)

        self._segmenter: SemanticSegmenter = self._build_segmenter(cfg)
        self._idle = isinstance(self._segmenter, NullSemanticSegmenter)
        if self._idle:
            self.get_logger().info(
                "semantic disabled — drivable_mask_node idle "
                "(no model loaded, no /perception/drivable_mask "
                "messages will be published)"
            )

        # Cached intrinsics. CameraFrame requires them; the segmenter
        # itself doesn't depend on the camera matrix, so a fallback is
        # fine until /perception/camera_info arrives.
        self._intrinsics: CameraIntrinsics | None = None

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
            Image, "/perception/drivable_mask", qos,
        )

    @staticmethod
    def _build_segmenter(cfg) -> SemanticSegmenter:
        """Factory mirroring launch.Pipeline._build_semantic_segmenter."""
        sc = cfg.semantic
        if not sc.enabled or sc.type == "null":
            return NullSemanticSegmenter()
        if sc.type == "mask2former":
            from perception.semantic_segmenter import (
                Mask2FormerSemanticSegmenter,
            )
            seg = Mask2FormerSemanticSegmenter(
                device=sc.device, model_name=sc.model, dataset=sc.dataset,
            )
            seg.warmup()
            return seg
        raise ValueError(f"Unknown semantic.type={sc.type!r}")

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
        img = imgmsg_to_numpy(msg)
        intr = self._intrinsics or CameraIntrinsics(
            # Reasonable fallback if /perception/camera_info hasn't
            # arrived yet — the segmenter doesn't consume intrinsics,
            # so any internally-consistent values work.
            fx=500.0, fy=500.0,
            cx=img.shape[1] / 2.0, cy=img.shape[0] / 2.0,
            width=img.shape[1], height=img.shape[0],
        )
        frame = CameraFrame(
            image=img, timestamp=time.monotonic(),
            frame_idx=0, intrinsics=intr, source_id="ros2",
        )
        sm = self._segmenter.segment(frame)
        if sm is None:
            return
        mono = drivable_mask_mono8(sm)
        out = numpy_to_imgmsg(mono, encoding="mono8")
        out.header = msg.header
        self._pub.publish(out)


def main():
    rclpy.init()
    node = DrivableMaskNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
