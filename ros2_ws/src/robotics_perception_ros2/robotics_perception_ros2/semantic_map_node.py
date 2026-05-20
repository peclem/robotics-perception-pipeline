"""
semantic_map_node — persistent metric-semantic voxel map on the ROS2 graph.

This is the ROS2 face of the semantic SLAM mapping layer
(`world_model/semantic_map.py`). It integrates per-frame depth + semantic
labels + camera pose into a persistent `SemanticMap` and republishes the
map as a class-coloured point cloud.

Subscribes
----------
  /perception/image_raw    sensor_msgs/Image       (BGR8)
  /perception/camera_info  sensor_msgs/CameraInfo  (cached intrinsics)
  /perception/depth        sensor_msgs/Image       (32FC1 metres — depth_node)
  /tf, /tf_static          tf2 — camera pose in the world frame

Publishes
---------
  /perception/semantic_map  sensor_msgs/PointCloud2 (XYZRGB)
                            One point per occupied voxel centre; the RGB
                            field encodes the voxel's fused (MAP) class
                            via `world_model.semantic_map.class_colour`.
                            rviz / Foxglove render it directly.

Design — segmenter in-node
--------------------------
The map needs a *full* per-pixel label image. drivable_mask_node only
publishes the binary drivable mask, so there is no full-label topic to
consume; this node runs the `SemanticSegmenter` itself, exactly as
drivable_mask_node does. A future split (a `semantic_node` publishing the
label image + a thinner map node consuming it) would mirror the
drivable_mask / drivable_costmap division — deferred until a second
consumer of full labels exists.

Gating
------
When `semantic.enabled=false` (or `semantic.type='null'`) the node starts
but stays idle — no model, no publishing — same as drivable_mask_node.
SemanticMap geometry/integration knobs come from the `semantic_map`
config block.

Loop-closure caveat
-------------------
Like the standalone SemanticMap, voxels are integrated at the pose
available per frame; a later loop closure that shifts the trajectory does
not re-deform already-integrated voxels. See semantic_map.py.
"""

from __future__ import annotations

import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.duration import Duration
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField
from std_msgs.msg import Header
from tf2_ros import Buffer, TransformListener, TransformException
from scipy.spatial.transform import Rotation

from perception.camera_interface import CameraFrame, CameraIntrinsics
from perception.config_loader import load_config
from perception.pose_estimator import CameraPose
from perception.semantic_segmenter import (
    NullSemanticSegmenter, SemanticSegmenter,
)
from world_model.semantic_map import (
    SemanticMap, SemanticMapParams, class_colour,
)
from robotics_perception_ros2.image_bridge import imgmsg_to_numpy


def _tf_to_camera_pose(tf_msg, timestamp: float) -> CameraPose:
    """
    geometry_msgs/TransformStamped → CameraPose (world ← camera).
    The TF stores translation = child origin in the parent frame and
    rotation = parent ← child, which is exactly the CameraPose
    convention. Mirrors drivable_costmap_node's helper.
    """
    t = tf_msg.transform.translation
    q = tf_msg.transform.rotation
    R = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
    return CameraPose(
        R=R,
        t=np.array([t.x, t.y, t.z], dtype=np.float64),
        timestamp=timestamp,
        frame_idx=0,
        source="tf",
    )


def _semantic_map_to_pointcloud2(
    semantic_map: SemanticMap, header: Header,
) -> PointCloud2:
    """
    Pack a SemanticMap's occupied voxels into an XYZRGB PointCloud2.

    Layout: 16-byte points — x, y, z as FLOAT32, then a packed RGB
    FLOAT32 (the classic PCL/rviz `rgb` convention: a uint32
    0x00RRGGBB reinterpreted as float32).
    """
    centres = semantic_map.voxel_centres_world().astype(np.float32)
    labels = semantic_map.voxel_labels()
    n = int(centres.shape[0])

    buf = np.zeros(n, dtype=[
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("rgb", "<f4"),
    ])
    if n:
        buf["x"] = centres[:, 0]
        buf["y"] = centres[:, 1]
        buf["z"] = centres[:, 2]
        # One class_colour lookup per distinct class, not per voxel.
        lut = {int(c): class_colour(int(c)) for c in np.unique(labels)}
        rgb_u32 = np.zeros(n, dtype=np.uint32)
        for i, c in enumerate(labels):
            r, g, b = lut[int(c)]
            rgb_u32[i] = (r << 16) | (g << 8) | b
        buf["rgb"] = rgb_u32.view(np.float32)

    msg = PointCloud2()
    msg.header = header
    msg.height = 1
    msg.width = n
    msg.fields = [
        PointField(name="x",   offset=0,  datatype=PointField.FLOAT32, count=1),
        PointField(name="y",   offset=4,  datatype=PointField.FLOAT32, count=1),
        PointField(name="z",   offset=8,  datatype=PointField.FLOAT32, count=1),
        PointField(name="rgb", offset=12, datatype=PointField.FLOAT32, count=1),
    ]
    msg.is_bigendian = False
    msg.point_step = 16
    msg.row_step = msg.point_step * n
    msg.is_dense = True
    msg.data = buf.tobytes()
    return msg


class SemanticMapNode(Node):

    def __init__(self, *, enable_intra_process: bool = False) -> None:
        # enable_intra_process kept for composite-launcher parity; rclpy
        # Python (Humble) doesn't expose intra-process comms.
        super().__init__("semantic_map_node")

        self.declare_parameter("config_path", "config/default.yaml")
        cfg = load_config(self.get_parameter("config_path").value)

        self._segmenter: SemanticSegmenter = self._build_segmenter(cfg)
        self._idle = isinstance(self._segmenter, NullSemanticSegmenter)
        if self._idle:
            self.get_logger().info(
                "semantic disabled — semantic_map_node idle "
                "(no model loaded, no /perception/semantic_map published)"
            )

        sm = cfg.semantic_map
        self._semantic_map = SemanticMap(SemanticMapParams(
            voxel_size_m=sm.voxel_size_m,
            min_range_m=sm.min_range_m,
            max_range_m=sm.max_range_m,
            pixel_stride=sm.pixel_stride,
            occupancy_hit_logodds=sm.occupancy_hit_logodds,
            occupancy_clamp=sm.occupancy_clamp,
            observation_weight=sm.observation_weight,
            max_voxels=sm.max_voxels,
        ))
        self._prune_age_s = sm.prune_age_s
        self._world_frame = cfg.coordinate_frames.root_frame

        self._intrinsics: CameraIntrinsics | None = None
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        # Depth maps cached by source-image stamp so an image and its
        # depth still pair up if they arrive out of order. Bounded ring
        # buffer — same approach as drivable_costmap_node.
        self._depth_cache: dict[tuple[int, int], np.ndarray] = {}
        self._depth_cache_order: list[tuple[int, int]] = []
        self._depth_cache_max = 8

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.create_subscription(
            CameraInfo, "/perception/camera_info", self._on_camera_info, qos,
        )
        self.create_subscription(
            Image, "/perception/depth", self._on_depth, qos,
        )
        self.create_subscription(
            Image, "/perception/image_raw", self._on_image, qos,
        )
        self._pub = self.create_publisher(
            PointCloud2, "/perception/semantic_map", qos,
        )

        if not self._idle:
            self.get_logger().info(
                f"semantic_map_node ready "
                f"(voxel {sm.voxel_size_m} m, range "
                f"{sm.min_range_m}-{sm.max_range_m} m, "
                f"world frame '{self._world_frame}')"
            )

    @staticmethod
    def _build_segmenter(cfg) -> SemanticSegmenter:
        """Factory mirroring drivable_mask_node / launch.Pipeline."""
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

    def _on_depth(self, msg: Image) -> None:
        key = (msg.header.stamp.sec, msg.header.stamp.nanosec)
        self._depth_cache[key] = imgmsg_to_numpy(msg)
        self._depth_cache_order.append(key)
        if len(self._depth_cache_order) > self._depth_cache_max:
            old = self._depth_cache_order.pop(0)
            self._depth_cache.pop(old, None)

    def _on_image(self, msg: Image) -> None:
        if self._idle or self._intrinsics is None:
            return

        # Depth is mandatory for integration — the map is metric. Skip
        # the frame (don't fall back) when no matching depth is cached.
        key = (msg.header.stamp.sec, msg.header.stamp.nanosec)
        depth_map = self._depth_cache.get(key)
        if depth_map is None:
            self.get_logger().warn(
                "no /perception/depth matching this image stamp — "
                "skipping frame (is depth_node running?)",
                throttle_duration_sec=5.0,
            )
            return

        # Camera world-pose via tf2, at the image stamp.
        try:
            tf_msg = self._tf_buffer.lookup_transform(
                self._world_frame, msg.header.frame_id,
                msg.header.stamp, timeout=Duration(seconds=0.05),
            )
        except TransformException as e:
            self.get_logger().warn(
                f"tf lookup {self._world_frame} ← {msg.header.frame_id} "
                f"failed: {e}", throttle_duration_sec=5.0,
            )
            return

        img = imgmsg_to_numpy(msg)
        frame = CameraFrame(
            image=img, timestamp=time.monotonic(),
            frame_idx=0, intrinsics=self._intrinsics, source_id="ros2",
        )
        semantic_mask = self._segmenter.segment(frame)
        if semantic_mask is None:
            return

        camera_pose = _tf_to_camera_pose(tf_msg, time.monotonic())
        # integrate() defensively skips a depth/mask shape mismatch.
        self._semantic_map.integrate(
            depth_map, semantic_mask, camera_pose, self._intrinsics,
        )
        if self._prune_age_s > 0.0:
            self._semantic_map.prune(time.monotonic(), self._prune_age_s)

        header = Header()
        header.stamp = msg.header.stamp
        header.frame_id = self._world_frame
        self._pub.publish(
            _semantic_map_to_pointcloud2(self._semantic_map, header),
        )


def main():
    rclpy.init()
    node = SemanticMapNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
