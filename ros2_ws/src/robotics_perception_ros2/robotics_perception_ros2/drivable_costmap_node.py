"""
drivable_costmap_node — IPM-projected drivable freespace as nav_msgs/OccupancyGrid.

Subscribes
----------
  /perception/drivable_mask    sensor_msgs/Image (mono8 — drivable_mask_node output)
  /perception/depth            sensor_msgs/Image (32FC1 — depth_node output, optional)
  /perception/camera_info      sensor_msgs/CameraInfo (cached intrinsics)
  /tf, /tf_static              tf2 — camera pose in the world frame

Publishes
---------
  /perception/drivable_costmap  nav_msgs/OccupancyGrid
                                -1 = unknown, 0 = drivable / free. 100
                                (occupied) is reserved for a future
                                non-drivable-surface projection backend.

Backends
--------
- Depth-aware (preferred): when /perception/depth is being published and
  the depth-image header.stamp matches the mask's header.stamp (within a
  small cache window), each drivable pixel is back-projected through its
  metric depth — handles slopes, stairs, uneven terrain.
- Flat-ground (fallback): when no matching depth is available, falls
  back to inverse perspective mapping against z = z_ground. Works for
  ground robots on roughly flat surfaces.

Pipeline position
-----------------
This node consumes the output of drivable_mask_node (which produces
image-space mask via Mask2Former) and emits a Nav2-ready costmap.
Splitting the two responsibilities (image segmentation vs. geometric
projection) matches the rest of the adapter graph and lets the
projector be unit-tested without a GPU model.
"""

from __future__ import annotations

import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.duration import Duration
from sensor_msgs.msg import CameraInfo, Image
from nav_msgs.msg import OccupancyGrid
from tf2_ros import Buffer, TransformListener, TransformException
from scipy.spatial.transform import Rotation

from perception.camera_interface import CameraIntrinsics
from perception.config_loader import load_config
from perception.pose_estimator import CameraPose
from world_model.drivable_projector import (
    DrivableProjectorParams, project_drivable_to_grid,
)
from world_model.occupancy_grid import OccupancyGridParams
from robotics_perception_ros2.image_bridge import imgmsg_to_numpy


def _tf_to_camera_pose(tf_msg, timestamp: float) -> CameraPose:
    """
    geometry_msgs/TransformStamped → CameraPose with world←camera
    convention. The TF stores translation = child origin in parent
    frame, rotation = parent←child as a quaternion (Hamilton, xyzw
    in ROS), so the transform IS what we want directly.
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


class DrivableCostmapNode(Node):

    def __init__(self, *, enable_intra_process: bool = False) -> None:
        super().__init__("drivable_costmap_node")

        self.declare_parameter("config_path", "config/default.yaml")
        cfg = load_config(self.get_parameter("config_path").value)

        # Reuse the OccupancyGridConfig for grid spec. A separate config
        # block would be appropriate if we wanted the drivable costmap
        # to live in a different frame / resolution than the dynamic
        # obstacle layer, but co-locating them simplifies Nav2 fusion.
        og = cfg.occupancy_grid
        self._grid_params = OccupancyGridParams(
            resolution_m=og.resolution_m,
            size_x_m=og.size_x_m, size_y_m=og.size_y_m,
            origin_x_m=og.origin_x_m, origin_y_m=og.origin_y_m,
        )
        self._world_frame = cfg.coordinate_frames.root_frame
        # Flat-ground assumption — z=0 in the world frame by default.
        self._projector_params = DrivableProjectorParams(
            grid_params=self._grid_params,
            z_ground_m=0.0,
        )

        self._intrinsics: CameraIntrinsics | None = None
        self._tf_buffer  = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        # Cache of recent depth maps keyed by (sec, nanosec) of the
        # source image. Lookup is O(1) when the mask + depth share a
        # stamp; we keep a tiny ring buffer so an out-of-order frame
        # still finds its mate. Bounded to avoid leaking memory if a
        # publisher dies mid-stream.
        self._depth_cache: dict[tuple[int, int], np.ndarray] = {}
        self._depth_cache_order: list[tuple[int, int]] = []
        self._depth_cache_max = 8

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
            Image, "/perception/drivable_mask",
            self._on_mask, qos,
        )
        self.create_subscription(
            Image, "/perception/depth",
            self._on_depth, qos,
        )
        self._pub = self.create_publisher(
            OccupancyGrid, "/perception/drivable_costmap", qos,
        )

        self.get_logger().info(
            f"drivable_costmap_node ready "
            f"(grid {og.size_x_m}×{og.size_y_m} m at {og.resolution_m} m/cell, "
            f"world frame '{self._world_frame}')"
        )

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

    def _on_mask(self, msg: Image) -> None:
        if self._intrinsics is None:
            return
        # tf2 lookup at the mask's stamp; brief timeout to ride out
        # in-flight TF buffering. If unavailable, skip this frame.
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

        mask = imgmsg_to_numpy(msg)
        camera_pose = _tf_to_camera_pose(
            tf_msg, time.monotonic(),
        )
        # Look up the matching depth map by stamp. None → projector
        # falls back to the flat-ground backend.
        key = (msg.header.stamp.sec, msg.header.stamp.nanosec)
        depth_map = self._depth_cache.get(key)
        if depth_map is not None and depth_map.shape != mask.shape:
            # Mask + depth resolutions disagree — surfaces a config bug
            # (different /perception/image_raw scalings upstream). Skip
            # the depth-aware backend rather than throw mid-publish.
            self.get_logger().warn(
                f"depth shape {depth_map.shape} != mask shape {mask.shape} — "
                "falling back to flat-ground projection",
                throttle_duration_sec=5.0,
            )
            depth_map = None
        grid = project_drivable_to_grid(
            mask, self._intrinsics, camera_pose, self._projector_params,
            depth_map=depth_map,
        )

        out = OccupancyGrid()
        out.header.stamp = msg.header.stamp
        out.header.frame_id = self._world_frame
        out.info.resolution = float(self._grid_params.resolution_m)
        out.info.width  = int(grid.shape[1])
        out.info.height = int(grid.shape[0])
        out.info.origin.position.x = float(self._grid_params.origin_x_m)
        out.info.origin.position.y = float(self._grid_params.origin_y_m)
        out.info.origin.position.z = 0.0
        out.info.origin.orientation.w = 1.0
        out.data = grid.flatten().tolist()
        self._pub.publish(out)


def main():
    rclpy.init()
    node = DrivableCostmapNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
