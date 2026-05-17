"""
occupancy_grid_node — dynamic obstacle grid published as nav_msgs/OccupancyGrid.

Subscribes
----------
  /perception/scene        vision_msgs/Detection3DArray
                           (from scene_graph_node — world-frame Detection3Ds)

Publishes
---------
  /perception/costmap      nav_msgs/OccupancyGrid
                           Inflated dynamic obstacle layer ready to feed
                           into a Nav2 costmap or any planner expecting
                           the standard message.

Architecture note
-----------------
This node bridges the standalone OccupancyGridBuilder (pure Python,
no ROS) to the ROS2 graph. We reconstruct minimal ObjectState
equivalents from Detection3DArray rather than running a second
SceneGraph here — the source-of-truth scene state lives upstream in
scene_graph_node. This mirrors the lossiness caveat already documented
in scene_graph_node: full KF covariance doesn't round-trip through
vision_msgs, so the published grid uses the reduced uncertainty
information that does survive (2×2 position covariance via
pose.covariance).
"""

from __future__ import annotations

import time
from typing import List

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import Pose
from nav_msgs.msg import OccupancyGrid
from vision_msgs.msg import Detection3DArray

from perception.config_loader import load_config
from world_model.object_state import ObjectState
from world_model.occupancy_grid import (
    OccupancyGridBuilder, OccupancyGridParams,
)


def _detection3d_to_object_state(d) -> ObjectState | None:
    """
    Reconstruct enough of an ObjectState for OccupancyGridBuilder to
    consume the Detection3D produced by scene_graph_node.

    Returns None when the Detection3D lacks a usable position.
    """
    tid_str, _, cname = (d.id or "0:unknown").partition(":")
    try:
        tid = int(tid_str)
    except ValueError:
        return None

    cls_id = 0
    score = 0.0
    pose_cov = np.eye(6) * 1.0
    if d.results:
        hyp = d.results[0]
        if hyp.hypothesis.class_id.isdigit():
            cls_id = int(hyp.hypothesis.class_id)
        score = float(hyp.hypothesis.score)
        if len(hyp.pose.covariance) == 36:
            pose_cov = np.asarray(hyp.pose.covariance,
                                  dtype=np.float64).reshape(6, 6)

    # 8×8 KF-shaped covariance with the 2×2 position block lifted from
    # Detection3D's pose covariance (the only uncertainty info that
    # survived the ROS round-trip).
    cov_8 = np.eye(8) * 10.0
    cov_8[:2, :2] = pose_cov[:2, :2]

    px = float(d.bbox.center.position.x)
    py = float(d.bbox.center.position.y)
    pz = float(d.bbox.center.position.z)

    return ObjectState(
        track_id=tid,
        class_id=cls_id,
        class_name=cname or "unknown",
        position=np.array([0.0, 0.0]),  # not used by the grid
        covariance=cov_8,
        velocity=np.zeros(4),
        score=score,
        last_seen=time.monotonic(),
        n_updates=1,
        position_world=np.array([px, py, pz]),
        max_history=1,
    )


class OccupancyGridNode(Node):

    def __init__(self, *, enable_intra_process: bool = False) -> None:
        # Note: Python rclpy in Humble doesn't expose use_intra_process_comms
        # as a Node kwarg (C++ ComposableNodeContainer only). enable_intra_process
        # is kept on the signature so the composite launcher can pass it for
        # future-proofing; it's currently a no-op at this layer. Real savings
        # from the composite come from a single CUDA context + reduced
        # process overhead.
        super().__init__("occupancy_grid_node")

        self.declare_parameter("config_path", "config/default.yaml")
        self.declare_parameter("frame_id", "map")

        cfg = load_config(self.get_parameter("config_path").value)
        self._frame_id = self.get_parameter("frame_id").value

        og = cfg.occupancy_grid
        params = OccupancyGridParams(
            resolution_m=og.resolution_m,
            size_x_m=og.size_x_m, size_y_m=og.size_y_m,
            origin_x_m=og.origin_x_m, origin_y_m=og.origin_y_m,
            default_inflation_m=og.default_inflation_m,
            min_inflation_m=og.min_inflation_m,
            per_class_inflation_m=dict(og.per_class_inflation_m),
        )
        # Intrinsics-aware inflation would need /camera_info plumbing
        # here too; keep this node intrinsics-free and rely on the
        # per-class fallback. The standalone builder still supports
        # depth-projected inflation when used inside launch.py.
        self._builder = OccupancyGridBuilder(params, intrinsics=None)

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.create_subscription(
            Detection3DArray, "/perception/scene",
            self._on_scene, qos,
        )
        self._pub = self.create_publisher(
            OccupancyGrid, "/perception/costmap", qos,
        )
        self.get_logger().info(
            f"occupancy_grid_node ready ("
            f"{self._builder.width_cells}×{self._builder.height_cells} cells, "
            f"{params.resolution_m} m/cell)"
        )

    def _on_scene(self, msg: Detection3DArray) -> None:
        objs: List[ObjectState] = []
        for d in msg.detections:
            obj = _detection3d_to_object_state(d)
            if obj is not None:
                objs.append(obj)

        grid_arr = self._builder.build(objs)

        out = OccupancyGrid()
        out.header.stamp = msg.header.stamp
        out.header.frame_id = self._frame_id
        out.info.resolution = self._builder.params.resolution_m
        out.info.width  = self._builder.width_cells
        out.info.height = self._builder.height_cells
        out.info.origin = Pose()
        out.info.origin.position.x = self._builder.params.origin_x_m
        out.info.origin.position.y = self._builder.params.origin_y_m
        out.info.origin.position.z = 0.0
        out.info.origin.orientation.w = 1.0
        # nav_msgs/OccupancyGrid data is row-major int8 of length H*W.
        out.data = grid_arr.flatten().tolist()
        self._pub.publish(out)


def main():
    rclpy.init()
    node = OccupancyGridNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
