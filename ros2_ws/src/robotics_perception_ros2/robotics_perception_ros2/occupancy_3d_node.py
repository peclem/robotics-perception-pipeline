"""
occupancy_3d_node — sparse 3D obstacle grid published to the ROS2 graph.

Subscribes
----------
  /perception/scene        vision_msgs/Detection3DArray
                           (from scene_graph_node — world-frame Detection3Ds)

Publishes
---------
  /perception/voxels       sensor_msgs/PointCloud2
                           One point per occupied voxel centre (always).
  /perception/octomap      octomap_msgs/Octomap
                           Sparse octree (only if octomap_msgs imports).

Why both wire formats
---------------------
- PointCloud2 is universal: rviz, Foxglove, any planner. Zero new deps.
- octomap_msgs/Octomap is the "right" 3D occupancy message for Nav2's
  octomap layer / movable obstacles work. The wire bytes are produced by
  the octomap C++ library, which has to be installed separately. We gate
  the publisher on a successful import so the rest of the graph runs
  regardless.

Lossiness caveat (same as occupancy_grid_node)
----------------------------------------------
We reconstruct minimal ObjectState equivalents from Detection3DArray
rather than running a second SceneGraph. Full KF covariance does not
round-trip through vision_msgs; the published grid uses only the
position-block covariance that survives. Intrinsics-aware inflation
would need /camera_info plumbing here too; this node stays
intrinsics-free and uses per-class fallbacks.
"""

from __future__ import annotations

import struct
import time
from typing import List, Optional

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
from vision_msgs.msg import Detection3DArray

from perception.config_loader import load_config
from world_model.object_state import ObjectState
from world_model.occupancy_3d import (
    Occupancy3D, Occupancy3DBuilder, Occupancy3DParams,
)


# Octomap publishing is gated on BOTH:
#   - octomap_msgs (provides the wire message type)
#   - octomap      (provides OcTree + binary serialisation)
# If either is missing we publish only the PointCloud2 — a message-type-
# without-tree-bytes Octomap would be dishonest.
try:
    from octomap_msgs.msg import Octomap as _OctomapMsg
    import octomap as _octomap
    _HAVE_OCTOMAP = True
except ImportError:
    _OctomapMsg = None
    _octomap = None
    _HAVE_OCTOMAP = False


def _detection3d_to_object_state(d) -> Optional[ObjectState]:
    """
    Reconstruct enough of an ObjectState for Occupancy3DBuilder.
    Mirrors occupancy_grid_node's helper exactly — same caveats.
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

    cov_8 = np.eye(8) * 10.0
    cov_8[:2, :2] = pose_cov[:2, :2]

    return ObjectState(
        track_id=tid,
        class_id=cls_id,
        class_name=cname or "unknown",
        position=np.array([0.0, 0.0]),
        covariance=cov_8,
        velocity=np.zeros(4),
        score=score,
        last_seen=time.monotonic(),
        n_updates=1,
        position_world=np.array([
            float(d.bbox.center.position.x),
            float(d.bbox.center.position.y),
            float(d.bbox.center.position.z),
        ]),
        max_history=1,
    )


def _occupancy_to_pointcloud2(
    grid: Occupancy3D, header: Header,
) -> PointCloud2:
    """Pack occupied voxel centres into a PointCloud2 (XYZ float32)."""
    centres = grid.voxel_centres_world().astype(np.float32)
    msg = PointCloud2()
    msg.header = header
    msg.height = 1
    msg.width  = int(centres.shape[0])
    msg.fields = [
        PointField(name="x", offset=0,  datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4,  datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8,  datatype=PointField.FLOAT32, count=1),
    ]
    msg.is_bigendian = False
    msg.point_step   = 12
    msg.row_step     = msg.point_step * msg.width
    msg.is_dense     = True
    msg.data         = centres.tobytes()
    return msg


class Occupancy3DNode(Node):

    def __init__(self, *, enable_intra_process: bool = False) -> None:
        # enable_intra_process kept for symmetry with the other adapter
        # nodes; rclpy Python doesn't expose it (C++ only).
        super().__init__("occupancy_3d_node")

        self.declare_parameter("config_path", "config/default.yaml")
        self.declare_parameter("frame_id", "map")

        cfg = load_config(self.get_parameter("config_path").value)
        self._frame_id = self.get_parameter("frame_id").value

        o3 = cfg.occupancy_3d
        params = Occupancy3DParams(
            resolution_m=o3.resolution_m,
            size_x_m=o3.size_x_m, size_y_m=o3.size_y_m, size_z_m=o3.size_z_m,
            origin_x_m=o3.origin_x_m, origin_y_m=o3.origin_y_m,
            origin_z_m=o3.origin_z_m,
            default_inflation_m=o3.default_inflation_m,
            min_inflation_m=o3.min_inflation_m,
            per_class_inflation_m=dict(o3.per_class_inflation_m),
        )
        self._builder = Occupancy3DBuilder(params, intrinsics=None)

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.create_subscription(
            Detection3DArray, "/perception/scene",
            self._on_scene, qos,
        )
        self._pc_pub = self.create_publisher(
            PointCloud2, "/perception/voxels", qos,
        )
        self._octomap_pub = None
        if _HAVE_OCTOMAP:
            self._octomap_pub = self.create_publisher(
                _OctomapMsg, "/perception/octomap", qos,
            )

        self.get_logger().info(
            f"occupancy_3d_node ready ("
            f"{self._builder.width_cells}×{self._builder.height_cells}"
            f"×{self._builder.depth_cells} voxels, "
            f"{params.resolution_m} m/voxel, "
            f"octomap={'on' if _HAVE_OCTOMAP else 'off (octomap+octomap_msgs not installed)'})"
        )

    def _on_scene(self, msg: Detection3DArray) -> None:
        objs: List[ObjectState] = []
        for d in msg.detections:
            obj = _detection3d_to_object_state(d)
            if obj is not None:
                objs.append(obj)

        grid = self._builder.build(objs)

        header = Header()
        header.stamp = msg.header.stamp
        header.frame_id = self._frame_id

        pc = _occupancy_to_pointcloud2(grid, header)
        self._pc_pub.publish(pc)

        if self._octomap_pub is not None:
            try:
                self._octomap_pub.publish(self._build_octomap_msg(grid, header))
            except Exception as exc:
                # Different octomap Python bindings expose subtly different
                # APIs (updateNode signatures, writeBinary return types).
                # If construction fails the PointCloud2 path still carries
                # the data; disable the octomap publisher to stop spamming.
                self.get_logger().warn(
                    f"Octomap serialisation failed ({exc!r}); "
                    "disabling /perception/octomap. PointCloud2 still active."
                )
                self._octomap_pub = None

    def _build_octomap_msg(self, grid: Occupancy3D, header: Header):
        """
        Build an OcTree from the occupied voxels and emit it as
        octomap_msgs/Octomap. Only called when both octomap and
        octomap_msgs imported successfully (see top-of-file guard).
        """
        tree = _octomap.OcTree(float(grid.params.resolution_m))
        for centre in grid.voxel_centres_world():
            tree.updateNode(centre.astype(np.float64), True, lazy_eval=True)
        tree.updateInnerOccupancy()

        m = _OctomapMsg()
        m.header = header
        m.binary = True
        m.id = "OcTree"
        m.resolution = float(grid.params.resolution_m)
        m.data = list(tree.writeBinary())
        return m


def main():
    rclpy.init()
    node = Occupancy3DNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
