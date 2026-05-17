"""
scene_graph_node — wraps world_model.scene_graph.SceneGraph.

Subscribes
----------
  /perception/tracks   vision_msgs/Detection2DArray
                       (Detection2D.id encoded as 'track_id:class_name')
  /perception/odom     nav_msgs/Odometry          (optional, for world frame)

Publishes
---------
  /perception/scene    vision_msgs/Detection3DArray
                       with per-object position + position covariance
                       (subset of full KF covariance — see caveat below)

Caveat — lossiness across the ROS boundary
------------------------------------------
The full SceneGraph internally relies on Track objects carrying 8×8 KF
state. Detection2DArray only carries bbox geometry, not KF covariance.
This node reconstructs minimal Track-equivalent state with default
identity covariance to update SceneGraph. The published Detection3DArray
therefore reflects geometric tracking, not the standalone pipeline's
full uncertainty model. A production deployment would use a custom
message carrying the (4×4) position+velocity covariance block.

Without ego-pose, positions stay in pixel coordinates and Detection3D
fills only z=0 (no metric depth from this node yet — depth wiring is
deliberately deferred to keep this PR scope-clean).
"""

from __future__ import annotations

import time
from collections import OrderedDict
from typing import Optional

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from nav_msgs.msg import Odometry
from vision_msgs.msg import (
    Detection2DArray, Detection3D, Detection3DArray,
    BoundingBox3D, ObjectHypothesisWithPose, ObjectHypothesis,
)

from perception.config_loader import load_config
from perception.pose_estimator import CameraPose
from perception.transform_tree import TransformTree
from tracking.track import Track, TrackState
from perception.detector import Detection
from world_model.scene_graph import SceneGraph


def _quat_to_rot(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    return np.array([
        [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw),     1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw),     1 - 2*(qx*qx + qy*qy)],
    ], dtype=np.float64)


class _SyntheticTrack:
    """
    Minimal Track stand-in for SceneGraph._upsert.

    Carries only the fields _upsert reads. Covariance defaults to
    identity-scaled — the published Detection3D's pose covariance will
    therefore not reflect the standalone pipeline's KF uncertainty.
    Acknowledged in module docstring.
    """
    class _KF:
        def snapshot(self, timestamp, frame_idx):
            from state_estimation.kalman_filter import KFSnapshot
            return KFSnapshot(
                timestamp=timestamp, frame_idx=frame_idx,
                state=np.zeros(8), covariance=np.eye(8),
                nis=0.0, n_updates=1,
            )

    def __init__(self, track_id, class_id, class_name, position, score):
        self.track_id   = int(track_id)
        self.class_id   = int(class_id)
        self.class_name = str(class_name)
        self.position   = np.asarray(position, dtype=np.float64)
        self.covariance = np.eye(8, dtype=np.float64) * 10.0
        self.velocity   = np.zeros(4, dtype=np.float64)
        self.score      = float(score)
        self.n_hits     = 1
        self.kf         = _SyntheticTrack._KF()


class SceneGraphNode(Node):

    def __init__(self) -> None:
        super().__init__("scene_graph_node")

        self.declare_parameter("config_path", "config/default.yaml")
        cfg = load_config(self.get_parameter("config_path").value)
        raw = cfg.as_dict()

        # Build transform tree if config enables it.
        tree: Optional[TransformTree] = None
        if cfg.coordinate_frames.enabled:
            tree = TransformTree(root_frame=cfg.coordinate_frames.root_frame)
            for ext in cfg.coordinate_frames.static_extrinsics:
                R = np.asarray(ext.R, dtype=np.float64).reshape(3, 3)
                t = np.asarray(ext.t, dtype=np.float64)
                tree.set_static(ext.parent, ext.child, R, t)

        self._sg = SceneGraph(raw, transform_tree=tree)
        self._tree = tree
        self._camera_pose: Optional[CameraPose] = None

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.create_subscription(
            Detection2DArray, "/perception/tracks",
            self._on_tracks, qos,
        )
        self.create_subscription(
            Odometry, "/perception/odom",
            self._on_odom, qos,
        )
        self._pub = self.create_publisher(
            Detection3DArray, "/perception/scene", qos,
        )
        self.get_logger().info(
            f"scene_graph_node ready (transform_tree={'on' if tree else 'off'})"
        )

    def _on_odom(self, msg: Odometry) -> None:
        q = msg.pose.pose.orientation
        p = msg.pose.pose.position
        R = _quat_to_rot(q.x, q.y, q.z, q.w)
        self._camera_pose = CameraPose(
            R=R, t=np.array([p.x, p.y, p.z], dtype=np.float64),
            timestamp=time.monotonic(),
            frame_idx=0,
            confidence=1.0, source="ros2-odom",
        )
        if self._tree is not None:
            self._tree.update_from_camera_pose(self._camera_pose)

    def _on_tracks(self, msg: Detection2DArray) -> None:
        synthetic_tracks: list = []
        for d in msg.detections:
            # id is "track_id:class_name" from tracking_node
            tid_str, _, cname = (d.id or "0:unknown").partition(":")
            try:
                tid = int(tid_str)
            except ValueError:
                continue
            score = d.results[0].hypothesis.score if d.results else 0.0
            cls_id = (
                int(d.results[0].hypothesis.class_id)
                if d.results and d.results[0].hypothesis.class_id.isdigit()
                else 0
            )
            synthetic_tracks.append(_SyntheticTrack(
                track_id=tid, class_id=cls_id, class_name=cname or "unknown",
                position=[d.bbox.center.position.x, d.bbox.center.position.y],
                score=score,
            ))

        self._sg.update(
            confirmed_tracks=synthetic_tracks,
            lost_tracks=[],
            timestamp=time.monotonic(),
            depth_estimates=None,
            camera_pose=self._camera_pose,
        )

        out = Detection3DArray()
        out.header = msg.header
        for obj in self._sg.all_objects(include_lost=False):
            det = Detection3D()
            det.header = msg.header
            det.id = f"{obj.track_id}:{obj.class_name}"

            bbox = BoundingBox3D()
            if obj.position_world is not None:
                bbox.center.position.x = float(obj.position_world[0])
                bbox.center.position.y = float(obj.position_world[1])
                bbox.center.position.z = float(obj.position_world[2])
            else:
                # Camera-frame pixels — z=0. Downstream consumers should
                # check the per-object frame via .header.frame_id which we
                # do NOT change here. Document this limitation.
                bbox.center.position.x = float(obj.position[0])
                bbox.center.position.y = float(obj.position[1])
                bbox.center.position.z = 0.0
            bbox.center.orientation.w = 1.0
            bbox.size.x = bbox.size.y = bbox.size.z = 1.0
            det.bbox = bbox

            det.results = [ObjectHypothesisWithPose(
                hypothesis=ObjectHypothesis(
                    class_id=str(obj.class_id), score=float(obj.score),
                )
            )]
            # 2D position covariance, embedded in the 6×6 pose covariance.
            cov = np.zeros((6, 6))
            cov[:2, :2] = obj.covariance[:2, :2]
            det.results[0].pose.covariance = cov.flatten().tolist()
            out.detections.append(det)
        self._pub.publish(out)


def main():
    rclpy.init()
    node = SceneGraphNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
