"""
pose_node — wraps a PoseEstimator (NullPoseEstimator, DPVOPoseEstimator,
or VIOPoseEstimator when vio.enabled=true).

Subscribes
----------
  /perception/image_raw    sensor_msgs/Image
  /perception/camera_info  sensor_msgs/CameraInfo  (cached)

Publishes
---------
  /perception/odom         nav_msgs/Odometry      (map ← camera_frame)
  /tf                      tf2 broadcast          (map → camera_frame)

Backend selection is delegated to
`perception.pose_estimator_factory.build_pose_estimator`, which is
the same factory the standalone `launch.py` uses. So:

  pose_estimator.type='null'                → NullPoseEstimator
  pose_estimator.type='dpvo'                → DPVOPoseEstimator
  vio.enabled=true (any visual backend)     → VIOPoseEstimator wrapping
                                              the visual + IMU via the
                                              loosely-coupled error-
                                              state EKF (fused pose)

When VIO is on, the published /perception/odom + /tf are the fused
pose, not the bare visual measurement.
"""

from __future__ import annotations

import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import CameraInfo, Image
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster

from perception.camera_interface import CameraFrame, CameraIntrinsics
from perception.config_loader import load_config
from perception.pose_estimator_factory import build_pose_estimator
from robotics_perception_ros2.image_bridge import imgmsg_to_numpy


def _rot_to_quat(R: np.ndarray) -> tuple[float, float, float, float]:
    """3×3 rotation matrix → (qx, qy, qz, qw). Shepperd's method."""
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = 2.0 * np.sqrt(tr + 1.0)
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
    return float(qx), float(qy), float(qz), float(qw)


class PoseNode(Node):

    def __init__(self, *, enable_intra_process: bool = False) -> None:
        # Note: Python rclpy in Humble doesn't expose use_intra_process_comms
        # as a Node kwarg (C++ ComposableNodeContainer only). enable_intra_process
        # is kept on the signature so the composite launcher can pass it for
        # future-proofing; it's currently a no-op at this layer. Real savings
        # from the composite come from a single CUDA context + reduced
        # process overhead.
        super().__init__("pose_node")

        self.declare_parameter("config_path", "config/default.yaml")
        self.declare_parameter("world_frame", "map")
        self.declare_parameter("camera_frame", "camera_frame")
        cfg = load_config(self.get_parameter("config_path").value)
        raw = cfg.as_dict()

        self._world_frame = self.get_parameter("world_frame").value
        self._camera_frame = self.get_parameter("camera_frame").value

        # Single source of truth for pose-backend construction. Picks
        # up vio.enabled automatically — same code path as launch.py.
        self._estimator = build_pose_estimator(cfg, raw)
        self.get_logger().info(f"PoseEstimator: {self._estimator!r}")

        self._intrinsics: CameraIntrinsics | None = None
        self._frame_idx = 0

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
        self._odom_pub = self.create_publisher(
            Odometry, "/perception/odom", qos,
        )
        self._tf_broadcaster = TransformBroadcaster(self)

    def _on_info(self, msg: CameraInfo) -> None:
        self._intrinsics = CameraIntrinsics(
            fx=msg.k[0], fy=msg.k[4], cx=msg.k[2], cy=msg.k[5],
            width=msg.width, height=msg.height,
            dist_coeffs=np.asarray(msg.d, dtype=np.float64)
                if msg.d else np.zeros(5),
        )

    def _on_image(self, msg: Image) -> None:
        if self._intrinsics is None:
            return

        frame = CameraFrame(
            image=imgmsg_to_numpy(msg),
            timestamp=time.monotonic(),
            frame_idx=self._frame_idx,
            intrinsics=self._intrinsics,
            source_id="ros2",
        )
        self._frame_idx += 1

        pose = self._estimator.estimate(frame)
        if pose is None:
            return   # not yet initialised (DPVO bootstrap window)

        qx, qy, qz, qw = _rot_to_quat(pose.R)

        odom = Odometry()
        odom.header.stamp = msg.header.stamp
        odom.header.frame_id = self._world_frame
        odom.child_frame_id = self._camera_frame
        odom.pose.pose.position.x = float(pose.t[0])
        odom.pose.pose.position.y = float(pose.t[1])
        odom.pose.pose.position.z = float(pose.t[2])
        odom.pose.pose.orientation.x = qx
        odom.pose.pose.orientation.y = qy
        odom.pose.pose.orientation.z = qz
        odom.pose.pose.orientation.w = qw
        self._odom_pub.publish(odom)

        tf = TransformStamped()
        tf.header.stamp = msg.header.stamp
        tf.header.frame_id = self._world_frame
        tf.child_frame_id = self._camera_frame
        tf.transform.translation.x = float(pose.t[0])
        tf.transform.translation.y = float(pose.t[1])
        tf.transform.translation.z = float(pose.t[2])
        tf.transform.rotation.x = qx
        tf.transform.rotation.y = qy
        tf.transform.rotation.z = qz
        tf.transform.rotation.w = qw
        self._tf_broadcaster.sendTransform(tf)


def main():
    rclpy.init()
    node = PoseNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
