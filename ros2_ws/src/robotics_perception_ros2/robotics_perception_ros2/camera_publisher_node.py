"""
camera_publisher_node — wraps the existing CameraInterface.

Publishes
---------
  /perception/image_raw   sensor_msgs/Image
  /perception/camera_info sensor_msgs/CameraInfo

Source backend selected by parameter `source` ∈ {synthetic, video, webcam}.
All paths route through the same CameraInterface ABC the standalone
pipeline uses — no duplicated camera logic.

Parameters
----------
  source        : 'synthetic' | 'video' | 'webcam' (default: 'video')
  video_path    : when source='video', path to file
  config_path   : path to the project's YAML config (default: config/default.yaml)
  publish_rate  : Hz at which to drive the camera (default: from pipeline.target_hz)
"""

from __future__ import annotations

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import CameraInfo, Image

from perception.config_loader import load_config
from perception.camera_interface import (
    SyntheticCamera, VideoFileCamera, WebcamCamera,
)
from robotics_perception_ros2.image_bridge import numpy_to_imgmsg


class CameraPublisherNode(Node):

    def __init__(self) -> None:
        super().__init__("camera_publisher_node")

        self.declare_parameter("source", "video")
        self.declare_parameter("video_path", "data/sample.mp4")
        self.declare_parameter("config_path", "config/default.yaml")
        self.declare_parameter("frame_id", "camera_frame")

        source      = self.get_parameter("source").value
        video_path  = self.get_parameter("video_path").value
        config_path = self.get_parameter("config_path").value
        self._frame_id = self.get_parameter("frame_id").value

        cfg = load_config(config_path)
        raw = cfg.as_dict()

        if source == "synthetic":
            self._camera = SyntheticCamera(
                raw,
                width=cfg.synthetic_camera.width,
                height=cfg.synthetic_camera.height,
                num_frames=cfg.synthetic_camera.num_frames,
                fps=cfg.synthetic_camera.fps,
                num_objects=cfg.synthetic_camera.num_objects,
                seed=cfg.synthetic_camera.seed,
            )
        elif source == "video":
            self._camera = VideoFileCamera(
                raw,
                video_path=video_path,
                intrinsics_path=cfg.camera.intrinsics_path,
                loop=cfg.video.loop,
                playback_fps=cfg.video.playback_fps,
            )
        elif source == "webcam":
            self._camera = WebcamCamera(
                raw,
                device_index=cfg.camera.device_index,
                intrinsics_path=cfg.camera.intrinsics_path,
            )
        else:
            raise ValueError(
                f"camera_publisher_node: unknown source '{source}'. "
                "Use 'synthetic', 'video', or 'webcam'."
            )
        self._camera.open()

        # Sensor-data QoS: best-effort, keep last — standard for high-rate
        # streams where missing one frame is fine.
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self._pub_image = self.create_publisher(
            Image, "/perception/image_raw", qos,
        )
        self._pub_info = self.create_publisher(
            CameraInfo, "/perception/camera_info", qos,
        )

        period = 1.0 / cfg.pipeline.target_hz
        self._timer = self.create_timer(period, self._tick)
        self.get_logger().info(
            f"camera_publisher: source={source} → "
            f"/perception/image_raw at {cfg.pipeline.target_hz:.1f} Hz"
        )

    def _tick(self) -> None:
        frame = self._camera.get_frame()
        if frame is None:
            self.get_logger().info("camera exhausted, shutting down")
            self.destroy_timer(self._timer)
            rclpy.shutdown()
            return

        stamp = self.get_clock().now().to_msg()
        img_msg = numpy_to_imgmsg(
            frame.image, encoding="bgr8",
            stamp=stamp, frame_id=self._frame_id,
        )
        self._pub_image.publish(img_msg)

        info = CameraInfo()
        info.header.stamp = stamp
        info.header.frame_id = self._frame_id
        info.height = frame.intrinsics.height
        info.width  = frame.intrinsics.width
        # Pinhole, no distortion (project assumes pre-rectified streams).
        info.distortion_model = "plumb_bob"
        info.d = list(frame.intrinsics.dist_coeffs.tolist())
        info.k = [
            frame.intrinsics.fx, 0.0, frame.intrinsics.cx,
            0.0, frame.intrinsics.fy, frame.intrinsics.cy,
            0.0, 0.0, 1.0,
        ]
        info.p = [
            frame.intrinsics.fx, 0.0, frame.intrinsics.cx, 0.0,
            0.0, frame.intrinsics.fy, frame.intrinsics.cy, 0.0,
            0.0, 0.0, 1.0, 0.0,
        ]
        self._pub_info.publish(info)


def main():
    rclpy.init()
    node = CameraPublisherNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
