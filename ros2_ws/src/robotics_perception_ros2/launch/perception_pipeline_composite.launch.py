"""
Composite launch — all perception nodes in a single process.

Trade-off vs the multi-process variant
(perception_pipeline.launch.py):

  Multi-process              Composite
  =========================  =========================
  Crash isolation: full      Crash isolation: none
  Throughput: DDS-bound      Throughput: 2-3× higher
  CUDA contexts: N           CUDA contexts: 1
  Debug: per-node logs       Debug: one combined log

Both pipelines expose the same /perception/* topics + /diagnostics.

Run:
    ros2 launch robotics_perception_ros2 \\
        perception_pipeline_composite.launch.py \\
        source:=video \\
        video_path:=/abs/path/to/clip.mp4
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    source_arg = DeclareLaunchArgument(
        "source", default_value="video",
        description="Camera backend: 'video' | 'synthetic' | 'webcam'",
    )
    video_arg = DeclareLaunchArgument(
        "video_path", default_value="data/sample.mp4",
        description="Path to video file when source=video",
    )
    config_arg = DeclareLaunchArgument(
        "config_path", default_value="config/default.yaml",
        description="Project config YAML",
    )

    composite = Node(
        package="robotics_perception_ros2",
        executable="composite_node",
        name="perception_composite",
        output="screen",
        parameters=[{
            "source":      LaunchConfiguration("source"),
            "video_path":  LaunchConfiguration("video_path"),
            "config_path": LaunchConfiguration("config_path"),
        }],
    )

    return LaunchDescription([source_arg, video_arg, config_arg, composite])
