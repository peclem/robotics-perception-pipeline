"""
Composite launch for the robotics-perception-pipeline ROS2 graph.

Brings up the full chain:
    camera_publisher → detection → tracking → scene_graph
                    \\→ pose

Run:
    ros2 launch robotics_perception_ros2 perception_pipeline.launch.py \\
        source:=video video_path:=/abs/path/to/clip.mp4

Topic graph:
    /perception/image_raw     (camera_publisher → detection, pose)
    /perception/camera_info   (camera_publisher → detection, tracking, pose)
    /perception/detections    (detection → tracking)
    /perception/tracks        (tracking → scene_graph)
    /perception/odom          (pose → scene_graph)
    /tf                       (pose, map→camera_frame)
    /perception/scene         (scene_graph, terminal output)
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
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

    common_params = [{"config_path": LaunchConfiguration("config_path")}]

    nodes = [
        Node(
            package="robotics_perception_ros2",
            executable="camera_publisher_node",
            name="camera_publisher",
            output="screen",
            parameters=common_params + [{
                "source":     LaunchConfiguration("source"),
                "video_path": LaunchConfiguration("video_path"),
            }],
        ),
        Node(
            package="robotics_perception_ros2",
            executable="detection_node",
            name="detection",
            output="screen",
            parameters=common_params,
        ),
        Node(
            package="robotics_perception_ros2",
            executable="tracking_node",
            name="tracking",
            output="screen",
            parameters=common_params,
        ),
        Node(
            package="robotics_perception_ros2",
            executable="pose_node",
            name="pose",
            output="screen",
            parameters=common_params,
        ),
        Node(
            package="robotics_perception_ros2",
            executable="scene_graph_node",
            name="scene_graph",
            output="screen",
            parameters=common_params,
        ),
        Node(
            package="robotics_perception_ros2",
            executable="occupancy_grid_node",
            name="occupancy_grid",
            output="screen",
            parameters=common_params,
        ),
        Node(
            package="robotics_perception_ros2",
            executable="health_monitor_node",
            name="health_monitor",
            output="screen",
            parameters=common_params,
        ),
    ]

    return LaunchDescription([source_arg, video_arg, config_arg, *nodes])
