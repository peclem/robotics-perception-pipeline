"""
ament_python setup for the robotics_perception_ros2 package.

Entry points register each adapter node so they can be launched
individually with `ros2 run` or composed in a launch file.
"""

from glob import glob
from setuptools import setup

package_name = "robotics_perception_ros2"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages",
         ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
        ("share/" + package_name + "/config", glob("config/*.rviz")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="peclem",
    maintainer_email="pechnyk.clement@gmail.com",
    description="ROS2 adapter nodes for the robotics-perception-pipeline.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "camera_publisher_node = "
            f"{package_name}.camera_publisher_node:main",
            "detection_node = "
            f"{package_name}.detection_node:main",
            "tracking_node = "
            f"{package_name}.tracking_node:main",
            "pose_node = "
            f"{package_name}.pose_node:main",
            "scene_graph_node = "
            f"{package_name}.scene_graph_node:main",
            "occupancy_grid_node = "
            f"{package_name}.occupancy_grid_node:main",
            "occupancy_3d_node = "
            f"{package_name}.occupancy_3d_node:main",
            "health_monitor_node = "
            f"{package_name}.health_monitor_node:main",
            "composite_node = "
            f"{package_name}.composite_node:main",
        ],
    },
)
