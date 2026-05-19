"""
composite_node — runs all perception adapter nodes in a single process.

Why
---
The multi-process launch (perception_pipeline.launch.py) gives the
clean "every node is its own ros2 process" topology, but its
throughput is bounded by DDS image serialisation (~2.7 MB per
1280×720 BGR frame, per subscriber). The watchdog flags this as
ERROR on /diagnostics — exactly the failure mode this composite
launcher fixes.

How
---
Instantiate every node with `enable_intra_process=True` and run
them under a single `MultiThreadedExecutor` in one Python process.
Publishers and subscribers with matching QoS in the same process
exchange messages by reference instead of serialising through DDS.
For 1280×720 BGR images that's ~3 MB / frame of avoided IPC work.

Caveats
-------
- Python rclpy's intra-process gains are smaller than C++'s
  (~2-3× rather than ~10×) due to GIL + Python-binding overhead.
- All nodes load into one process → CUDA contexts (detector, pose,
  appearance extractor) all share one context too, which is itself
  a meaningful win (~hundreds of MB of GPU memory).
- A crash in one node now takes down all of them. For production
  use, prefer per-node ros2-launch isolation; use this composite
  variant for throughput-bound demos and benchmarking.

Run
---
    ros2 launch robotics_perception_ros2 perception_pipeline_composite.launch.py
"""

from __future__ import annotations

import rclpy
from rclpy.executors import MultiThreadedExecutor

from robotics_perception_ros2.camera_publisher_node import CameraPublisherNode
from robotics_perception_ros2.detection_node       import DetectionNode
from robotics_perception_ros2.tracking_node        import TrackingNode
from robotics_perception_ros2.pose_node            import PoseNode
from robotics_perception_ros2.scene_graph_node     import SceneGraphNode
from robotics_perception_ros2.occupancy_grid_node  import OccupancyGridNode
from robotics_perception_ros2.occupancy_3d_node    import Occupancy3DNode
from robotics_perception_ros2.drivable_mask_node   import DrivableMaskNode
from robotics_perception_ros2.health_monitor_node  import HealthMonitorNode


def main() -> None:
    rclpy.init()

    # Order matters for diagnostic logging — instantiate the camera
    # producer first so the subscribers come up with a publisher to
    # discover. ROS2 discovery is async so it usually doesn't matter,
    # but it makes the startup log easier to read.
    nodes = [
        CameraPublisherNode(enable_intra_process=True),
        DetectionNode(enable_intra_process=True),
        TrackingNode(enable_intra_process=True),
        PoseNode(enable_intra_process=True),
        SceneGraphNode(enable_intra_process=True),
        OccupancyGridNode(enable_intra_process=True),
        Occupancy3DNode(enable_intra_process=True),
        DrivableMaskNode(enable_intra_process=True),
        HealthMonitorNode(enable_intra_process=True),
    ]

    executor = MultiThreadedExecutor(num_threads=len(nodes))
    for n in nodes:
        executor.add_node(n)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        for n in nodes:
            n.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
