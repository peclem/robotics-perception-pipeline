"""
health_monitor_node — publishes per-stage diagnostics.

Subscribes
----------
  /perception/image_raw   sensor_msgs/Image       (camera_publisher latency)
  /perception/detections  vision_msgs/Detection2DArray  (detection latency)
  /perception/tracks      vision_msgs/Detection2DArray  (tracking latency)
  /perception/odom        nav_msgs/Odometry             (pose latency)
  /perception/scene       vision_msgs/Detection3DArray  (scene graph latency)
  /perception/costmap     nav_msgs/OccupancyGrid        (costmap latency)

Publishes
---------
  /diagnostics            diagnostic_msgs/DiagnosticArray
                          per-topic publish rate + latency + health.

Design note
-----------
The ROS2 graph runs each module as a separate process, so we can't
wrap each stage with a LatencyTracker context manager the way the
standalone launch.py does. Instead this node observes inter-arrival
times on each topic — a good proxy for "is each stage producing
output at the expected rate?". When the standalone pipeline is the
one running, its in-process HealthMonitor is the authoritative
source; the ROS2 node here is the equivalent for the multi-process
graph.

Stale topics (no message for `stale_after_s`) are flagged ERROR —
that's the production failure mode this node exists to catch.
"""

from __future__ import annotations

import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from diagnostic_msgs.msg import (
    DiagnosticArray, DiagnosticStatus, KeyValue,
)
from nav_msgs.msg import OccupancyGrid, Odometry
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, Detection3DArray

from perception.config_loader import load_config
from perception.health_monitor import (
    HealthMonitor, HealthStatus, LatencyTracker,
)


# Stage name (HealthMonitor key) → ROS topic + msg type.
# Tracked inter-arrival ≈ "this stage's output rate, in ms between msgs".
_TOPIC_STAGES = [
    ("camera",       "/perception/image_raw",   Image),
    ("detector",     "/perception/detections",  Detection2DArray),
    ("tracker",      "/perception/tracks",      Detection2DArray),
    ("pose",         "/perception/odom",        Odometry),
    ("scene_graph",  "/perception/scene",       Detection3DArray),
    ("costmap",      "/perception/costmap",     OccupancyGrid),
]


class HealthMonitorNode(Node):

    def __init__(self) -> None:
        super().__init__("health_monitor_node")

        self.declare_parameter("config_path", "config/default.yaml")
        self.declare_parameter("publish_period_s", 1.0)
        cfg = load_config(self.get_parameter("config_path").value)
        hm_cfg = cfg.health_monitor

        self._monitor = HealthMonitor()
        for name, _topic, _msg in _TOPIC_STAGES:
            self._monitor.register(
                name,
                # Topic inter-arrival budget = 1000 ms / target Hz.
                # Use the configured per-stage budget when present,
                # else fall back to the pipeline's frame budget.
                budget_ms=hm_cfg.stage_budgets_ms.get(
                    name, hm_cfg.default_budget_ms,
                ),
                window=hm_cfg.window,
                warn_after=hm_cfg.warn_after,
                error_after=hm_cfg.error_after,
                stale_after_s=hm_cfg.stale_after_s,
            )
        self._last_arrival_s: dict[str, float] = {}

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        # One subscription per tracked topic; only purpose is to time
        # message arrival, so the message contents are ignored.
        for name, topic, msg_type in _TOPIC_STAGES:
            self.create_subscription(
                msg_type, topic,
                lambda _msg, n=name: self._on_message(n), qos,
            )

        self._pub = self.create_publisher(
            DiagnosticArray, "/diagnostics", 10,
        )
        period = float(self.get_parameter("publish_period_s").value)
        self.create_timer(period, self._publish_diagnostics)
        self.get_logger().info(
            f"health_monitor_node ready ({len(_TOPIC_STAGES)} topics, "
            f"diagnostics @ {1.0/period:.1f} Hz)"
        )

    def _on_message(self, stage: str) -> None:
        now = time.monotonic()
        prev = self._last_arrival_s.get(stage)
        self._last_arrival_s[stage] = now
        if prev is not None:
            interval_ms = (now - prev) * 1000.0
            tracker = self._monitor.get(stage)
            if tracker is not None:
                tracker.observe(interval_ms)

    def _publish_diagnostics(self) -> None:
        out = DiagnosticArray()
        out.header.stamp = self.get_clock().now().to_msg()
        for report in self._monitor.reports():
            ds = DiagnosticStatus()
            ds.name = f"perception/{report.name}"
            ds.hardware_id = "robotics_perception_pipeline"
            ds.level = bytes([int(report.status)])
            ds.message = report.message
            ds.values = [
                KeyValue(key="median_ms", value=f"{report.median_ms:.2f}"),
                KeyValue(key="p95_ms",    value=f"{report.p95_ms:.2f}"),
                KeyValue(key="max_ms",    value=f"{report.max_ms:.2f}"),
                KeyValue(key="last_ms",   value=f"{report.last_ms:.2f}"),
                KeyValue(key="budget_ms", value=f"{report.budget_ms:.2f}"),
                KeyValue(key="n_observations",
                         value=str(report.n_observations)),
                KeyValue(key="n_breaches",
                         value=str(report.n_breaches)),
            ]
            out.status.append(ds)
        self._pub.publish(out)


def main():
    rclpy.init()
    node = HealthMonitorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
