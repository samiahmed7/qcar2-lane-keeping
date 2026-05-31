#!/usr/bin/env python3
"""Final command safety gate for QCar2 lane keeping."""
import math

import rclpy
from geometry_msgs.msg import Twist
from qcar2_msgs.msg import LaneModel
from rclpy.node import Node


def clamp(value, low, high):
    return max(low, min(high, value))


class SafetySupervisorNode(Node):
    def __init__(self):
        super().__init__('safety_supervisor_node')

        self.declare_parameter('lane_model_topic', '/qcar2/lane/model')
        self.declare_parameter('raw_cmd_topic', '/qcar2/control/raw_cmd_vel')
        self.declare_parameter('cmd_topic', '/model/qcar2/cmd_vel')
        self.declare_parameter('min_confidence', 0.3)
        self.declare_parameter('timeout_sec', 0.5)
        self.declare_parameter('control_rate', 20.0)
        self.declare_parameter('max_linear', 0.35)
        self.declare_parameter('max_angular', 0.8)

        self.min_confidence = float(self.get_parameter('min_confidence').value)
        self.timeout_sec = float(self.get_parameter('timeout_sec').value)
        self.max_linear = abs(float(self.get_parameter('max_linear').value))
        self.max_angular = abs(float(self.get_parameter('max_angular').value))

        self.last_lane = None
        self.last_lane_time = None
        self.last_raw_cmd = None
        self.last_raw_cmd_time = None

        self.cmd_pub = self.create_publisher(
            Twist,
            self.get_parameter('cmd_topic').value,
            10,
        )
        self.create_subscription(
            LaneModel,
            self.get_parameter('lane_model_topic').value,
            self._on_lane,
            10,
        )
        self.create_subscription(
            Twist,
            self.get_parameter('raw_cmd_topic').value,
            self._on_raw_cmd,
            10,
        )

        rate = max(float(self.get_parameter('control_rate').value), 1.0)
        self.create_timer(1.0 / rate, self._tick)

        self.get_logger().info(
            'Safety supervisor ready: '
            f'min_confidence={self.min_confidence}, timeout_sec={self.timeout_sec}, '
            f'max_linear={self.max_linear}, max_angular={self.max_angular}'
        )

    def _on_lane(self, msg: LaneModel):
        self.last_lane = msg
        self.last_lane_time = self.get_clock().now()

    def _on_raw_cmd(self, msg: Twist):
        self.last_raw_cmd = msg
        self.last_raw_cmd_time = self.get_clock().now()

    def _age_sec(self, stamp):
        if stamp is None:
            return math.inf
        return max(0.0, (self.get_clock().now() - stamp).nanoseconds * 1e-9)

    def _lane_is_unsafe(self):
        if self.last_lane is None:
            return True
        if self._age_sec(self.last_lane_time) > self.timeout_sec:
            return True
        if not math.isfinite(self.last_lane.confidence):
            return True
        return self.last_lane.confidence < self.min_confidence

    def _raw_cmd_is_stale(self):
        if self.last_raw_cmd is None:
            return True
        return self._age_sec(self.last_raw_cmd_time) > self.timeout_sec

    def _tick(self):
        if self._lane_is_unsafe() or self._raw_cmd_is_stale():
            self.cmd_pub.publish(Twist())
            return

        cmd = self._clamp_cmd(self.last_raw_cmd)
        self.cmd_pub.publish(cmd)

    def _clamp_cmd(self, raw_cmd):
        cmd = Twist()
        if math.isfinite(raw_cmd.linear.x):
            cmd.linear.x = clamp(raw_cmd.linear.x, -self.max_linear, self.max_linear)
        if math.isfinite(raw_cmd.angular.z):
            cmd.angular.z = clamp(raw_cmd.angular.z, -self.max_angular, self.max_angular)
        return cmd


def main(args=None):
    rclpy.init(args=args)
    node = SafetySupervisorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
