#!/usr/bin/env python3
"""Periodic text metrics for the QCar2 autonomy stack."""
import math

import rclpy
from qcar2_msgs.msg import BehaviorState, LaneModel, ObstacleState
from rclpy.node import Node


class MetricsLogger(Node):
    def __init__(self):
        super().__init__('metrics_logger')
        self.declare_parameter('lane_topic', '/qcar2/lane_model')
        self.declare_parameter('obstacle_topic', '/qcar2/obstacle_state')
        self.declare_parameter('behavior_topic', '/qcar2/behavior_state')
        self.declare_parameter('log_period_s', 2.0)

        self.last_lane = None
        self.last_obstacle = None
        self.last_behavior = None

        self.create_subscription(
            LaneModel,
            self.get_parameter('lane_topic').value,
            self._on_lane,
            10,
        )
        self.create_subscription(
            ObstacleState,
            self.get_parameter('obstacle_topic').value,
            self._on_obstacle,
            10,
        )
        self.create_subscription(
            BehaviorState,
            self.get_parameter('behavior_topic').value,
            self._on_behavior,
            10,
        )

        period = float(self.get_parameter('log_period_s').value)
        self.create_timer(max(period, 0.2), self._tick)

    def _on_lane(self, msg: LaneModel):
        self.last_lane = msg

    def _on_obstacle(self, msg: ObstacleState):
        self.last_obstacle = msg

    def _on_behavior(self, msg: BehaviorState):
        self.last_behavior = msg

    def _tick(self):
        lane = self.last_lane
        obstacle = self.last_obstacle
        behavior = self.last_behavior

        lane_text = 'lane=missing'
        if lane is not None:
            lane_text = (
                f'lane=detected:{lane.detected} conf={lane.confidence:.2f} '
                f'c0={lane.c0:+.3f}'
            )

        obstacle_text = 'obstacle=missing'
        if obstacle is not None:
            dist = 'inf' if not math.isfinite(obstacle.front_distance) else f'{obstacle.front_distance:.2f}m'
            obstacle_text = (
                f'obstacle=blocked:{obstacle.blocked} stale:{obstacle.stale} '
                f'front={dist}'
            )

        behavior_text = 'behavior=missing'
        if behavior is not None:
            behavior_text = (
                f'behavior={behavior.state} drive={behavior.drive_enabled} '
                f'reason={behavior.reason}'
            )

        self.get_logger().info(f'{lane_text} | {obstacle_text} | {behavior_text}')


def main(args=None):
    rclpy.init(args=args)
    node = MetricsLogger()
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
