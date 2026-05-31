#!/usr/bin/env python3
"""Timed lane-change behavior FSM for QCar2 autonomy."""
import math

import rclpy
from qcar2_msgs.msg import BehaviorState, LaneModel, ObstacleState
from rclpy.node import Node


KEEP_RIGHT = 'KEEP_RIGHT'
PREPARE_LEFT_CHANGE = 'PREPARE_LEFT_CHANGE'
CHANGE_LEFT = 'CHANGE_LEFT'
LEFT_LANE_KEEP = 'LEFT_LANE_KEEP'
RETURN_RIGHT = 'RETURN_RIGHT'
EMERGENCY_STOP = 'EMERGENCY_STOP'
RIGHT = 'RIGHT'
LEFT = 'LEFT'


class BehaviorFsmNode(Node):
    def __init__(self):
        super().__init__('behavior_fsm_node')

        self.declare_parameter('lane_model_topic', '/qcar2/lane/model')
        self.declare_parameter('obstacle_state_topic', '/qcar2/obstacle/state')
        self.declare_parameter('behavior_state_topic', '/qcar2/behavior/state')
        self.declare_parameter('min_lane_confidence', 0.3)
        self.declare_parameter('publish_rate', 10.0)
        # Speed planning per FSM state. desired_speed is kept for backward
        # compatibility but the per-state speeds below override it.
        self.declare_parameter('desired_speed', 0.25)
        self.declare_parameter('normal_speed', 0.25)
        self.declare_parameter('lane_change_speed', 0.18)
        self.declare_parameter('slow_speed', 0.12)
        self.declare_parameter('prepare_duration_sec', 0.4)
        self.declare_parameter('lane_change_duration_sec', 3.0)
        self.declare_parameter('min_pass_time_sec', 1.2)
        self.declare_parameter('clear_hold_sec', 0.8)
        self.declare_parameter('obstacle_timeout_sec', 0.8)

        self.min_lane_confidence = float(
            self.get_parameter('min_lane_confidence').value
        )
        self.desired_speed = float(self.get_parameter('desired_speed').value)
        self.normal_speed = max(
            0.0, float(self.get_parameter('normal_speed').value)
        )
        self.lane_change_speed = max(
            0.0, float(self.get_parameter('lane_change_speed').value)
        )
        self.slow_speed = max(
            0.0, float(self.get_parameter('slow_speed').value)
        )
        self.prepare_duration_sec = max(
            0.0,
            float(self.get_parameter('prepare_duration_sec').value),
        )
        self.lane_change_duration_sec = max(
            0.5,
            float(self.get_parameter('lane_change_duration_sec').value),
        )
        self.min_pass_time_sec = max(
            0.0,
            float(self.get_parameter('min_pass_time_sec').value),
        )
        self.clear_hold_sec = max(
            0.0,
            float(self.get_parameter('clear_hold_sec').value),
        )
        self.obstacle_timeout_sec = max(
            0.1,
            float(self.get_parameter('obstacle_timeout_sec').value),
        )

        self.state = KEEP_RIGHT
        self.target_lane = RIGHT
        self.state_entered = self.get_clock().now()
        self.clear_since = None
        self.last_lane = None
        self.last_obstacle = None
        self.last_obstacle_time = None

        self.pub = self.create_publisher(
            BehaviorState,
            self.get_parameter('behavior_state_topic').value,
            10,
        )
        self.create_subscription(
            LaneModel,
            self.get_parameter('lane_model_topic').value,
            self._on_lane,
            10,
        )
        self.create_subscription(
            ObstacleState,
            self.get_parameter('obstacle_state_topic').value,
            self._on_obstacle,
            10,
        )

        rate = max(float(self.get_parameter('publish_rate').value), 1.0)
        self.create_timer(1.0 / rate, self._tick)

        self.get_logger().info(
            'Behavior FSM ready: timed prepare/change/pass/return states enabled'
        )

    def _on_lane(self, msg: LaneModel):
        self.last_lane = msg

    def _on_obstacle(self, msg: ObstacleState):
        self.last_obstacle = msg
        self.last_obstacle_time = self.get_clock().now()

    def _tick(self):
        obstacle = self.last_obstacle
        emergency_stop = False

        if self._obstacle_stale():
            self._enter(EMERGENCY_STOP, RIGHT)
            emergency_stop = True
        elif self.state == KEEP_RIGHT:
            if obstacle.front_blocked and obstacle.left_clear:
                self._enter(PREPARE_LEFT_CHANGE, RIGHT)
            elif obstacle.front_blocked:
                self._enter(EMERGENCY_STOP, RIGHT)
                emergency_stop = True
            else:
                self.target_lane = RIGHT

        elif self.state == PREPARE_LEFT_CHANGE:
            if obstacle.front_blocked and not obstacle.left_clear:
                self._enter(EMERGENCY_STOP, RIGHT)
                emergency_stop = True
            elif not obstacle.front_blocked:
                self._enter(KEEP_RIGHT, RIGHT)
            elif self._state_age_sec() >= self.prepare_duration_sec:
                self._enter(CHANGE_LEFT, LEFT)

        elif self.state == CHANGE_LEFT:
            if not obstacle.left_clear:
                self._enter(EMERGENCY_STOP, LEFT)
                emergency_stop = True
            elif self._state_age_sec() >= self.lane_change_duration_sec:
                self._enter(LEFT_LANE_KEEP, LEFT)

        elif self.state == LEFT_LANE_KEEP:
            self.target_lane = LEFT
            if obstacle.front_blocked:
                self.clear_since = None
            elif obstacle.right_clear:
                if self.clear_since is None:
                    self.clear_since = self.get_clock().now()
                if (
                    self._state_age_sec() >= self.min_pass_time_sec
                    and self._age_sec(self.clear_since) >= self.clear_hold_sec
                ):
                    self._enter(RETURN_RIGHT, RIGHT)
            else:
                self.clear_since = None

        elif self.state == RETURN_RIGHT:
            if not obstacle.right_clear:
                self._enter(LEFT_LANE_KEEP, LEFT)
            elif self._state_age_sec() >= self.lane_change_duration_sec:
                self._enter(KEEP_RIGHT, RIGHT)

        elif self.state == EMERGENCY_STOP:
            emergency_stop = True
            if obstacle.front_blocked and obstacle.left_clear:
                self._enter(PREPARE_LEFT_CHANGE, RIGHT)
                emergency_stop = False
            elif not obstacle.front_blocked:
                self._enter(KEEP_RIGHT, RIGHT)
                emergency_stop = False

        if self.state == EMERGENCY_STOP:
            emergency_stop = True

        self._publish(emergency_stop)

    def _enter(self, state, target_lane):
        if state == self.state and target_lane == self.target_lane:
            return
        self.state = state
        self.target_lane = target_lane
        self.state_entered = self.get_clock().now()
        self.clear_since = None

    def _publish(self, emergency_stop):
        msg = BehaviorState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.state = self.state
        msg.target_lane = self.target_lane
        msg.emergency_stop = emergency_stop
        msg.desired_speed = float(self._plan_speed(emergency_stop))
        self.pub.publish(msg)

    def _plan_speed(self, emergency_stop):
        # Spec speed plan, evaluated top-down. Hard stops win over everything.
        if emergency_stop or self.state == EMERGENCY_STOP:
            return 0.0
        if self._lane_confidence_low():
            return self.slow_speed
        if self.state in (PREPARE_LEFT_CHANGE, CHANGE_LEFT, RETURN_RIGHT):
            return self.lane_change_speed
        if self.state == LEFT_LANE_KEEP:
            return self.normal_speed
        # KEEP_RIGHT and any future steady-state lane-keep
        return self.normal_speed

    def _lane_confidence_low(self):
        if self.last_lane is None:
            return False
        confidence = self.last_lane.confidence
        return (not math.isfinite(confidence)) or confidence < self.min_lane_confidence

    def _obstacle_stale(self):
        return (
            self.last_obstacle is None
            or self._age_sec(self.last_obstacle_time) > self.obstacle_timeout_sec
        )

    def _state_age_sec(self):
        return self._age_sec(self.state_entered)

    def _age_sec(self, stamp):
        if stamp is None:
            return math.inf
        return max(0.0, (self.get_clock().now() - stamp).nanoseconds * 1e-9)


def main(args=None):
    rclpy.init(args=args)
    node = BehaviorFsmNode()
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
