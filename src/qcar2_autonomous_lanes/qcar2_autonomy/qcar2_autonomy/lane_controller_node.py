#!/usr/bin/env python3
"""Lane controller driven by /qcar2/lane/model with switchable controller_mode.

Modes:
  P                    Plain proportional on normalized pixel error.
  PURE_PURSUIT_SIMPLE  Image-space simplified pure pursuit:
                         angular_z = -kp * normalized_error / lookahead_factor
                       where lookahead_factor scales with commanded speed.
                       Smoother on real tracks; angular command is also
                       low-pass filtered before publishing.
"""
import math

import rclpy
from geometry_msgs.msg import Twist
from qcar2_msgs.msg import BehaviorState, LaneModel
from rclpy.node import Node


CONTROLLER_MODE_P = 'P'
CONTROLLER_MODE_PP_SIMPLE = 'PURE_PURSUIT_SIMPLE'
SUPPORTED_MODES = (CONTROLLER_MODE_P, CONTROLLER_MODE_PP_SIMPLE)


def clamp(value, low, high):
    return max(low, min(high, value))


class LaneControllerNode(Node):
    def __init__(self):
        super().__init__('lane_controller_node')

        self.declare_parameter('lane_model_topic', '/qcar2/lane/model')
        self.declare_parameter('behavior_state_topic', '/qcar2/behavior/state')
        self.declare_parameter('cmd_topic', '/qcar2/control/raw_cmd_vel')
        self.declare_parameter('kp', 1.15)
        self.declare_parameter('base_speed', 0.25)
        self.declare_parameter('min_speed', 0.12)
        self.declare_parameter('max_angular', 0.8)
        self.declare_parameter('image_half_width_px', 320.0)
        self.declare_parameter('min_confidence', 0.3)
        self.declare_parameter('timeout_sec', 0.8)
        self.declare_parameter('control_rate', 20.0)
        self.declare_parameter('large_error_slowdown_px', 240.0)
        self.declare_parameter('lane_width_m', 0.75)
        self.declare_parameter('lookahead_min_m', 0.55)
        self.declare_parameter('lookahead_max_m', 2.0)
        self.declare_parameter('lookahead_speed_gain', 1.8)
        self.declare_parameter('lookahead_curvature_gain', 3.0)
        self.declare_parameter('curvature_slowdown_gain', 1.4)
        self.declare_parameter('max_path_curvature', 3.0)
        self.declare_parameter('angular_filter_alpha', 0.35)
        # Controller architecture switch + simple-PP lookahead tuning.
        self.declare_parameter('controller_mode', CONTROLLER_MODE_PP_SIMPLE)
        self.declare_parameter('lookahead_base', 1.0)

        self.kp = float(self.get_parameter('kp').value)
        self.base_speed = float(self.get_parameter('base_speed').value)
        self.min_speed = float(self.get_parameter('min_speed').value)
        self.max_angular = abs(float(self.get_parameter('max_angular').value))
        self.image_half_width_px = max(
            1.0,
            float(self.get_parameter('image_half_width_px').value),
        )
        self.min_confidence = float(self.get_parameter('min_confidence').value)
        self.timeout_sec = float(self.get_parameter('timeout_sec').value)
        self.large_error_slowdown_px = max(
            1.0,
            float(self.get_parameter('large_error_slowdown_px').value),
        )
        self.lane_width_m = max(0.1, float(self.get_parameter('lane_width_m').value))
        self.lookahead_min_m = max(
            0.1,
            float(self.get_parameter('lookahead_min_m').value),
        )
        self.lookahead_max_m = max(
            self.lookahead_min_m,
            float(self.get_parameter('lookahead_max_m').value),
        )
        self.lookahead_speed_gain = max(
            0.0,
            float(self.get_parameter('lookahead_speed_gain').value),
        )
        self.lookahead_curvature_gain = max(
            0.0,
            float(self.get_parameter('lookahead_curvature_gain').value),
        )
        self.curvature_slowdown_gain = max(
            0.0,
            float(self.get_parameter('curvature_slowdown_gain').value),
        )
        self.max_path_curvature = max(
            0.1,
            float(self.get_parameter('max_path_curvature').value),
        )
        self.angular_filter_alpha = clamp(
            float(self.get_parameter('angular_filter_alpha').value),
            0.0,
            0.95,
        )
        requested_mode = str(self.get_parameter('controller_mode').value).strip().upper()
        if requested_mode not in SUPPORTED_MODES:
            self.get_logger().warn(
                f"Unknown controller_mode '{requested_mode}', "
                f"falling back to {CONTROLLER_MODE_PP_SIMPLE}"
            )
            requested_mode = CONTROLLER_MODE_PP_SIMPLE
        self.controller_mode = requested_mode
        self.lookahead_base = max(
            0.05,
            float(self.get_parameter('lookahead_base').value),
        )

        self.last_lane = None
        self.last_lane_time = None
        self.last_behavior = None
        self.last_behavior_time = None
        self.filtered_angular_z = 0.0

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
            BehaviorState,
            self.get_parameter('behavior_state_topic').value,
            self._on_behavior,
            10,
        )

        rate = max(float(self.get_parameter('control_rate').value), 1.0)
        self.create_timer(1.0 / rate, self._tick)

        self.get_logger().info(
            f'Lane controller ready: mode={self.controller_mode}, '
            f'kp={self.kp}, base_speed={self.base_speed}, '
            f'lookahead_base={self.lookahead_base}, '
            f'max_angular={self.max_angular}'
        )

    def _on_lane(self, msg: LaneModel):
        self.last_lane = msg
        self.last_lane_time = self.get_clock().now()

    def _on_behavior(self, msg: BehaviorState):
        self.last_behavior = msg
        self.last_behavior_time = self.get_clock().now()

    def _lane_age_sec(self):
        if self.last_lane_time is None:
            return math.inf
        return max(0.0, (self.get_clock().now() - self.last_lane_time).nanoseconds * 1e-9)

    def _behavior_age_sec(self):
        if self.last_behavior_time is None:
            return math.inf
        return max(0.0, (self.get_clock().now() - self.last_behavior_time).nanoseconds * 1e-9)

    def _tick(self):
        lane = self.last_lane
        behavior = self.last_behavior

        # Hard stop: behavior FSM says emergency. Publish zero raw cmd so the
        # safety supervisor sees a stale-free zero, not just a missing topic.
        if (
            behavior is not None
            and self._behavior_age_sec() <= self.timeout_sec
            and bool(behavior.emergency_stop)
        ):
            self._publish_stop()
            return

        if not self._lane_is_valid(lane):
            self._publish_stop()
            return

        # Speed setpoint from the FSM (falls back to base_speed if stale/None).
        speed_setpoint = self._speed_setpoint(behavior)
        normalized_error = lane.error_px / self.image_half_width_px

        if self.controller_mode == CONTROLLER_MODE_P:
            speed, angular_z = self._p_command(lane, normalized_error, speed_setpoint)
        else:
            speed, angular_z = self._pp_simple_command(lane, normalized_error, speed_setpoint)

        angular_z = self._filter_angular(angular_z)

        cmd = Twist()
        cmd.linear.x = float(speed)
        cmd.angular.z = float(angular_z)
        self.cmd_pub.publish(cmd)

    def _p_command(self, lane, normalized_error, setpoint):
        """Plain proportional controller on normalized pixel error."""
        angular_z = clamp(
            -self.kp * normalized_error,
            -self.max_angular,
            self.max_angular,
        )
        if setpoint <= 0.0:
            return 0.0, angular_z
        speed = setpoint
        if abs(lane.error_px) > self.large_error_slowdown_px:
            scale = self.large_error_slowdown_px / abs(lane.error_px)
            speed = max(self.min_speed, setpoint * scale)
        return clamp(speed, self.min_speed, setpoint), angular_z

    def _pp_simple_command(self, lane, normalized_error, setpoint):
        """Simplified image-space pure pursuit.

        angular_z = -kp * normalized_error / lookahead_factor

        lookahead_factor grows with commanded speed, which softens the steering
        as the car goes faster (longer effective lookahead). Curvature slowdown
        then reduces forward speed when the resulting steer is large, so the
        car arcs smoothly into tight corrections instead of slewing hard.
        """
        lookahead_factor = max(
            self.lookahead_base + self.lookahead_speed_gain * max(setpoint, 0.0),
            1e-3,
        )
        raw_angular = -self.kp * normalized_error / lookahead_factor
        angular_z = clamp(raw_angular, -self.max_angular, self.max_angular)

        if setpoint <= 0.0:
            return 0.0, angular_z
        curvature_scale = 1.0 / (1.0 + self.curvature_slowdown_gain * abs(angular_z))
        speed = setpoint * curvature_scale
        if abs(lane.error_px) > self.large_error_slowdown_px:
            error_scale = self.large_error_slowdown_px / abs(lane.error_px)
            speed *= error_scale
        return clamp(speed, self.min_speed, setpoint), angular_z

    def _speed_setpoint(self, behavior):
        if behavior is None or self._behavior_age_sec() > self.timeout_sec:
            return self.base_speed
        if not math.isfinite(behavior.desired_speed):
            return self.base_speed
        # Behavior chooses the cruise speed; controller still clamps it within
        # its own envelope so a misconfigured FSM cannot exceed safe limits.
        return clamp(float(behavior.desired_speed), 0.0, self.base_speed)

    def _lane_is_valid(self, lane):
        if lane is None:
            return False
        if self._lane_age_sec() > self.timeout_sec:
            return False
        if not math.isfinite(lane.confidence) or lane.confidence < self.min_confidence:
            return False
        return math.isfinite(lane.error_px)

    def _lateral_error_m(self, lane):
        # Image x grows to the right; vehicle y grows left. Flip sign here so
        # positive lateral error means "steer left" in the ROS base frame.
        return -lane.error_px * self._meters_per_pixel(lane)

    def _meters_per_pixel(self, lane):
        if (
            math.isfinite(lane.estimated_lane_width_px)
            and lane.estimated_lane_width_px > 20.0
        ):
            return self.lane_width_m / lane.estimated_lane_width_px
        return self.lane_width_m / (2.0 * self.image_half_width_px)

    def _curvature_hint(self, lane, lateral_error_m):
        lane_curvature = lane.curvature if math.isfinite(lane.curvature) else 0.0
        apparent_curvature = (
            2.0 * abs(lateral_error_m)
            / max(self.lookahead_max_m * self.lookahead_max_m, 1e-3)
        )
        return clamp(
            max(abs(lane_curvature), apparent_curvature),
            0.0,
            self.max_path_curvature,
        )

    def _lookahead_distance(self, speed, curvature_hint):
        speed_term = self.lookahead_min_m + self.lookahead_speed_gain * abs(speed)
        curvature_term = 1.0 + self.lookahead_curvature_gain * abs(curvature_hint)
        return clamp(
            speed_term / curvature_term,
            self.lookahead_min_m,
            self.lookahead_max_m,
        )

    def _target_speed(self, lane, path_curvature, setpoint):
        # If the FSM requested zero, honour it directly (e.g. emergency stop).
        if setpoint <= 0.0:
            return 0.0
        error_scale = clamp(
            self.large_error_slowdown_px / max(abs(lane.error_px), 1.0),
            self.min_speed / max(setpoint, 1e-3),
            1.0,
        )
        curvature_scale = 1.0 / (1.0 + self.curvature_slowdown_gain * abs(path_curvature))
        speed = setpoint * error_scale * curvature_scale
        return clamp(speed, self.min_speed, setpoint)

    def _filter_angular(self, angular_z):
        self.filtered_angular_z = (
            self.angular_filter_alpha * self.filtered_angular_z
            + (1.0 - self.angular_filter_alpha) * angular_z
        )
        return self.filtered_angular_z

    def _publish_stop(self):
        self.filtered_angular_z = 0.0
        self.cmd_pub.publish(Twist())


def main(args=None):
    rclpy.init(args=args)
    node = LaneControllerNode()
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
