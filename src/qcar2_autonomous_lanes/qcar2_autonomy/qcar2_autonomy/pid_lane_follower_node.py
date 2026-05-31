#!/usr/bin/env python3
"""PID lane-follower: convert validated target_x (pixel) into /cmd_vel (Twist).

This is the missing closed-loop link between the spec's perception/validation
pipeline (which emits a pixel target on /planning/validated_target_x) and the
kinematics node / sim driver (which consumes geometry_msgs/Twist on /cmd_vel).

The error signal is the horizontal pixel offset between the target and the
camera principal point, normalized to [-1, 1] so the PID gains are independent
of image width. A classical PID then produces a yaw rate, paired with a
constant forward speed:

    e_t = (target_x - cx) / cx        # normalized pixel error
    u_t = Kp*e_t + Ki*sum(e_t)*dt + Kd*(e_t - e_{t-1})/dt
    angular.z = -u_t                  # negative: target right of center -> right turn
    linear.x  = base_speed
"""
import math

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from std_msgs.msg import Float32, String


LANE_KEEP = 'LANE_KEEP'
LANE_CHANGE = 'LANE_CHANGE'
LANE_RETURN = 'LANE_RETURN'


class PidLaneFollowerNode(Node):
    """Track a pixel target with a PID and emit Twist commands."""

    def __init__(self):
        super().__init__('pid_lane_follower_node')

        self.declare_parameter('kp', 0.9)
        self.declare_parameter('ki', 0.0)
        self.declare_parameter('kd', 0.05)
        self.declare_parameter('base_speed', 0.4)         # m/s forward
        self.declare_parameter('max_angular', 1.2)        # rad/s clamp on yaw
        self.declare_parameter('camera_center_x', 320.0)  # principal point in px
        self.declare_parameter('target_timeout_sec', 0.5) # safety: stop if stale
        self.declare_parameter('control_rate_hz', 20.0)
        self.declare_parameter('integral_clamp', 2.0)     # anti-windup bound on I

        # Manual swerve behavior when state machine signals LANE_CHANGE.
        # Real deployments hand this off to Nav2 DWA; this is a lightweight
        # in-node alternative so the sim demo shows visible avoidance without
        # standing up the full Nav2 stack.
        self.declare_parameter('lane_change_speed', 0.20)         # m/s during swerve
        self.declare_parameter('lane_change_angular', 0.6)        # rad/s, +ve = left
        self.declare_parameter('state_topic', '/system/current_state')

        self.declare_parameter('target_topic', '/planning/validated_target_x')
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')

        # Obstacle-avoidance lane bias. While the FSM is in LANE_CHANGE the
        # controller offsets its steering target sideways so the car follows the
        # road curvature (via vision) while hugging the clear side, instead of an
        # open-loop swerve that diverges on curves and gets re-centred by the
        # whole-road segmentation. The offset is ramped in/out to avoid a jerk.
        #   negative = shift target left  (avoid to the left)
        #   positive = shift target right (avoid to the right)
        self.declare_parameter('avoidance_offset_px', 0.0)
        self.declare_parameter('offset_ramp_sec', 1.5)
        # On a stale/lost vision target, creep slowly straight instead of a hard
        # stop, so the camera view keeps changing and the detector can re-acquire
        # the lane (a stationary car gets the same frame forever and never
        # recovers). 0.0 keeps the old full-stop behavior.
        self.declare_parameter('lost_creep_speed', 0.0)
        # gate_on_state True  -> stay silent unless LANE_KEEP (legacy mode where a
        #                         separate planner/DWA owns the wheel mid-maneuver).
        # gate_on_state False -> always drive and perform the lateral-offset lane
        #                         change in-controller.
        self.declare_parameter('gate_on_state', True)

        self.kp = float(self.get_parameter('kp').value)
        self.ki = float(self.get_parameter('ki').value)
        self.kd = float(self.get_parameter('kd').value)
        self.base_speed = float(self.get_parameter('base_speed').value)
        self.max_angular = float(self.get_parameter('max_angular').value)
        self.camera_center_x = float(self.get_parameter('camera_center_x').value)
        self.target_timeout_sec = float(
            self.get_parameter('target_timeout_sec').value
        )
        self.integral_clamp = float(self.get_parameter('integral_clamp').value)
        self.lane_change_speed = float(self.get_parameter('lane_change_speed').value)
        self.lane_change_angular = float(self.get_parameter('lane_change_angular').value)
        self.avoidance_offset_px = float(self.get_parameter('avoidance_offset_px').value)
        self.offset_ramp_sec = max(0.0, float(self.get_parameter('offset_ramp_sec').value))
        self.lost_creep_speed = float(self.get_parameter('lost_creep_speed').value)
        self.gate_on_state = bool(self.get_parameter('gate_on_state').value)
        self.current_offset = 0.0
        self._offset_time = None
        self.current_state = LANE_KEEP

        if self.camera_center_x <= 0.0:
            raise ValueError('camera_center_x must be > 0')

        # PID state
        self.last_target_x = math.nan
        self.last_target_time = self.get_clock().now()
        self.integral = 0.0
        self.last_error = 0.0
        self.last_step_time = self.get_clock().now()

        self.cmd_pub = self.create_publisher(
            Twist,
            self.get_parameter('cmd_vel_topic').value,
            10,
        )
        self.create_subscription(
            Float32,
            self.get_parameter('target_topic').value,
            self._on_target,
            10,
        )
        self.create_subscription(
            String,
            self.get_parameter('state_topic').value,
            self._on_state,
            10,
        )

        rate = float(self.get_parameter('control_rate_hz').value)
        self.create_timer(1.0 / rate, self._tick)

        self.get_logger().info(
            f'PID lane follower ready: kp={self.kp}, ki={self.ki}, kd={self.kd}, '
            f'base_speed={self.base_speed} m/s, '
            f'target={self.get_parameter("target_topic").value} -> '
            f'cmd={self.get_parameter("cmd_vel_topic").value}'
        )

    def _on_target(self, msg: Float32):
        self.last_target_x = float(msg.data)
        self.last_target_time = self.get_clock().now()

    def _on_state(self, msg: String):
        new_state = msg.data.strip().upper() or LANE_KEEP
        if new_state != self.current_state:
            self.get_logger().info(
                f'state transition {self.current_state} -> {new_state}'
            )
            # Reset PID integrator on transition so accumulated lane-keep error
            # does not bleed into the next steady-state phase.
            self.integral = 0.0
            self.last_error = 0.0
        self.current_state = new_state

    def _tick(self):
        now = self.get_clock().now()

        # Legacy mode (gate_on_state): a separate planner/DWA owns /cmd_vel
        # during LANE_CHANGE/LANE_RETURN, so we stay silent to keep one author.
        if self.gate_on_state and self.current_state != LANE_KEEP:
            return

        # Lateral-offset lane change: ramp the steering-target offset toward the
        # value commanded by the current FSM state. LANE_CHANGE biases the target
        # to the clear side; LANE_KEEP/LANE_RETURN ramp it back to centre. Vision
        # still supplies the curvature, so this tracks curved sections too.
        target_offset = (
            self.avoidance_offset_px if self.current_state == LANE_CHANGE else 0.0
        )
        self._ramp_offset(target_offset, now)

        # Safety: if the planner stopped publishing (perception lost both lines
        # or a node died), emit a zero Twist so the car coasts to a stop rather
        # than continuing to drive on a stale target.
        age = (now - self.last_target_time).nanoseconds * 1e-9
        if math.isnan(self.last_target_x) or age > self.target_timeout_sec:
            if self.lost_creep_speed > 0.0:
                # Creep straight so the view changes and vision can re-acquire.
                creep = Twist()
                creep.linear.x = self.lost_creep_speed
                self.cmd_pub.publish(creep)
            else:
                self._publish_stop()
            return

        # Normalized error in [-1, 1] for resolution-independent PID gains. The
        # ramped lateral offset shifts the effective target sideways during an
        # obstacle pass.
        error = (
            (self.last_target_x + self.current_offset - self.camera_center_x)
            / self.camera_center_x
        )

        dt = max(1e-3, (now - self.last_step_time).nanoseconds * 1e-9)
        self.last_step_time = now

        # Integral with symmetric clamp (anti-windup): stops error accumulating
        # to absurd magnitudes during long single-line stretches.
        self.integral += error * dt
        if self.integral > self.integral_clamp:
            self.integral = self.integral_clamp
        elif self.integral < -self.integral_clamp:
            self.integral = -self.integral_clamp

        derivative = (error - self.last_error) / dt
        self.last_error = error

        u = self.kp * error + self.ki * self.integral + self.kd * derivative

        # angular.z convention: +z is left (ROS REP-103). The pixel error is
        # positive when the lane center sits to the *right* of the camera
        # center, which requires a *right* (negative-z) turn -> negate.
        omega = -u
        if omega > self.max_angular:
            omega = self.max_angular
        elif omega < -self.max_angular:
            omega = -self.max_angular

        out = Twist()
        out.linear.x = self.base_speed
        out.angular.z = omega
        self.cmd_pub.publish(out)

    def _ramp_offset(self, target, now):
        """Move current_offset toward target at most |avoidance_offset_px|/ramp_sec."""
        if self.offset_ramp_sec <= 0.0:
            self.current_offset = target
            self._offset_time = now
            return
        dt = (
            (now - self._offset_time).nanoseconds * 1e-9
            if self._offset_time is not None else 0.0
        )
        self._offset_time = now
        max_step = abs(self.avoidance_offset_px) / self.offset_ramp_sec * max(0.0, dt)
        delta = target - self.current_offset
        if max_step <= 0.0 or abs(delta) <= max_step:
            self.current_offset = target
        else:
            self.current_offset += math.copysign(max_step, delta)

    def _publish_stop(self):
        self.cmd_pub.publish(Twist())


def main(args=None):
    rclpy.init(args=args)
    node = PidLaneFollowerNode()
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
