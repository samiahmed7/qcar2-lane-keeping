#!/usr/bin/env python3
"""Bang-zero-bang lane-change planner: turn, drive straight, counter-turn.

Three constant-twist phases in sequence (no smooth math, no curves):

    phase A (turn_duration)     : omega = +step_angular   -> head into new lane
    phase B (straight_duration) : omega = 0               -> drive straight,
                                                             gains lateral
    phase C (turn_duration)     : omega = -step_angular   -> straighten back

After phase C the planner sets maneuver_active=False and the lane-keeper
(RF-DETR + PID) takes the wheel. By construction phase C cancels phase A's
rotation, so the car finishes near its original heading and the PID just has
to lane-keep on whichever lane it now sees.

LANE_RETURN runs the same three phases with the sign of step_angular flipped.
"""
import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from std_msgs.msg import Bool, Float32MultiArray, String


LANE_KEEP = 'LANE_KEEP'
LANE_CHANGE = 'LANE_CHANGE'
LANE_RETURN = 'LANE_RETURN'
MANEUVER_ACTIVE_STATES = (LANE_CHANGE, LANE_RETURN)


class LaneChangePlannerNode(Node):

    def __init__(self):
        super().__init__('lane_change_planner_node')

        self.declare_parameter('cruise_speed', 0.30)
        self.declare_parameter('step_angular', 1.0)        # rad/s, sign = direction
        self.declare_parameter('turn_duration', 0.6)       # phase A and C length, s
        self.declare_parameter('straight_duration', 1.2)   # phase B length, s
        self.declare_parameter('max_angular', 1.5)
        self.declare_parameter('publish_rate', 50.0)
        self.declare_parameter('cmd_topic', '/qcar2/control/maneuver_cmd_vel')
        self.declare_parameter('state_topic', '/system/current_state')
        self.declare_parameter('odom_topic', '/model/qcar2/odometry')
        self.declare_parameter('complete_topic', '/qcar2/control/maneuver_complete')
        self.declare_parameter('active_topic', '/qcar2/control/maneuver_active')
        self.declare_parameter('params_topic', '/qcar2/control/lane_change_params')

        self.cruise_speed = float(self.get_parameter('cruise_speed').value)
        self.step_angular = float(self.get_parameter('step_angular').value)
        self.turn_duration = max(0.05, float(self.get_parameter('turn_duration').value))
        self.straight_duration = max(0.0, float(self.get_parameter('straight_duration').value))
        self.duration = 2.0 * self.turn_duration + self.straight_duration
        self.max_angular = float(self.get_parameter('max_angular').value)

        self.current_state = LANE_KEEP
        self.start_time = None
        self.direction = 0.0      # sign of the active turn: +1 left, -1 right
        self.maneuver_running = False
        # Direction + straight-phase duration commanded by the FSM (which sizes
        # the shift from the measured obstacle width and picks the clear side).
        # Defaults reproduce the static behavior if no command has arrived.
        self.cmd_direction = 1.0
        self.cmd_straight_duration = self.straight_duration

        self.cmd_pub = self.create_publisher(
            Twist,
            self.get_parameter('cmd_topic').value,
            10,
        )
        self.complete_pub = self.create_publisher(
            Bool,
            self.get_parameter('complete_topic').value,
            10,
        )
        self.active_pub = self.create_publisher(
            Bool,
            self.get_parameter('active_topic').value,
            10,
        )
        self.completion_signaled = False
        self.last_active_state = False
        self.create_subscription(
            String,
            self.get_parameter('state_topic').value,
            self._on_state,
            10,
        )
        self.create_subscription(
            Float32MultiArray,
            self.get_parameter('params_topic').value,
            self._on_params,
            10,
        )

        rate = max(1.0, float(self.get_parameter('publish_rate').value))
        self.create_timer(1.0 / rate, self._tick)

        self.get_logger().info(
            f'Bang-zero-bang lane-change planner ready: '
            f'v={self.cruise_speed} m/s, step_omega={self.step_angular} rad/s, '
            f'turn={self.turn_duration} s + straight={self.straight_duration} s + '
            f'turn={self.turn_duration} s '
            f'(total {self.duration} s), max_omega={self.max_angular} rad/s'
        )

    def _on_params(self, msg: Float32MultiArray):
        """FSM-commanded [direction(+1 left/-1 right), straight_phase_sec]."""
        if len(msg.data) >= 1 and msg.data[0] != 0.0:
            self.cmd_direction = 1.0 if msg.data[0] > 0.0 else -1.0
        if len(msg.data) >= 2 and msg.data[1] > 0.0:
            self.cmd_straight_duration = max(0.0, float(msg.data[1]))

    def _apply_shift_size(self):
        """Adopt the commanded straight-phase duration for this maneuver."""
        self.straight_duration = self.cmd_straight_duration
        self.duration = 2.0 * self.turn_duration + self.straight_duration

    def _on_state(self, msg: String):
        new_state = msg.data.strip().upper() or LANE_KEEP
        if new_state == self.current_state:
            return
        self.current_state = new_state

        if new_state == LANE_CHANGE:
            self._apply_shift_size()
            self._begin_maneuver(direction=self.cmd_direction)
            side = 'LEFT' if self.cmd_direction > 0 else 'RIGHT'
            self.get_logger().info(
                f'step LANE_CHANGE: dodge {side}, omega='
                f'{self.cmd_direction * self.step_angular:+.2f} rad/s for {self.duration:.1f} s'
            )
            return

        if new_state == LANE_RETURN:
            # Mirror the change: same magnitude, opposite direction.
            self._apply_shift_size()
            self._begin_maneuver(direction=-self.cmd_direction)
            self.get_logger().info(
                f'step LANE_RETURN: omega='
                f'{-self.cmd_direction * self.step_angular:+.2f} rad/s for {self.duration:.1f} s'
            )
            return

        if new_state == LANE_KEEP:
            self.maneuver_running = False

    def _begin_maneuver(self, direction: float):
        self.direction = float(direction)
        self.start_time = self.get_clock().now()
        self.maneuver_running = True
        self.completion_signaled = False
        self._publish_active(True)

    def _tick(self):
        if self.current_state not in MANEUVER_ACTIVE_STATES:
            return  # mux ignores this topic during LANE_KEEP
        if not self.maneuver_running or self.start_time is None:
            # Trigger fired but no maneuver running yet: creep forward straight
            # rather than freeze in front of the obstacle.
            twist = Twist()
            twist.linear.x = self.cruise_speed
            self.cmd_pub.publish(twist)
            return

        now = self.get_clock().now()
        elapsed = max(0.0, (now - self.start_time).nanoseconds * 1e-9)

        if elapsed >= self.duration:
            # All three phases done. Hand the wheel back to the lane-keeper.
            if not self.completion_signaled:
                done = Bool()
                done.data = True
                self.complete_pub.publish(done)
                self.completion_signaled = True
                self._publish_active(False)
                self.maneuver_running = False
            return

        # Phase A: turn into the new lane
        if elapsed < self.turn_duration:
            omega = self.direction * self.step_angular
        # Phase B: hold heading, drive straight (this is where the lateral
        # shift accumulates -- the car moves diagonally for straight_duration).
        elif elapsed < self.turn_duration + self.straight_duration:
            omega = 0.0
        # Phase C: counter-turn to straighten back out
        else:
            omega = -self.direction * self.step_angular

        self._publish(omega)

    def _publish(self, omega):
        omega = max(-self.max_angular, min(self.max_angular, float(omega)))
        twist = Twist()
        twist.linear.x = self.cruise_speed
        twist.angular.z = omega
        self.cmd_pub.publish(twist)

    def _publish_active(self, active: bool):
        if active == self.last_active_state:
            return
        msg = Bool()
        msg.data = bool(active)
        self.active_pub.publish(msg)
        self.last_active_state = bool(active)


def main(args=None):
    rclpy.init(args=args)
    node = LaneChangePlannerNode()
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
