#!/usr/bin/env python3
"""Bridge std_msgs/String FSM state to qcar2_msgs/BehaviorState.

state_machine_node publishes the high-level FSM on /system/current_state as
a String (LANE_KEEP / LANE_CHANGE / LANE_RETURN). The classical control nodes
(bev_lane_detector_node, lane_controller_node) consume the richer custom type
qcar2_msgs/BehaviorState. This node converts between them at a fixed 10 Hz so
the downstream consumers always see a fresh value.

Mapping:
    String              -> BehaviorState
    -------                -------------
    LANE_KEEP              state=KEEP_RIGHT,  emergency_stop=False
    LANE_CHANGE            state=CHANGE_LEFT, emergency_stop=True   (DWA owns /cmd_vel)
    LANE_RETURN            state=CHANGE_RIGHT,emergency_stop=True   (DWA owns /cmd_vel)
    (anything else)        state=KEEP_RIGHT,  emergency_stop=False

emergency_stop=True is what tells lane_controller_node to publish zero Twist
during the maneuver; the mux node further drops those publishes so DWA's
output reaches the simulator unopposed.
"""
import rclpy
from qcar2_msgs.msg import BehaviorState
from rclpy.node import Node
from std_msgs.msg import String


LANE_KEEP = 'LANE_KEEP'
LANE_CHANGE = 'LANE_CHANGE'
LANE_RETURN = 'LANE_RETURN'


class StateToBehaviorBridgeNode(Node):

    def __init__(self):
        super().__init__('state_to_behavior_bridge_node')

        self.declare_parameter('state_topic', '/system/current_state')
        self.declare_parameter('behavior_topic', '/qcar2/behavior/state')
        self.declare_parameter('desired_speed', 0.25)
        self.declare_parameter('rate_hz', 10.0)

        self.desired_speed = float(self.get_parameter('desired_speed').value)
        self.current = LANE_KEEP

        self.pub = self.create_publisher(
            BehaviorState,
            self.get_parameter('behavior_topic').value,
            10,
        )
        self.create_subscription(
            String,
            self.get_parameter('state_topic').value,
            self._on_state,
            10,
        )
        rate = float(self.get_parameter('rate_hz').value)
        self.create_timer(1.0 / rate, self._tick)

        self.get_logger().info(
            f'state -> behavior bridge ready: '
            f'in={self.get_parameter("state_topic").value} '
            f'out={self.get_parameter("behavior_topic").value}'
        )

    def _on_state(self, msg: String):
        text = msg.data.strip().upper() or LANE_KEEP
        if text != self.current:
            self.get_logger().info(f'state {self.current} -> {text}')
        self.current = text

    def _tick(self):
        msg = BehaviorState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.desired_speed = self.desired_speed
        if self.current == LANE_CHANGE:
            msg.state = 'CHANGE_LEFT'
            msg.target_lane = 'LEFT'
            msg.emergency_stop = True   # silences lane_controller
        elif self.current == LANE_RETURN:
            msg.state = 'CHANGE_RIGHT'
            msg.target_lane = 'RIGHT'
            msg.emergency_stop = True
        else:
            msg.state = 'KEEP_RIGHT'
            msg.target_lane = 'RIGHT'
            msg.emergency_stop = False
        self.pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = StateToBehaviorBridgeNode()
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
