#!/usr/bin/env python3
"""Convert Nav2/DWA Twist commands to physical AckermannDrive commands."""
import math

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node

try:
    from ackermann_msgs.msg import AckermannDrive
except ImportError:  # pragma: no cover - depends on system ROS package install.
    AckermannDrive = None


class KinematicsNode(Node):
    """DWA Twist to Ackermann steering converter for the Quanser QCar2."""

    def __init__(self):
        super().__init__('kinematics_node')

        if AckermannDrive is None:
            raise RuntimeError(
                'ackermann_msgs is required. Install it with: '
                'sudo apt-get install ros-jazzy-ackermann-msgs'
            )

        self.declare_parameter('wheelbase', 0.256)
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('ackermann_topic', '/qcar/ackermann_cmd')
        self.declare_parameter('zero_speed_epsilon', 1.0e-6)

        self.wheelbase = float(self.get_parameter('wheelbase').value)
        self.zero_speed_epsilon = abs(
            float(self.get_parameter('zero_speed_epsilon').value)
        )

        self.ackermann_pub = self.create_publisher(
            AckermannDrive,
            self.get_parameter('ackermann_topic').value,
            10,
        )
        self.create_subscription(
            Twist,
            self.get_parameter('cmd_vel_topic').value,
            self._on_cmd_vel,
            10,
        )

        self.get_logger().info(
            'Kinematics node ready: '
            f'wheelbase={self.wheelbase}, '
            f'cmd_vel_topic={self.get_parameter("cmd_vel_topic").value}, '
            f'ackermann_topic={self.get_parameter("ackermann_topic").value}'
        )

    def _on_cmd_vel(self, msg: Twist):
        v = float(msg.linear.x)
        omega = float(msg.angular.z)

        steering_angle = self.compute_steering_angle(
            v,
            omega,
            self.wheelbase,
            self.zero_speed_epsilon,
        )

        out = AckermannDrive()
        out.steering_angle = float(steering_angle)
        out.speed = float(v)
        self.ackermann_pub.publish(out)

    @staticmethod
    def compute_steering_angle(v, omega, wheelbase, zero_speed_epsilon=1.0e-6):
        """Return Ackermann steer angle from unicycle velocity commands.

        Nav2/DWA publishes a body-frame unicycle command:
            v     = forward speed [m/s]
            omega = yaw rate [rad/s]

        For a bicycle/Ackermann model:
            omega = v * tan(delta) / L

        Solving for steering angle delta:
            delta = atan(L * omega / v)

        The formula is undefined at v=0. At a true stop, steering command is
        set to zero so the vehicle does not twitch the steering servo in place.
        """
        if abs(v) <= zero_speed_epsilon:
            return 0.0
        return math.atan((float(wheelbase) * float(omega)) / float(v))


def main(args=None):
    rclpy.init(args=args)
    node = KinematicsNode()
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
