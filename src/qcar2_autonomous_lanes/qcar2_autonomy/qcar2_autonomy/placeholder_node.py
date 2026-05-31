#!/usr/bin/env python3
"""No-op node used by launch skeletons before algorithms are implemented."""
import rclpy
from rclpy.node import Node


class PlaceholderNode(Node):
    def __init__(self):
        super().__init__('qcar2_placeholder')
        self.declare_parameter('placeholder_role', 'placeholder')
        self.role = str(self.get_parameter('placeholder_role').value)
        self.get_logger().info(f'{self.role} placeholder started')
        self.create_timer(10.0, self._heartbeat)

    def _heartbeat(self):
        self.get_logger().debug(f'{self.role} placeholder alive')


def main(args=None):
    rclpy.init(args=args)
    node = PlaceholderNode()
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
