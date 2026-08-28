#!/usr/bin/env python3
"""Print live distance AND relative direction from current pose (via TF,
map -> base_link) to a fixed target (x0, y0), phrased relative to the car's
current heading. Works with any localization backend that publishes the
map -> base_link TF chain (AMCL or Cartographer).
Usage: python3 dist_to_start.py <x0> <y0>
"""
import sys
import math

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.time import Time
from tf2_ros import Buffer, TransformListener, TransformException


class DistToStart(Node):
    def __init__(self, x0, y0):
        super().__init__('dist_to_start_watcher')
        self.x0 = x0
        self.y0 = y0
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self.create_timer(0.1, self.tick)
        self.get_logger().info(f'Watching distance to start=({x0:.3f}, {y0:.3f})')

    def tick(self):
        try:
            tr = self._tf_buffer.lookup_transform(
                'map', 'base_link', Time(), timeout=Duration(seconds=0.1))
        except TransformException:
            return

        x = tr.transform.translation.x
        y = tr.transform.translation.y
        q = tr.transform.rotation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                          1.0 - 2.0 * (q.y * q.y + q.z * q.z))

        dx = self.x0 - x
        dy = self.y0 - y
        dist = math.hypot(dx, dy)

        x_body = dx * math.cos(yaw) + dy * math.sin(yaw)
        y_body = -dx * math.sin(yaw) + dy * math.cos(yaw)

        fwd_back = 'AHEAD' if x_body >= 0 else 'BEHIND'
        left_right = 'LEFT' if y_body >= 0 else 'RIGHT'
        bearing_deg = math.degrees(math.atan2(y_body, x_body))

        marker = '  <<< CLOSE ENOUGH, STOP' if dist < 0.10 else ''
        print(f'DIST={dist:6.3f}m   target is {fwd_back} and to your {left_right} '
              f'({abs(bearing_deg):5.1f} deg off nose){marker}')


def main():
    if len(sys.argv) < 3:
        print('Usage: python3 dist_to_start.py <x0> <y0>')
        sys.exit(1)

    x0 = float(sys.argv[1])
    y0 = float(sys.argv[2])

    rclpy.init()
    node = DistToStart(x0, y0)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
