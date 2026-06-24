#!/usr/bin/env python3
"""Pure BEV/HSV lane keeper — right lane only.

Subscribes to the LaneModel published by bev_lane_detector_node and
drives the car using a simple P controller on error_px (pixels from
the target right-lane centre to the image centre).

Run with:
    python3 scripts/bev_lane_keeper.py
"""
import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from qcar2_msgs.msg import LaneModel


SPEED = 0.30          # m/s forward speed
KP    = 0.0014        # rad/s per pixel error  (tune this)
MIN_CONF = 0.15       # below this confidence → hold heading, slow down
TIMEOUT  = 0.5        # seconds without lane msg → stop


class BevLaneKeeper(Node):
    def __init__(self):
        super().__init__('bev_lane_keeper')
        self.pub = self.create_publisher(Twist, '/model/qcar2/cmd_vel', 10)
        self.create_subscription(LaneModel, '/qcar2/lane/model', self._on_lane, 10)
        self.create_timer(0.05, self._safety_tick)   # 20 Hz watchdog
        self._last_msg_time = None
        self.get_logger().info('BEV lane keeper ready — right lane, P control')

    def _on_lane(self, msg: LaneModel):
        self._last_msg_time = self.get_clock().now()
        cmd = Twist()

        if msg.confidence >= MIN_CONF and math.isfinite(msg.error_px):
            # error_px > 0 → target right of image centre → steer right (−ω)
            cmd.linear.x  = SPEED
            cmd.angular.z = -KP * msg.error_px
        else:
            # Low confidence: crawl straight and wait for a detection
            cmd.linear.x  = SPEED * 0.4
            cmd.angular.z = 0.0

        self.pub.publish(cmd)
        self.get_logger().info(
            f'conf={msg.confidence:.2f}  err={msg.error_px:+.1f}px  '
            f'v={cmd.linear.x:.2f}  ω={cmd.angular.z:+.3f}',
            throttle_duration_sec=0.4,
        )

    def _safety_tick(self):
        if self._last_msg_time is None:
            return
        age = (self.get_clock().now() - self._last_msg_time).nanoseconds * 1e-9
        if age > TIMEOUT:
            self.pub.publish(Twist())   # stop — lost lane signal


def main():
    rclpy.init()
    node = BevLaneKeeper()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.pub.publish(Twist())      # stop on exit
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
