#!/usr/bin/env python3
"""Log QCar2 avoidance run: state/target/cmd/odom to CSV + periodic debug frames."""
import math
import sys
import time

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import Image
from std_msgs.msg import Float32, String


def to_bgr(msg):
    img = np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape(msg.height, msg.width, 3)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if msg.encoding == 'rgb8' else img


class Mon(Node):
    def __init__(self, out_dir, duration):
        super().__init__('mon')
        self.out_dir = out_dir
        self.t_end = time.monotonic() + duration
        self.state = '?'
        self.target = float('nan')
        self.v = self.w = 0.0
        self.ox = self.oy = self.yaw = float('nan')
        self.sim = 0.0
        self.frame = None
        self.last_img_save = 0.0
        self.img_idx = 0
        self.last_state = None
        self.rows = []
        self.create_subscription(String, '/system/current_state', lambda m: setattr(self, 'state', m.data), 10)
        self.create_subscription(Float32, '/planning/validated_target_x', lambda m: setattr(self, 'target', m.data), 10)
        self.create_subscription(Twist, '/model/qcar2/cmd_vel', self._cmd, 10)
        self.create_subscription(Odometry, '/model/qcar2/odometry', self._odom, 10)
        self.create_subscription(Clock, '/clock', lambda m: setattr(self, 'sim', m.clock.sec + m.clock.nanosec * 1e-9), 10)
        self.create_subscription(Image, '/rfdetr_lane/debug_image', lambda m: setattr(self, 'frame', to_bgr(m)), 10)
        self.create_timer(0.2, self._tick)

    def _cmd(self, m):
        self.v, self.w = m.linear.x, m.angular.z

    def _odom(self, m):
        p = m.pose.pose.position
        q = m.pose.pose.orientation
        self.ox, self.oy = p.x, p.y
        self.yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))

    def _tick(self):
        now = time.monotonic()
        self.rows.append((f'{self.sim:.2f}', self.state, f'{self.target:.1f}',
                          f'{self.v:.3f}', f'{self.w:.3f}', f'{self.ox:.3f}',
                          f'{self.oy:.3f}', f'{self.yaw:.3f}'))
        # save a frame on state change AND every ~4 wall-sec
        if self.frame is not None and (self.state != self.last_state or now - self.last_img_save >= 4.0):
            self.last_img_save = now
            self.last_state = self.state
            path = f'{self.out_dir}/f{self.img_idx:02d}_{self.state}_x{self.ox:.1f}_y{self.oy:.2f}.png'
            cv2.imwrite(path, self.frame)
            self.img_idx += 1
        if now >= self.t_end:
            with open(f'{self.out_dir}/trajectory.csv', 'w') as fh:
                fh.write('sim,state,target_x,v,w,odom_x,odom_y,yaw\n')
                for r in self.rows:
                    fh.write(','.join(r) + '\n')
            self.get_logger().info(f'wrote CSV ({len(self.rows)} rows), {self.img_idx} frames')
            rclpy.shutdown()


def main():
    rclpy.init()
    Mon(sys.argv[1], float(sys.argv[2]))
    try:
        rclpy.spin(rclpy.node.Node) if False else rclpy.spin(_node())
    except Exception:
        pass


_NODE = None


def _node():
    return _NODE


if __name__ == '__main__':
    rclpy.init()
    n = Mon(sys.argv[1], float(sys.argv[2]))
    try:
        rclpy.spin(n)
    except Exception:
        pass
