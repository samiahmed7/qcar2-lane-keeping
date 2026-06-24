#!/usr/bin/env python3
"""Smooth keyboard teleop for QCar2 recording.

Controls (non-blocking, values HOLD until you change them):
  W / S   — speed up / slow down  (0.05 m/s steps)
  A / D   — steer left / right    (0.05 rad/s steps)
  SPACE   — stop (zero speed, keep steering)
  R       — reset steering to straight
  Q       — quit and stop car

The current speed+steering are published at 20 Hz so the car holds
a curve continuously without you hammering a key.
"""
import sys, tty, termios, threading, math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

SPEED_STEP  = 0.05   # m/s per keypress
STEER_STEP  = 0.05   # rad/s per keypress
MAX_SPEED   = 0.60   # m/s
MAX_STEER   = 0.60   # rad/s
RATE_HZ     = 20

def getch():
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)

class Teleop(Node):
    def __init__(self):
        super().__init__('drive_teleop')
        self.pub = self.create_publisher(Twist, '/model/qcar2/cmd_vel', 10)
        self.speed  = 0.0
        self.steer  = 0.0
        self.running = True
        self.create_timer(1.0 / RATE_HZ, self._publish)

    def _publish(self):
        t = Twist()
        t.linear.x  = float(self.speed)
        t.angular.z = float(self.steer)
        self.pub.publish(t)

    def stop(self):
        self.speed = 0.0
        t = Twist()
        self.pub.publish(t)

def main():
    rclpy.init()
    node = Teleop()
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    print("\n=== QCar2 drive teleop (for waypoint recording) ===")
    print("  W/S  speed up / slow down")
    print("  A/D  steer left / right")
    print("  SPC  stop (keep steer)")
    print("  R    straighten wheels")
    print("  Q    quit\n")
    print(f"  speed={node.speed:+.2f}  steer={node.steer:+.2f}", end='\r')

    try:
        while node.running:
            ch = getch().lower()
            if ch == 'w':
                node.speed = min(MAX_SPEED, node.speed + SPEED_STEP)
            elif ch == 's':
                node.speed = max(-MAX_SPEED, node.speed - SPEED_STEP)
            elif ch == 'a':
                node.steer = min(MAX_STEER, node.steer + STEER_STEP)
            elif ch == 'd':
                node.steer = max(-MAX_STEER, node.steer - STEER_STEP)
            elif ch == ' ':
                node.speed = 0.0
            elif ch == 'r':
                node.steer = 0.0
            elif ch in ('q', '\x03'):
                node.running = False
                break
            print(f"  speed={node.speed:+.2f}  steer={node.steer:+.2f}  (Q=quit, SPC=stop, R=straight)  ", end='\r')
    finally:
        node.stop()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
