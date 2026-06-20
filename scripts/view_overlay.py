#!/usr/bin/env python3
"""Small, movable live viewer for a ROS Image topic (lighter than rqt).

Usage:
    python3 scripts/view_overlay.py                       # /qcar2/lane/debug_image
    python3 scripts/view_overlay.py /qcar2/front_camera/image
Press 'q' in the window to quit.
"""
import sys
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

TOPIC = sys.argv[1] if len(sys.argv) > 1 else '/qcar2/lane/debug_image'
WIN = 'QCar2 view'


def main():
    rclpy.init()
    node = Node('view_overlay')
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)   # resizable
    cv2.resizeWindow(WIN, 480, 360)           # small
    cv2.moveWindow(WIN, 60, 60)               # movable (drag the title bar too)
    latest = {}

    def cb(m):
        img = np.frombuffer(bytes(m.data), np.uint8).reshape(m.height, m.width, 3)
        latest['img'] = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if m.encoding == 'rgb8' else img

    node.create_subscription(Image, TOPIC, cb, 10)
    print(f'viewing {TOPIC} — press q to quit')
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.05)
            if 'img' in latest:
                cv2.imshow(WIN, latest['img'])
            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                break
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
