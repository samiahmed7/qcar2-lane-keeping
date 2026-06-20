#!/usr/bin/env python3
"""Grab one frame from a sensor_msgs/Image topic and save it as a PNG, then exit.

    python3 scripts/grab_image.py <topic> <out.png> [timeout_s]
"""
import sys
import time

import cv2
import rclpy
from sensor_msgs.msg import Image

from qcar2_autonomy.image_msg_utils import image_msg_to_bgr


def main():
    topic = sys.argv[1]
    out = sys.argv[2]
    timeout = float(sys.argv[3]) if len(sys.argv) > 3 else 30.0

    rclpy.init()
    node = rclpy.create_node("grab_image")
    got = {}

    def cb(msg):
        try:
            got["img"] = image_msg_to_bgr(msg)
        except Exception as exc:  # noqa: BLE001
            print(f"decode failed: {exc}")

    node.create_subscription(Image, topic, cb, 10)
    t0 = time.time()
    while rclpy.ok() and "img" not in got and time.time() - t0 < timeout:
        rclpy.spin_once(node, timeout_sec=0.2)

    if "img" in got:
        cv2.imwrite(out, got["img"])
        print(f"saved {out}  shape={got['img'].shape}")
    else:
        print(f"NO FRAME on {topic} within {timeout:.0f}s")
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == "__main__":
    main()
