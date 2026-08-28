#!/usr/bin/env python3
"""Seed cartographer's pure_localization trajectory with a known pose.

Why this exists
---------------
qcar2_2d_localization.lua now rejects weak global relocalisation matches
(global_localization_min_score raised to 0.90) because false matches were
teleporting the car mid-drive between two look-alike stretches of track.
That fixes driving but removes the mechanism the e-stop -> relaunch recovery
used to lean on. This restores recovery a better way: the car cannot move
while it is stopped, so the pose path_mpc last trusted is still true at
relaunch. Seed cartographer with it instead of making it search blind.

How
---
cartographer auto-starts trajectory 1 on launch with no initial pose. There
is no way to re-seed a running trajectory, so finish it and start a fresh one
positioned relative to the frozen loaded map (trajectory 0).

Deliberately additive: if anything here fails, the auto-started trajectory is
left alone and the stack behaves exactly as before. Seeding is an
improvement, never a new way for startup to break.
"""

import sys
import math
import os

import rclpy
from rclpy.node import Node

from cartographer_ros_msgs.srv import FinishTrajectory, StartTrajectory

POSE_FILE = "/home/nvidia/ros2_ws_sami/last_known_pose.seed"
CONFIG_DIR = "/home/nvidia/ros2_ws_sami/install/qcar2_nodes/share/qcar2_nodes/config"
CONFIG_BASENAME = "qcar2_2d_localization.lua"

# cartographer auto-starts the first trajectory after the frozen loaded one.
AUTO_STARTED_TRAJECTORY_ID = 1
FROZEN_MAP_TRAJECTORY_ID = 0


def read_pose(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path) as fh:
            parts = fh.read().split()
        if len(parts) != 3:
            return None
        return tuple(float(p) for p in parts)
    except (OSError, ValueError):
        return None


class Seeder(Node):
    def __init__(self):
        super().__init__("seed_cartographer")

    def call(self, client, req, what):
        if not client.wait_for_service(timeout_sec=10.0):
            self.get_logger().error(f"{what}: service unavailable")
            return None
        fut = client.call_async(req)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=20.0)
        if not fut.done():
            self.get_logger().error(f"{what}: timed out")
            return None
        return fut.result()


def main():
    pose = read_pose(POSE_FILE)
    if pose is None:
        print(f"no usable pose in {POSE_FILE} — leaving auto-started "
              f"trajectory alone (cartographer will localise on its own)")
        return 0

    x, y, yaw = pose
    print(f"seeding cartographer at x={x:.3f} y={y:.3f} yaw={math.degrees(yaw):.1f}deg")

    rclpy.init()
    node = Seeder()
    try:
        finish = node.create_client(FinishTrajectory, "/finish_trajectory")
        req = FinishTrajectory.Request()
        req.trajectory_id = AUTO_STARTED_TRAJECTORY_ID
        res = node.call(finish, req, "finish_trajectory")
        if res is None:
            print("SEED FAILED at finish_trajectory — auto-started trajectory "
                  "still running, stack is usable but unseeded")
            return 1
        print(f"  finished trajectory {AUTO_STARTED_TRAJECTORY_ID}: "
              f"{res.status.message}")

        start = node.create_client(StartTrajectory, "/start_trajectory")
        sreq = StartTrajectory.Request()
        sreq.configuration_directory = CONFIG_DIR
        sreq.configuration_basename = CONFIG_BASENAME
        sreq.use_initial_pose = True
        sreq.relative_to_trajectory_id = FROZEN_MAP_TRAJECTORY_ID
        sreq.initial_pose.position.x = x
        sreq.initial_pose.position.y = y
        sreq.initial_pose.position.z = 0.0
        sreq.initial_pose.orientation.z = math.sin(yaw / 2.0)
        sreq.initial_pose.orientation.w = math.cos(yaw / 2.0)

        sres = node.call(start, sreq, "start_trajectory")
        if sres is None:
            print("SEED FAILED at start_trajectory — NO trajectory is running. "
                  "Relaunch localization before driving.")
            return 2
        print(f"  started seeded trajectory {sres.trajectory_id}: "
              f"{sres.status.message}")
        return 0
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    sys.exit(main())
