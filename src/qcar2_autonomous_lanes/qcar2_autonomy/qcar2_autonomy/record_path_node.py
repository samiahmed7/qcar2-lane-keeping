#!/usr/bin/env python3
"""Record the driven centreline as MPC waypoints.

Run the normal RF-DETR lane-keeper so the car drives the track, run this
alongside, and it samples /model/qcar2/odometry every `spacing` metres into a
waypoint file. Stops on loop-closure (returns near the start after leaving it)
or Ctrl-C, then saves an .npy of shape (2, N) = [x; y].
"""
import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node


class RecordPathNode(Node):
    def __init__(self):
        super().__init__("record_path_node")
        self.declare_parameter("odom_topic", "/model/qcar2/odometry")
        self.declare_parameter("out_path", "/home/hammad/rosbot_ws/track_waypoints.npy")
        self.declare_parameter("spacing", 0.20)
        self.declare_parameter("loop_close_dist", 0.40)
        self.declare_parameter("loop_min_points", 30)

        self.out_path = str(self.get_parameter("out_path").value)
        self.spacing = float(self.get_parameter("spacing").value)
        self.loop_close_dist = float(self.get_parameter("loop_close_dist").value)
        self.loop_min_points = int(self.get_parameter("loop_min_points").value)

        self.pts = []
        self.start = None
        self.left_start = False

        self.create_subscription(
            Odometry, str(self.get_parameter("odom_topic").value), self._on_odom, 10
        )
        self.get_logger().info(
            f"Recording waypoints every {self.spacing} m -> {self.out_path}. "
            "Drive a lap; Ctrl-C to stop and save."
        )

    def _on_odom(self, msg: Odometry):
        x = float(msg.pose.pose.position.x)
        y = float(msg.pose.pose.position.y)
        if self.start is None:
            self.start = (x, y)
            self.pts.append((x, y))
            return
        lx, ly = self.pts[-1]
        if np.hypot(x - lx, y - ly) < self.spacing:
            return
        self.pts.append((x, y))
        d_start = np.hypot(x - self.start[0], y - self.start[1])
        if d_start > 2.0 * self.loop_close_dist:
            self.left_start = True
        if (self.left_start and len(self.pts) >= self.loop_min_points
                and d_start < self.loop_close_dist):
            self.get_logger().info("Loop closed -> saving.")
            self.save()
            rclpy.shutdown()
        if len(self.pts) % 10 == 0:
            self.get_logger().info(f"{len(self.pts)} waypoints (d_start={d_start:.2f} m)")

    def save(self):
        if len(self.pts) < 3:
            self.get_logger().warn("Too few waypoints, not saving.")
            return
        arr = np.array(self.pts, dtype=np.float64).T  # (2, N)
        np.save(self.out_path, arr)
        self.get_logger().info(f"Saved {arr.shape[1]} waypoints to {self.out_path}")


def main(args=None):
    rclpy.init(args=args)
    node = RecordPathNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.save()
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
