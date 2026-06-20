#!/usr/bin/env python3
"""LiDAR obstacle perception for the MPC stack.

This node is intentionally outside the controller. It converts front LiDAR
returns into a compact world-frame obstacle observation:

    /mpc/obstacle Float32MultiArray
        [x, y, radius, front_distance, left_clear, right_clear]

When no obstacle is present, x/y are NaN and front_distance is inf.
"""
import math

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32MultiArray

from qcar2_autonomy.mpc_drive_node import yaw_from_quaternion


class MpcLidarObstacleNode(Node):
    def __init__(self):
        super().__init__("mpc_lidar_obstacle_node")

        self.declare_parameter("scan_topic", "/qcar2/lidar/scan")
        self.declare_parameter("odom_topic", "/model/qcar2/odometry")
        self.declare_parameter("obstacle_topic", "/mpc/obstacle")
        self.declare_parameter("front_angle_deg", 25.0)
        self.declare_parameter("detect_range_m", 2.5)
        self.declare_parameter("min_points", 2)
        self.declare_parameter("min_radius_m", 0.15)
        self.declare_parameter("face_depth_tolerance_m", 0.30)
        self.declare_parameter("forward_angle_rad", 0.0)
        self.declare_parameter("side_angle_min_deg", 30.0)
        self.declare_parameter("side_angle_max_deg", 150.0)

        self.front_angle = math.radians(float(self.get_parameter("front_angle_deg").value))
        self.detect_range = float(self.get_parameter("detect_range_m").value)
        self.min_points = max(1, int(self.get_parameter("min_points").value))
        self.min_radius = float(self.get_parameter("min_radius_m").value)
        self.face_tol = float(self.get_parameter("face_depth_tolerance_m").value)
        self.forward_angle_rad = float(self.get_parameter("forward_angle_rad").value)
        self.side_lo = math.radians(float(self.get_parameter("side_angle_min_deg").value))
        self.side_hi = math.radians(float(self.get_parameter("side_angle_max_deg").value))

        self.pose = None  # x, y, yaw

        self.pub = self.create_publisher(
            Float32MultiArray,
            str(self.get_parameter("obstacle_topic").value),
            10,
        )
        self.create_subscription(
            Odometry,
            str(self.get_parameter("odom_topic").value),
            self._on_odom,
            10,
        )
        self.create_subscription(
            LaserScan,
            str(self.get_parameter("scan_topic").value),
            self._on_scan,
            10,
        )

        self.get_logger().info(
            "MPC LiDAR obstacle node ready: "
            f"scan={self.get_parameter('scan_topic').value} -> "
            f"{self.get_parameter('obstacle_topic').value}"
        )

    def _on_odom(self, msg: Odometry):
        p = msg.pose.pose.position
        self.pose = (
            float(p.x),
            float(p.y),
            yaw_from_quaternion(msg.pose.pose.orientation),
        )

    def _on_scan(self, msg: LaserScan):
        if len(msg.ranges) == 0 or not math.isfinite(msg.angle_increment):
            self._publish_empty(math.inf, math.inf, math.inf)
            return

        err, ranges, valid = self._scan_arrays(msg)
        left_clear = self._sector_min(ranges, valid, err, self.side_lo, self.side_hi)
        right_clear = self._sector_min(ranges, valid, err, -self.side_hi, -self.side_lo)

        front = valid & (np.abs(err) < self.front_angle) & (ranges <= self.detect_range)
        if np.count_nonzero(front) < self.min_points or self.pose is None:
            front_min = float(ranges[front].min()) if np.any(front) else math.inf
            self._publish_empty(front_min, left_clear, right_clear)
            return

        front_ranges = ranges[front]
        front_err = err[front]
        front_min = float(front_ranges.min())
        face = front_ranges <= front_min + self.face_tol
        if np.count_nonzero(face) < self.min_points:
            self._publish_empty(front_min, left_clear, right_clear)
            return

        rf = front_ranges[face]
        ef = front_err[face]
        local_x = rf * np.cos(ef)
        local_y = rf * np.sin(ef)
        center_x = float(local_x.mean())
        center_y = float(local_y.mean())
        width = float(np.hypot(local_x.max() - local_x.min(), local_y.max() - local_y.min()))
        radius = max(self.min_radius, 0.5 * width)

        x, y, yaw = self.pose
        obs_x = x + center_x * math.cos(yaw) - center_y * math.sin(yaw)
        obs_y = y + center_x * math.sin(yaw) + center_y * math.cos(yaw)

        msg_out = Float32MultiArray()
        msg_out.data = [
            float(obs_x),
            float(obs_y),
            float(radius),
            float(front_min),
            float(left_clear),
            float(right_clear),
        ]
        self.pub.publish(msg_out)

    def _scan_arrays(self, scan: LaserScan):
        idx = np.arange(len(scan.ranges), dtype=np.float64)
        angles = scan.angle_min + idx * scan.angle_increment
        err = np.arctan2(
            np.sin(angles - self.forward_angle_rad),
            np.cos(angles - self.forward_angle_rad),
        )
        ranges = np.asarray(scan.ranges, dtype=np.float32)
        valid = (
            np.isfinite(ranges)
            & (ranges > 0.0)
            & (ranges >= float(scan.range_min))
            & (ranges <= float(scan.range_max))
        )
        return err, ranges, valid

    @staticmethod
    def _sector_min(ranges, valid, err, lo, hi):
        sel = valid & (err >= lo) & (err < hi)
        return float(ranges[sel].min()) if np.any(sel) else math.inf

    def _publish_empty(self, front_min, left_clear, right_clear):
        msg = Float32MultiArray()
        msg.data = [
            math.nan,
            math.nan,
            0.0,
            float(front_min),
            float(left_clear),
            float(right_clear),
        ]
        self.pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = MpcLidarObstacleNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
