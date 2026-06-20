#!/usr/bin/env python3
"""Optional stochastic sensor wrapper for simulation-to-real testing.

The simulator publishes near-perfect odometry and LiDAR. This node republishes
those topics with configurable Gaussian noise, bias, and scan dropout so the MPC
stack can be tested against sensor uncertainty before moving to the real QCar2.
"""
import copy
import math

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import LaserScan

from qcar2_autonomy.mpc_drive_node import yaw_from_quaternion


def quaternion_from_yaw(yaw):
    z = math.sin(0.5 * yaw)
    w = math.cos(0.5 * yaw)
    return z, w


class MpcSensorNoiseNode(Node):
    def __init__(self):
        super().__init__("mpc_sensor_noise_node")

        self.declare_parameter("raw_odom_topic", "/model/qcar2/odometry")
        self.declare_parameter("raw_scan_topic", "/qcar2/lidar/scan")
        self.declare_parameter("noisy_odom_topic", "/mpc/noisy_odometry")
        self.declare_parameter("noisy_scan_topic", "/mpc/noisy_scan")
        self.declare_parameter("seed", 7)
        self.declare_parameter("odom_position_stddev_m", 0.015)
        self.declare_parameter("odom_yaw_stddev_rad", 0.010)
        self.declare_parameter("odom_yaw_bias_rad", 0.005)
        self.declare_parameter("odom_velocity_stddev_mps", 0.020)
        self.declare_parameter("odom_angular_velocity_stddev_rps", 0.020)
        self.declare_parameter("scan_range_stddev_m", 0.015)
        self.declare_parameter("scan_range_bias_m", 0.005)
        self.declare_parameter("scan_dropout_probability", 0.01)

        seed = int(self.get_parameter("seed").value)
        self.rng = np.random.default_rng(seed)
        self.odom_position_stddev = float(
            self.get_parameter("odom_position_stddev_m").value
        )
        self.odom_yaw_stddev = float(self.get_parameter("odom_yaw_stddev_rad").value)
        self.odom_yaw_bias = float(self.get_parameter("odom_yaw_bias_rad").value)
        self.odom_velocity_stddev = float(
            self.get_parameter("odom_velocity_stddev_mps").value
        )
        self.odom_angular_velocity_stddev = float(
            self.get_parameter("odom_angular_velocity_stddev_rps").value
        )
        self.scan_range_stddev = float(self.get_parameter("scan_range_stddev_m").value)
        self.scan_range_bias = float(self.get_parameter("scan_range_bias_m").value)
        self.scan_dropout_probability = min(
            1.0,
            max(0.0, float(self.get_parameter("scan_dropout_probability").value)),
        )

        self.odom_pub = self.create_publisher(
            Odometry,
            str(self.get_parameter("noisy_odom_topic").value),
            10,
        )
        self.scan_pub = self.create_publisher(
            LaserScan,
            str(self.get_parameter("noisy_scan_topic").value),
            10,
        )
        self.create_subscription(
            Odometry,
            str(self.get_parameter("raw_odom_topic").value),
            self._on_odom,
            10,
        )
        self.create_subscription(
            LaserScan,
            str(self.get_parameter("raw_scan_topic").value),
            self._on_scan,
            10,
        )

        self.get_logger().info(
            "MPC sensor noise node ready: "
            f"odom_std={self.odom_position_stddev:.3f}m, "
            f"yaw_std={self.odom_yaw_stddev:.3f}rad, "
            f"scan_std={self.scan_range_stddev:.3f}m, "
            f"dropout={self.scan_dropout_probability:.3f}"
        )

    def _on_odom(self, msg: Odometry):
        out = copy.deepcopy(msg)
        pose = out.pose.pose
        twist = out.twist.twist

        pose.position.x += float(self.rng.normal(0.0, self.odom_position_stddev))
        pose.position.y += float(self.rng.normal(0.0, self.odom_position_stddev))
        yaw = yaw_from_quaternion(pose.orientation)
        yaw += self.odom_yaw_bias + float(self.rng.normal(0.0, self.odom_yaw_stddev))
        pose.orientation.x = 0.0
        pose.orientation.y = 0.0
        pose.orientation.z, pose.orientation.w = quaternion_from_yaw(yaw)
        out.pose.covariance[0] = self.odom_position_stddev ** 2
        out.pose.covariance[7] = self.odom_position_stddev ** 2
        out.pose.covariance[35] = self.odom_yaw_stddev ** 2

        twist.linear.x += float(self.rng.normal(0.0, self.odom_velocity_stddev))
        twist.angular.z += float(
            self.rng.normal(0.0, self.odom_angular_velocity_stddev)
        )
        out.twist.covariance[0] = self.odom_velocity_stddev ** 2
        out.twist.covariance[35] = self.odom_angular_velocity_stddev ** 2
        self.odom_pub.publish(out)

    def _on_scan(self, msg: LaserScan):
        out = copy.deepcopy(msg)
        ranges = np.asarray(out.ranges, dtype=np.float32)
        finite = np.isfinite(ranges)
        if np.any(finite):
            ranges[finite] += self.scan_range_bias
            ranges[finite] += self.rng.normal(
                0.0,
                self.scan_range_stddev,
                size=int(np.count_nonzero(finite)),
            ).astype(np.float32)
            ranges[finite] = np.clip(
                ranges[finite],
                float(out.range_min),
                float(out.range_max),
            )
            if self.scan_dropout_probability > 0.0:
                drop = self.rng.random(ranges.shape) < self.scan_dropout_probability
                ranges[finite & drop] = math.inf
        out.ranges = ranges.tolist()
        self.scan_pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = MpcSensorNoiseNode()
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
