#!/usr/bin/env python3

import numpy as np

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from std_msgs.msg import Bool


class DepthEmergencyNode(Node):

    def __init__(self):
        super().__init__("depth_emergency_node")

        # Distances in meters. NOT scaled up alongside the +15% speed pass
        # in path_mpc_node.py (unlike lidar_overtake_node.py's distances) --
        # the ROI (y 35%-80% of frame height) reaches down far enough that
        # its near edge reads ~0.5-0.6m on open floor, always, just from the
        # camera's mount height/tilt (confirmed against a live frame: fully
        # open track, no obstacle, still reads front-of-bumper floor at
        # min=0.52/p10=0.57-0.59m). stop_distance=0.45 sits just under that
        # floor-geometry floor by luck; raising it to the v^2-scaled 0.60
        # crosses above it and false-trips on open track every time. Real
        # fix is narrowing the ROI to exclude near-field floor rows (or
        # ground-plane subtraction) so it measures actual obstacles instead
        # of the floor -- left at the original values until that's done.
        self.avoid_distance = 0.80
        self.stop_distance = 0.45

        # Percentage of valid ROI pixels that must be close
        self.min_obstacle_ratio = 0.03
        self.min_emergency_ratio = 0.01

        # Require enough valid RealSense depth pixels
        self.min_valid_ratio = 0.10

        # Temporal filtering
        self.obstacle_hits = 0
        self.emergency_hits = 0
        self.obstacle_clear_hits = 0
        self.emergency_clear_hits = 0

        self.required_hits = 2
        self.required_clear_hits = 3

        self.last_obstacle = False
        self.last_emergency = False

        self.obstacle_pub = self.create_publisher(
            Bool,
            "/depth_obstacle_ahead",
            10,
        )

        self.emergency_pub = self.create_publisher(
            Bool,
            "/depth_emergency_stop",
            10,
        )

        self.depth_sub = self.create_subscription(
            Image,
            "/camera/depth_image",
            self.depth_callback,
            10,
        )

        self.timer = self.create_timer(
            0.1,
            self.publish_state,
        )

        self.get_logger().info("QCar2 RealSense depth emergency node started")

    def publish_state(self):
        obs_msg = Bool()
        obs_msg.data = bool(self.last_obstacle)
        self.obstacle_pub.publish(obs_msg)

        estop_msg = Bool()
        estop_msg.data = bool(self.last_emergency)
        self.emergency_pub.publish(estop_msg)

    def depth_callback(self, msg):
        try:
            if msg.encoding not in ["mono16", "16UC1"]:
                self.get_logger().warn(
                    f"Unexpected depth encoding: {msg.encoding}",
                    throttle_duration_sec=1.0,
                )

            if msg.height == 0 or msg.width == 0 or len(msg.data) < msg.height * msg.width * 2:
                # Malformed/empty frame (occasional camera driver hiccup under load).
                # Skip this frame and keep the previous obstacle/estop state rather
                # than resetting safety flags to "clear" on a bad read.
                self.get_logger().warn(
                    "Skipping malformed depth frame (empty or size-mismatched)",
                    throttle_duration_sec=1.0,
                )
                return

            depth = np.frombuffer(
                msg.data,
                dtype=np.uint16,
            ).reshape(
                msg.height,
                msg.width,
            ).astype(np.float32)

            # RealSense mono16 / 16UC1 depth is usually millimeters
            depth = depth / 1000.0

            h, w = depth.shape

            # QCar2 forward driving ROI.
            # This ignores the very top and very bottom of the image.
            x1 = int(w * 0.30)
            x2 = int(w * 0.70)

            y1 = int(h * 0.35)
            y2 = int(h * 0.80)

            roi = depth[y1:y2, x1:x2]

            roi_total_pixels = roi.size

            valid_mask = (
                np.isfinite(roi)
                & (roi > 0.10)
                & (roi < 4.0)
            )

            valid = roi[valid_mask]
            valid_ratio = valid.size / roi_total_pixels

            if valid_ratio < self.min_valid_ratio:
                self.obstacle_clear_hits += 1
                self.emergency_clear_hits += 1

                if self.obstacle_clear_hits >= self.required_clear_hits:
                    self.last_obstacle = False
                    self.obstacle_hits = 0

                if self.emergency_clear_hits >= self.required_clear_hits:
                    self.last_emergency = False
                    self.emergency_hits = 0

                self.get_logger().warn(
                    f"Low valid depth in ROI: "
                    f"valid_ratio={valid_ratio:.3f}, "
                    f"valid_px={valid.size}, "
                    f"roi_px={roi_total_pixels}",
                    throttle_duration_sec=0.5,
                )
                return

            close_obstacle = valid < self.avoid_distance
            close_emergency = valid < self.stop_distance

            obstacle_ratio = float(np.sum(close_obstacle)) / valid.size
            emergency_ratio = float(np.sum(close_emergency)) / valid.size

            # Percentile is more stable than minimum.
            # Minimum can be caused by one noisy pixel.
            p10_depth = float(np.percentile(valid, 10))
            median_depth = float(np.median(valid))
            min_depth = float(np.min(valid))

            obstacle_now = (
                obstacle_ratio >= self.min_obstacle_ratio
                or p10_depth < self.avoid_distance
            )

            emergency_now = (
                emergency_ratio >= self.min_emergency_ratio
                or p10_depth < self.stop_distance
            )

            if obstacle_now:
                self.obstacle_hits += 1
            else:
                self.obstacle_hits = 0

            if emergency_now:
                self.emergency_hits += 1
            else:
                self.emergency_hits = 0

            # Obstacle and emergency latches clear independently -- a
            # persistent mild obstacle (e.g. a nearby wall just inside
            # avoid_distance) must not keep the stricter emergency estop
            # latched forever once the acute close-range condition clears.
            if not obstacle_now:
                self.obstacle_clear_hits += 1
            else:
                self.obstacle_clear_hits = 0

            if not emergency_now:
                self.emergency_clear_hits += 1
            else:
                self.emergency_clear_hits = 0

            if self.obstacle_hits >= self.required_hits:
                self.last_obstacle = True

            if self.emergency_hits >= self.required_hits:
                self.last_emergency = True
                self.last_obstacle = True

            if self.obstacle_clear_hits >= self.required_clear_hits:
                self.last_obstacle = False

            if self.emergency_clear_hits >= self.required_clear_hits:
                self.last_emergency = False

            self.get_logger().info(
                f"ROI x={x1}:{x2}, y={y1}:{y2}, "
                f"valid_ratio={valid_ratio:.2f}, "
                f"min={min_depth:.2f}m, "
                f"p10={p10_depth:.2f}m, "
                f"median={median_depth:.2f}m, "
                f"obs_ratio={obstacle_ratio:.3f}, "
                f"estop_ratio={emergency_ratio:.3f}, "
                f"obs={self.last_obstacle}, "
                f"estop={self.last_emergency}",
                throttle_duration_sec=0.5,
            )

        except Exception as e:
            self.last_obstacle = False
            self.last_emergency = False
            self.get_logger().error(f"Depth error: {e}")


def main():
    rclpy.init()
    node = DepthEmergencyNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()