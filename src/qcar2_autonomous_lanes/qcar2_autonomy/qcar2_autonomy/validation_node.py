#!/usr/bin/env python3
"""Validate raw lane geometry and publish a safe target x coordinate.

This node protects the controller from perception edge cases without inventing
custom ROS message types. It consumes the compact lane array emitted by
``perception_node.py``:

    [left_fit_x, right_fit_x, status]

The key safety guard is the "histogram split trap" override. In a sharp curve,
one physical lane line can straddle the histogram midpoint and be reported as
two separate lines. If the observed two-line width is impossibly narrow, this
node rejects the pair, treats their midpoint as a single detected line, and
uses the last trusted EMA lane width to infer the lane center.
"""
import math

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, Float32MultiArray, String


STATUS_NONE = 0.0
STATUS_SINGLE = 1.0
STATUS_BOTH = 2.0

LANE_KEEP = 'LANE_KEEP'
LANE_CHANGE = 'LANE_CHANGE'

INITIAL_DYNAMIC_WIDTH_PX = 300.0
FALSE_POSITIVE_WIDTH_RATIO = 0.60


class ValidationNode(Node):
    """EMA lane-width tracker with false-positive rejection and curve fallback."""

    def __init__(self):
        super().__init__('validation_node')

        self.declare_parameter('alpha', 0.05)
        self.declare_parameter('camera_center_x', 320.0)
        self.declare_parameter('raw_lane_topic', '/perception/raw_lane_data')
        self.declare_parameter('current_state_topic', '/system/current_state')
        self.declare_parameter('target_topic', '/planning/validated_target_x')

        self.alpha = self._clamp(
            float(self.get_parameter('alpha').value),
            0.0,
            1.0,
        )
        self.camera_center_x = float(self.get_parameter('camera_center_x').value)

        # State machine memory. LANE_CHANGE pauses the EMA so temporary lane
        # geometry during avoidance maneuvers cannot corrupt the trusted width.
        self.current_state = LANE_KEEP
        self.dynamic_width = INITIAL_DYNAMIC_WIDTH_PX
        self.last_target_x = math.nan

        self.target_pub = self.create_publisher(
            Float32,
            self.get_parameter('target_topic').value,
            10,
        )
        self.create_subscription(
            Float32MultiArray,
            self.get_parameter('raw_lane_topic').value,
            self._on_raw_lane_data,
            10,
        )
        self.create_subscription(
            String,
            self.get_parameter('current_state_topic').value,
            self._on_system_state,
            10,
        )

        self.get_logger().info(
            'Validation node ready: '
            f'alpha={self.alpha}, camera_center_x={self.camera_center_x}, '
            f'initial_dynamic_width={self.dynamic_width}, '
            f'false_positive_width_ratio={FALSE_POSITIVE_WIDTH_RATIO}'
        )

    def _on_system_state(self, msg: String):
        """Track the current high-level behavior state."""
        state = msg.data.strip().upper()
        if state:
            self.current_state = state

    def _on_raw_lane_data(self, msg: Float32MultiArray):
        """Compute and publish target_x from [left_x, right_x, status]."""
        target_x = self._target_from_raw_lane_data(msg.data)
        if target_x is None:
            return

        out = Float32()
        out.data = float(target_x)
        self.target_pub.publish(out)
        self.last_target_x = float(target_x)

    def _target_from_raw_lane_data(self, data):
        if len(data) < 3:
            self.get_logger().warn(
                'raw lane array too short; expected [left_x, right_x, status]',
                throttle_duration_sec=2.0,
            )
            return None

        left_fit_x = float(data[0])
        right_fit_x = float(data[1])
        status = float(data[2])

        if not math.isfinite(status):
            return None

        if self._status_is_both(status):
            return self._target_from_two_line_report(left_fit_x, right_fit_x)

        if self._status_is_single(status):
            detected_x = self._single_detected_x(left_fit_x, right_fit_x)
            if detected_x is None:
                return None
            return self._target_from_single_line(detected_x)

        # No visible lines means no trustworthy geometric observation.
        # Downstream control should stop on target timeout rather than follow a
        # synthetic stale coordinate.
        return None

    def _target_from_two_line_report(self, left_fit_x, right_fit_x):
        """Handle a reported two-line observation, including false positives."""
        if not (
            self._valid_detection(left_fit_x)
            and self._valid_detection(right_fit_x)
        ):
            return None

        # Z_t is the raw pixel distance between the two reported lane fits at
        # the bottom of the image. Under normal conditions this is a lane-width
        # measurement. In the histogram split trap, both fits come from the same
        # physical line, making Z_t much smaller than the trusted EMA width.
        raw_width = abs(right_fit_x - left_fit_x)
        if raw_width <= 0.0 or not math.isfinite(raw_width):
            return None

        if self._is_histogram_split_trap(raw_width):
            false_peak_x = 0.5 * (left_fit_x + right_fit_x)
            threshold = FALSE_POSITIVE_WIDTH_RATIO * self.dynamic_width
            self.get_logger().warn(
                'False Positive Override: reported two-line width '
                f'{raw_width:.2f}px is below {threshold:.2f}px '
                f'({FALSE_POSITIVE_WIDTH_RATIO:.0%} of dynamic_width='
                f'{self.dynamic_width:.2f}px). Downgrading to single-line '
                f'fallback at false_peak_x={false_peak_x:.2f}px.',
                throttle_duration_sec=1.0,
            )

            # Do not update the EMA here. The last trusted dynamic_width is the
            # safety memory that lets single-line fallback remain useful.
            return self._target_from_single_line(false_peak_x)

        # During obstacle avoidance and return-to-lane, the camera may see
        # crossed/skewed lane geometry. Only update the EMA in true LANE_KEEP;
        # every other state preserves the trusted width used for fallback.
        if self.current_state == LANE_KEEP:
            self.dynamic_width = (
                self.alpha * raw_width
                + (1.0 - self.alpha) * self.dynamic_width
            )

        # With a plausible two-line observation, the lane center is the midpoint.
        return 0.5 * (left_fit_x + right_fit_x)

    def _target_from_single_line(self, detected_x):
        """Infer lane center from one visible boundary and trusted lane width."""
        if not self._valid_detection(detected_x):
            return None

        # If the visible line is right of the optical center, treat it as the
        # right boundary and shift left by half the trusted lane width. If it is
        # left of the optical center, treat it as the left boundary and shift
        # right. This is also used by the false-positive override.
        half_width = 0.5 * self.dynamic_width
        if detected_x > self.camera_center_x:
            return detected_x - half_width
        return detected_x + half_width

    def _is_histogram_split_trap(self, raw_width):
        minimum_plausible_width = FALSE_POSITIVE_WIDTH_RATIO * self.dynamic_width
        return raw_width < minimum_plausible_width

    @staticmethod
    def _status_is_both(status):
        return abs(status - STATUS_BOTH) < 0.1

    @staticmethod
    def _status_is_single(status):
        return abs(status - STATUS_SINGLE) < 0.1

    @staticmethod
    def _valid_detection(value):
        return math.isfinite(value) and value >= 0.0

    def _single_detected_x(self, left_fit_x, right_fit_x):
        left_valid = self._valid_detection(left_fit_x)
        right_valid = self._valid_detection(right_fit_x)

        if left_valid and not right_valid:
            return left_fit_x
        if right_valid and not left_valid:
            return right_fit_x

        # Defensive behavior: if status says single but both fields are valid,
        # use the point farther from the optical center. That is usually the
        # actual boundary rather than a near-center artifact.
        if left_valid and right_valid:
            left_distance = abs(left_fit_x - self.camera_center_x)
            right_distance = abs(right_fit_x - self.camera_center_x)
            if left_distance > right_distance:
                return left_fit_x
            return right_fit_x

        return None

    @staticmethod
    def _clamp(value, low, high):
        return max(low, min(high, value))


def main(args=None):
    rclpy.init(args=args)
    node = ValidationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
