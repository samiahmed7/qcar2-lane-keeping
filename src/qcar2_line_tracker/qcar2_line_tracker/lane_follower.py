#!/usr/bin/env python3
"""
Lane keeping for QCar2 with dynamic lane width + temporal smoothing.

The frame is split vertically at the centre. We find:
  - Left edge: centroid of white pixels in the LEFT half of the ROI
  - Right edge: centroid of white pixels in the RIGHT half of the ROI

Detection is restricted to a thin LOOK-AHEAD BAND near the top of the ROI —
this is "far field" where lines are more spatially stable, dashes blend
together, and small lateral errors translate to small angular errors,
which dampens oscillation.

When both edges are visible we record (right - left) as the current lane
width in pixels. When only one edge is visible (e.g. centre-dash gap) we
use half the last known lane width to estimate the missing edge, so the
target doesn't jump.

The midpoint is then low-pass filtered (EMA) so brief detection glitches
don't cause steering jerks.

Publishes:
  /model/qcar2/cmd_vel
  /qcar2/debug/image
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import numpy as np


class LaneFollower(Node):
    def __init__(self):
        super().__init__('lane_follower')

        self.declare_parameter('speed', 0.15)
        self.declare_parameter('crawl_speed', 0.05)
        self.declare_parameter('kp', 0.0022)
        self.declare_parameter('kd', 0.0030)
        self.declare_parameter('steering_limit', 0.5)
        self.declare_parameter('roi_start', 0.45)
        self.declare_parameter('look_ahead_top', 0.0)
        self.declare_parameter('look_ahead_bottom', 0.35)
        self.declare_parameter('white_v_low', 180)
        self.declare_parameter('min_pixels_per_side', 50)
        self.declare_parameter('default_lane_width_px', 240)
        self.declare_parameter('midpoint_smoothing', 0.70)

        self.speed = self.get_parameter('speed').value
        self.crawl_speed = self.get_parameter('crawl_speed').value
        self.kp = self.get_parameter('kp').value
        self.kd = self.get_parameter('kd').value
        self.steering_limit = self.get_parameter('steering_limit').value
        self.roi_start = self.get_parameter('roi_start').value
        self.look_ahead_top = self.get_parameter('look_ahead_top').value
        self.look_ahead_bottom = self.get_parameter('look_ahead_bottom').value
        self.white_v_low = self.get_parameter('white_v_low').value
        self.min_pixels_per_side = self.get_parameter('min_pixels_per_side').value
        self.lane_width = float(self.get_parameter('default_lane_width_px').value)
        self.alpha = float(self.get_parameter('midpoint_smoothing').value)

        self.prev_error = 0.0
        self.smoothed_midpoint = None
        self.frame_count = 0
        self.bridge = CvBridge()

        self.sub = self.create_subscription(
            Image, '/qcar2/front_camera/image', self.image_callback, 10
        )
        self.cmd_pub = self.create_publisher(Twist, '/model/qcar2/cmd_vel', 10)
        self.debug_pub = self.create_publisher(Image, '/qcar2/debug/image', 10)

        self.get_logger().info(
            f'Lane keeper running. kp={self.kp} kd={self.kd} '
            f'alpha={self.alpha} default_width={self.lane_width}'
        )

    def image_callback(self, msg: Image):
        self.frame_count += 1
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        h, w = frame.shape[:2]
        cx_img = w // 2

        roi_y0 = int(h * self.roi_start)
        roi = frame[roi_y0:, :]
        roi_h = roi.shape[0]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask_full = cv2.inRange(
            hsv,
            np.array([0, 0, self.white_v_low], dtype=np.uint8),
            np.array([180, 60, 255], dtype=np.uint8),
        )
        mask_full = cv2.morphologyEx(
            mask_full, cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        )

        # Restrict to a thin LOOK-AHEAD BAND near the top of the ROI
        band_y0 = int(roi_h * self.look_ahead_top)
        band_y1 = int(roi_h * self.look_ahead_bottom)
        mask = np.zeros_like(mask_full)
        mask[band_y0:band_y1, :] = mask_full[band_y0:band_y1, :]
        self._band_y_in_roi = (band_y0, band_y1)

        left_half = mask[:, :cx_img]
        right_half = mask[:, cx_img:]

        left_x = self._centroid_x(left_half)
        right_x = self._centroid_x(right_half)
        if right_x is not None:
            right_x += cx_img

        left_count = int(np.sum(left_half > 0))
        right_count = int(np.sum(right_half > 0))

        left_ok = left_count >= self.min_pixels_per_side
        right_ok = right_count >= self.min_pixels_per_side

        raw_midpoint = None
        status = 'NO_LINES'

        if left_ok and right_ok:
            raw_midpoint = (left_x + right_x) // 2
            measured = float(right_x - left_x)
            if 150 < measured < w:
                self.lane_width = 0.85 * self.lane_width + 0.15 * measured
            status = 'BOTH'
        elif left_ok:
            raw_midpoint = int(left_x + self.lane_width / 2.0)
            status = 'LEFT_ONLY'
        elif right_ok:
            raw_midpoint = int(right_x - self.lane_width / 2.0)
            status = 'RIGHT_ONLY'

        twist = Twist()

        if raw_midpoint is not None:
            raw_midpoint = int(np.clip(raw_midpoint, 0, w - 1))

            if self.smoothed_midpoint is None:
                self.smoothed_midpoint = float(raw_midpoint)
            else:
                self.smoothed_midpoint = (
                    self.alpha * self.smoothed_midpoint
                    + (1.0 - self.alpha) * raw_midpoint
                )
            mid = int(self.smoothed_midpoint)

            error = float(mid - cx_img)
            d_error = error - self.prev_error
            steering = self.kp * error + self.kd * d_error
            steering = float(np.clip(steering, -self.steering_limit, self.steering_limit))
            self.prev_error = error

            twist.linear.x = self.speed
            twist.angular.z = -steering
        else:
            mid = None
            twist.linear.x = self.crawl_speed
            twist.angular.z = 0.0
            self.smoothed_midpoint = None

        self.cmd_pub.publish(twist)

        if self.frame_count % 30 == 1:
            self.get_logger().info(
                f'#{self.frame_count} {status} '
                f'L={left_x}({left_count}) R={right_x}({right_count}) '
                f'mid={mid} width={self.lane_width:.0f} steer={twist.angular.z:+.3f}'
            )

        self._publish_debug(frame, mask, roi_y0,
                            left_x, right_x, raw_midpoint, mid,
                            left_count, right_count, status, twist)

    @staticmethod
    def _centroid_x(half_mask):
        M = cv2.moments(half_mask)
        if M['m00'] < 1:
            return None
        return int(M['m10'] / M['m00'])

    def _publish_debug(self, frame, mask, roi_y0, left_x, right_x,
                       raw_mid, smooth_mid, left_count, right_count, status, twist):
        h, w = frame.shape[:2]
        debug = frame.copy()

        cv2.rectangle(debug, (0, roi_y0), (w - 1, h - 1), (0, 255, 255), 1)
        band_y0, band_y1 = self._band_y_in_roi
        cv2.rectangle(debug, (0, roi_y0 + band_y0), (w - 1, roi_y0 + band_y1),
                      (255, 0, 255), 2)
        cv2.line(debug, (w // 2, 0), (w // 2, h - 1), (200, 200, 200), 1)

        overlay = np.zeros_like(debug)
        overlay[roi_y0:, :, 1] = mask
        debug = cv2.addWeighted(debug, 0.7, overlay, 0.5, 0)

        if left_x is not None:
            cv2.line(debug, (left_x, roi_y0), (left_x, h - 1), (255, 80, 0), 3)
        if right_x is not None:
            cv2.line(debug, (right_x, roi_y0), (right_x, h - 1), (0, 80, 255), 3)
        if raw_mid is not None:
            cv2.circle(debug, (raw_mid, h - 50), 6, (0, 180, 0), 2)
        if smooth_mid is not None:
            cv2.circle(debug, (smooth_mid, h - 25), 10, (0, 255, 0), -1)
            cv2.line(debug, (w // 2, h - 25), (smooth_mid, h - 25), (0, 255, 0), 2)

        cv2.putText(debug, f'status: {status}  width:{int(self.lane_width)}',
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(debug, f'L={left_count} R={right_count}', (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        cv2.putText(debug,
                    f'spd: {twist.linear.x:.2f}  steer: {twist.angular.z:+.3f}',
                    (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        self.debug_pub.publish(self.bridge.cv2_to_imgmsg(debug, encoding='bgr8'))


def main(args=None):
    rclpy.init(args=args)
    node = LaneFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
