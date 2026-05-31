#!/usr/bin/env python3
"""Standard-message lane perception node for a physical QCar2.

This node is intentionally separate from ``bev_lane_detector_node.py``. The
existing node belongs to the current custom-message simulation stack, while this
one publishes a compact ``Float32MultiArray`` so it can be deployed without
building custom message packages on the robot.
"""
import math

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge, CvBridgeError
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray


STATUS_NONE = 0.0
STATUS_SINGLE = 1.0
STATUS_BOTH = 2.0
MISSING_X = -1.0

# IPM source/destination polygon, expressed as image-normalized (x, y) ratios.
# Hoisted to module level so external tools (scripts/debug_perception.py) can
# import and stay in sync with the running node without manual mirroring.
# Ordering invariant: near-left, far-left, far-right, near-right.
IPM_SRC_RATIOS = [
    (0.000, 0.656),   # near left  (left edge, near-row)
    (0.156, 0.521),   # far left   (just below horizon)
    (0.844, 0.521),   # far right  (just below horizon)
    (1.000, 0.656),   # near right (right edge, near-row)
]
IPM_DST_RATIOS = [
    (0.22, 1.00),
    (0.22, 0.00),
    (0.78, 0.00),
    (0.78, 1.00),
]


class AdvancedPerceptionNode(Node):
    """Extract lane x-positions using IPM, histogram peaks, and sliding windows."""

    def __init__(self):
        super().__init__('perception_node')

        # Required interface parameters. Keep these names stable; downstream
        # validation logic assumes the same image coordinate frame.
        self.declare_parameter('image_width', 640)
        self.declare_parameter('image_height', 480)

        # Topic parameters make the node usable on both the real QCar2
        # (/camera/image_raw) and the existing simulator (/qcar2/front_camera/image)
        # without changing code.
        self.declare_parameter('image_topic', '/camera/image_raw')
        self.declare_parameter('raw_lane_topic', '/perception/raw_lane_data')
        self.declare_parameter('debug_image_topic', '/perception/debug_image')

        # debug_mode publishes an annotated camera frame on
        # /perception/debug_image (sensor_msgs/Image). Open it in rqt_image_view
        # to physically confirm the IPM source polygon (green trapezoid) lines
        # up with the lane markings before trusting the warp downstream.
        self.declare_parameter('debug_mode', True)

        # Vision tuning parameters. Defaults are conservative for white lane
        # markings; final values should be calibrated with the actual camera,
        # exposure, floor material, and lane tape/paint.
        self.declare_parameter('binary_threshold', 180)
        self.declare_parameter('use_canny', False)
        self.declare_parameter('canny_low_threshold', 50)
        self.declare_parameter('canny_high_threshold', 150)
        self.declare_parameter('sliding_window_count', 9)
        self.declare_parameter('sliding_window_margin', 100)
        self.declare_parameter('sliding_window_minpix', 50)
        self.declare_parameter('min_lane_pixels_for_fit', 120)
        self.declare_parameter('enable_undistort', False)

        self.image_width = int(self.get_parameter('image_width').value)
        self.image_height = int(self.get_parameter('image_height').value)
        self.binary_threshold = int(self.get_parameter('binary_threshold').value)
        self.use_canny = bool(self.get_parameter('use_canny').value)
        self.canny_low_threshold = int(
            self.get_parameter('canny_low_threshold').value
        )
        self.canny_high_threshold = int(
            self.get_parameter('canny_high_threshold').value
        )
        self.nwindows = max(
            1,
            int(self.get_parameter('sliding_window_count').value),
        )
        self.margin = max(
            1,
            int(self.get_parameter('sliding_window_margin').value),
        )
        self.minpix = max(
            1,
            int(self.get_parameter('sliding_window_minpix').value),
        )
        self.min_lane_pixels_for_fit = max(
            3,
            int(self.get_parameter('min_lane_pixels_for_fit').value),
        )
        self.enable_undistort = bool(self.get_parameter('enable_undistort').value)
        self.debug_mode = bool(self.get_parameter('debug_mode').value)

        self.bridge = CvBridge()

        # Placeholder camera calibration matrix for a 640x480 camera.
        #
        # The pinhole model maps 3D camera-frame rays to pixels:
        #
        #     [u]   [fx  0 cx] [X/Z]
        #     [v] = [ 0 fy cy] [Y/Z]
        #     [1]   [ 0  0  1] [ 1 ]
        #
        # fx/fy are focal lengths measured in pixels, and cx/cy are the optical
        # center. These values are placeholders only. Replace them with output
        # from a real checkerboard calibration before enabling undistortion.
        self.camera_matrix = np.array(
            [
                [600.0, 0.0, self.image_width / 2.0],
                [0.0, 600.0, self.image_height / 2.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

        # Placeholder distortion coefficients [k1, k2, p1, p2, k3].
        # k terms model radial barrel/pincushion distortion; p terms model
        # tangential distortion from lens/sensor misalignment.
        self.distortion_coefficients = np.zeros((5, 1), dtype=np.float32)

        self.src_points, self.dst_points = self._build_ipm_points()
        self.perspective_matrix = cv2.getPerspectiveTransform(
            self.src_points,
            self.dst_points,
        )

        self.raw_lane_pub = self.create_publisher(
            Float32MultiArray,
            self.get_parameter('raw_lane_topic').value,
            10,
        )
        self.debug_image_pub = self.create_publisher(
            Image,
            self.get_parameter('debug_image_topic').value,
            10,
        )
        self.create_subscription(
            Image,
            self.get_parameter('image_topic').value,
            self._on_image,
            10,
        )

        self.get_logger().info(
            'Standard perception node ready: '
            f'image={self.image_width}x{self.image_height}, '
            f'image_topic={self.get_parameter("image_topic").value}, '
            f'raw_lane_topic={self.get_parameter("raw_lane_topic").value}, '
            f'debug_mode={self.debug_mode}, '
            f'debug_image_topic={self.get_parameter("debug_image_topic").value}'
        )

    def _build_ipm_points(self):
        """Return placeholder source/destination polygons for bird's-eye view.

        The inverse perspective mapping (IPM) is a projective transform. It
        assumes the observed lane markings lie approximately on a flat ground
        plane. Four image points on that plane are enough to compute a 3x3
        homography H such that:
            [x_bev, y_bev, w]^T = H [x_img, y_img, 1]^T
        After division by w, perspective convergence is removed and parallel
        lane lines become nearly vertical in the warped image.
        """
        width = float(self.image_width)
        height = float(self.image_height)

        # Wide trapezoid covering the full track corridor (both lanes).
        #
        # Bug fix - "Tight Crop Trap": the previous polygon was scoped to the
        # right lane only. When the DWA planner swerved into the left lane to
        # avoid an obstacle, the middle dashed line and the far-left solid line
        # fell outside the IPM source region, perception went blind (status=0),
        # and the controller lost its target mid-maneuver.
        #
        # New geometry stretches the bottom edge to the image corners and the
        # top edge wide enough to catch the far-left solid line just below the
        # horizon. Coordinates below are the absolute pixel values from the spec
        # for a 640x480 frame, parameterized by image dimensions for portability.
        # Polygon definitions live at module scope (IPM_SRC_RATIOS / IPM_DST_RATIOS)
        # so external tools can import the same values. See the constants block
        # at the top of the file for the rationale behind the ratios chosen.
        src = np.float32([[rx * width, ry * height] for rx, ry in IPM_SRC_RATIOS])
        dst = np.float32([[rx * width, ry * height] for rx, ry in IPM_DST_RATIOS])
        return src, dst

    def _on_image(self, msg: Image):
        """Convert the incoming ROS image and publish lane x estimates."""
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError as exc:
            self.get_logger().warn(
                f'cv_bridge conversion failed: {exc}',
                throttle_duration_sec=2.0,
            )
            self._publish_lane_data(MISSING_X, MISSING_X, STATUS_NONE)
            return

        try:
            left_x, right_x, status = self._process_frame(bgr)
        except (cv2.error, ValueError, FloatingPointError) as exc:
            self.get_logger().warn(
                f'lane extraction failed: {exc}',
                throttle_duration_sec=2.0,
            )
            left_x, right_x, status = MISSING_X, MISSING_X, STATUS_NONE

        self._publish_lane_data(left_x, right_x, status)
        if self.debug_mode:
            self._publish_debug_image(bgr, left_x, right_x, status, msg.header)

    def _process_frame(self, bgr):
        """Run IPM, thresholding, histogram search, and polynomial fitting."""
        resized = self._resize_to_processing_shape(bgr)

        if self.enable_undistort:
            # With calibrated intrinsics, undistortion makes the homography more
            # consistent because straight lane markings remain straight before
            # IPM. Disabled by default because placeholder calibration values
            # should not be applied to real images.
            resized = cv2.undistort(
                resized,
                self.camera_matrix,
                self.distortion_coefficients,
            )

        binary = self._binary_lane_mask(resized)

        # Bird's-eye warp. The output shape is fixed to the processing image
        # size so all published x coordinates live in the same pixel frame.
        warped = cv2.warpPerspective(
            binary,
            self.perspective_matrix,
            (self.image_width, self.image_height),
            flags=cv2.INTER_LINEAR,
        )

        left_base, right_base = self._histogram_lane_bases(warped)
        left_indices, right_indices, nonzero_x, nonzero_y = self._sliding_windows(
            warped,
            left_base,
            right_base,
        )

        left_fit_x = self._fit_lane_x_at_bottom(
            nonzero_x,
            nonzero_y,
            left_indices,
        )
        right_fit_x = self._fit_lane_x_at_bottom(
            nonzero_x,
            nonzero_y,
            right_indices,
        )

        left_found = self._is_valid_x(left_fit_x)
        right_found = self._is_valid_x(right_fit_x)

        if left_found and right_found:
            status = STATUS_BOTH
        elif left_found or right_found:
            status = STATUS_SINGLE
        else:
            status = STATUS_NONE

        return (
            float(left_fit_x) if left_found else MISSING_X,
            float(right_fit_x) if right_found else MISSING_X,
            status,
        )

    def _resize_to_processing_shape(self, bgr):
        height, width = bgr.shape[:2]
        if width == self.image_width and height == self.image_height:
            return bgr
        return cv2.resize(
            bgr,
            (self.image_width, self.image_height),
            interpolation=cv2.INTER_AREA,
        )

    def _binary_lane_mask(self, bgr):
        """Create a binary image where lane pixels are 255 and background is 0."""
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        if self.use_canny:
            # Canny returns edges, not filled lane bands. It is useful when lane
            # paint is dim but edge contrast is still strong.
            edges = cv2.Canny(
                gray,
                self.canny_low_threshold,
                self.canny_high_threshold,
            )
            return edges

        _, binary = cv2.threshold(
            gray,
            self.binary_threshold,
            255,
            cv2.THRESH_BINARY,
        )

        # Small morphology removes isolated hot pixels and joins short gaps in
        # painted/taped lane markings before the histogram projection.
        kernel = np.ones((3, 3), dtype=np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        return binary

    def _histogram_lane_bases(self, warped):
        """Find starting x positions from the lower-half pixel histogram."""
        midpoint = warped.shape[1] // 2

        # Sum bright pixels vertically in the lower half. Peaks indicate where
        # lane-line pixels are densest near the vehicle; those x positions seed
        # the first sliding windows.
        histogram = np.sum(warped[warped.shape[0] // 2:, :] > 0, axis=0)

        left_peak = int(np.argmax(histogram[:midpoint]))
        right_peak = int(np.argmax(histogram[midpoint:]) + midpoint)

        if histogram[left_peak] == 0:
            left_peak = None
        if histogram[right_peak] == 0:
            right_peak = None

        return left_peak, right_peak

    def _sliding_windows(self, warped, left_base, right_base):
        """Collect lane-pixel indices by recentering windows from bottom to top."""
        nonzero_y, nonzero_x = warped.nonzero()
        window_height = int(warped.shape[0] / self.nwindows)

        left_current = left_base
        right_current = right_base
        left_lane_inds = []
        right_lane_inds = []

        for window in range(self.nwindows):
            # Window y extents move upward because image y=0 is the top row.
            win_y_low = warped.shape[0] - (window + 1) * window_height
            win_y_high = warped.shape[0] - window * window_height

            if left_current is not None:
                left_inds = self._indices_inside_window(
                    nonzero_x,
                    nonzero_y,
                    left_current,
                    win_y_low,
                    win_y_high,
                )
                left_lane_inds.append(left_inds)
                if left_inds.size > self.minpix:
                    # Mean x recenters the next window on the observed lane
                    # pixels, allowing the stack of windows to follow curvature.
                    left_current = int(np.mean(nonzero_x[left_inds]))

            if right_current is not None:
                right_inds = self._indices_inside_window(
                    nonzero_x,
                    nonzero_y,
                    right_current,
                    win_y_low,
                    win_y_high,
                )
                right_lane_inds.append(right_inds)
                if right_inds.size > self.minpix:
                    right_current = int(np.mean(nonzero_x[right_inds]))

        left_lane_inds = self._concat_indices(left_lane_inds)
        right_lane_inds = self._concat_indices(right_lane_inds)
        return left_lane_inds, right_lane_inds, nonzero_x, nonzero_y

    def _indices_inside_window(self, nonzero_x, nonzero_y, center_x, y_low, y_high):
        x_low = center_x - self.margin
        x_high = center_x + self.margin
        return (
            (nonzero_y >= y_low)
            & (nonzero_y < y_high)
            & (nonzero_x >= x_low)
            & (nonzero_x < x_high)
        ).nonzero()[0]

    @staticmethod
    def _concat_indices(indices):
        if not indices:
            return np.array([], dtype=np.int64)
        return np.concatenate(indices).astype(np.int64)

    def _fit_lane_x_at_bottom(self, nonzero_x, nonzero_y, lane_indices):
        """Fit x(y) = ay^2 + by + c and evaluate it at the bottom row.

        ``np.polyfit`` receives y as the independent variable because the lane
        line is nearly vertical in bird's-eye view. Evaluating the polynomial at
        y=image_height gives the lane-line x coordinate closest to the car.
        """
        if lane_indices.size < self.min_lane_pixels_for_fit:
            return math.nan

        lane_x = nonzero_x[lane_indices]
        lane_y = nonzero_y[lane_indices]

        fit = np.polyfit(lane_y.astype(np.float32), lane_x.astype(np.float32), 2)
        y_eval = float(self.image_height)
        x_eval = fit[0] * y_eval * y_eval + fit[1] * y_eval + fit[2]
        return float(x_eval)

    def _is_valid_x(self, x_value):
        return (
            math.isfinite(x_value)
            and 0.0 <= x_value <= float(self.image_width - 1)
        )

    def _publish_lane_data(self, left_fit_x, right_fit_x, status):
        msg = Float32MultiArray()
        msg.data = [float(left_fit_x), float(right_fit_x), float(status)]
        self.raw_lane_pub.publish(msg)

    def _publish_debug_image(self, bgr, left_x, right_x, status, header):
        """Annotate the raw frame so the user can verify IPM in rqt_image_view.

        Draws the IPM source polygon (green) so misalignment with the lane
        markings is visually obvious, the camera principal point (red vertical
        line) for reference, and the detected lane x positions (blue/yellow
        dots) so the user can see what the histogram + sliding-window stage
        produced *before* the warp.
        """
        try:
            annotated = self._resize_to_processing_shape(bgr).copy()

            # Source polygon overlay (Rule from spec: this is the snapshot the
            # IPM is taking; if it doesn't hug the lane edges, the warp is
            # garbage and everything downstream is too).
            pts = self.src_points.astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated, [pts], True, (0, 255, 0), 2)

            # Principal point reference line.
            cx = self.image_width // 2
            cv2.line(annotated, (cx, 0), (cx, self.image_height), (0, 0, 255), 1)

            # Detected lane positions, projected back to the bottom row.
            y_mark = self.image_height - 10
            if self._is_valid_x(left_x):
                cv2.circle(annotated, (int(left_x), y_mark), 8, (255, 200, 0), -1)
            if self._is_valid_x(right_x):
                cv2.circle(annotated, (int(right_x), y_mark), 8, (0, 200, 255), -1)

            # HUD text: status + numeric x's, so the rqt viewer doesn't also
            # need a topic echo to interpret the frame.
            status_label = {
                STATUS_BOTH: 'BOTH',
                STATUS_SINGLE: 'SINGLE',
                STATUS_NONE: 'NONE',
            }.get(status, '?')
            hud = f'status={status_label}  L={left_x:.1f}  R={right_x:.1f}'
            cv2.putText(annotated, hud, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)

            out = self.bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
            out.header = header
            self.debug_image_pub.publish(out)
        except (cv2.error, CvBridgeError) as exc:
            self.get_logger().warn(
                f'debug image publish failed: {exc}',
                throttle_duration_sec=2.0,
            )


def main(args=None):
    rclpy.init(args=args)
    node = AdvancedPerceptionNode()
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
