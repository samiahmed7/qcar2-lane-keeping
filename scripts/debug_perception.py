#!/usr/bin/env python3
"""Live OpenCV visualizer for the perception_node pipeline.

Opens four windows that mirror what perception_node.py is computing internally:

    1. raw         - original sim camera frame, with the IPM source polygon
                     drawn in green so you can see what the homography sees
    2. binary      - thresholded grayscale (the lane-mask fed into the warp)
    3. warped      - bird's-eye view of the binary mask
    4. windows     - warped image with the 9 sliding windows + the fitted
                     polynomial drawn in green/red. The colored vertical bars
                     mark the histogram peaks the search started from.

Defaults mirror perception_node.py exactly so what you see here matches what
the running pipeline is computing. Press 'q' or ESC in any window to quit.
"""
import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image

# Import the polygon ratios directly from the running node's module so this
# debug viewer can never drift out of sync with what perception_node is using.
# Requires the workspace to be sourced before launching this script (the run
# scripts and ros2 setup already do this).
from qcar2_autonomy.perception_node import IPM_SRC_RATIOS, IPM_DST_RATIOS


IMAGE_W, IMAGE_H = 640, 480
BINARY_THRESHOLD = 180
NWINDOWS = 9
MARGIN = 100
MINPIX = 50
MIN_LANE_PIXELS_FOR_FIT = 120


def build_ipm(w, h):
    src = np.float32([[rx * w, ry * h] for rx, ry in IPM_SRC_RATIOS])
    dst = np.float32([[rx * w, ry * h] for rx, ry in IPM_DST_RATIOS])
    return src, dst, cv2.getPerspectiveTransform(src, dst)


def binary_mask(bgr):
    gray = cv2.GaussianBlur(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    _, b = cv2.threshold(gray, BINARY_THRESHOLD, 255, cv2.THRESH_BINARY)
    k = np.ones((3, 3), np.uint8)
    b = cv2.morphologyEx(b, cv2.MORPH_OPEN, k)
    b = cv2.morphologyEx(b, cv2.MORPH_CLOSE, k)
    return b


def sliding_window_overlay(warped):
    """Replicate perception_node's sliding search and draw it on a color canvas."""
    h, w = warped.shape
    canvas = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)

    histogram = np.sum(warped[h // 2:, :] > 0, axis=0)
    mid = w // 2
    left_base = int(np.argmax(histogram[:mid]))
    right_base = int(np.argmax(histogram[mid:]) + mid)
    if histogram[left_base] == 0:
        left_base = None
    if histogram[right_base] == 0:
        right_base = None

    # Histogram peak markers (cyan = left, magenta = right).
    if left_base is not None:
        cv2.line(canvas, (left_base, h - 5), (left_base, h - 30), (255, 255, 0), 2)
    if right_base is not None:
        cv2.line(canvas, (right_base, h - 5), (right_base, h - 30), (255, 0, 255), 2)

    nonzero_y, nonzero_x = warped.nonzero()
    win_h = int(h / NWINDOWS)
    left_inds, right_inds = [], []
    lc, rc = left_base, right_base

    for window in range(NWINDOWS):
        y_low = h - (window + 1) * win_h
        y_high = h - window * win_h
        if lc is not None:
            x_low, x_high = lc - MARGIN, lc + MARGIN
            cv2.rectangle(canvas, (x_low, y_low), (x_high, y_high), (0, 255, 0), 1)
            good = (
                (nonzero_y >= y_low) & (nonzero_y < y_high)
                & (nonzero_x >= x_low) & (nonzero_x < x_high)
            ).nonzero()[0]
            left_inds.append(good)
            if good.size > MINPIX:
                lc = int(np.mean(nonzero_x[good]))
        if rc is not None:
            x_low, x_high = rc - MARGIN, rc + MARGIN
            cv2.rectangle(canvas, (x_low, y_low), (x_high, y_high), (0, 255, 0), 1)
            good = (
                (nonzero_y >= y_low) & (nonzero_y < y_high)
                & (nonzero_x >= x_low) & (nonzero_x < x_high)
            ).nonzero()[0]
            right_inds.append(good)
            if good.size > MINPIX:
                rc = int(np.mean(nonzero_x[good]))

    li = np.concatenate(left_inds) if left_inds else np.array([], dtype=np.int64)
    ri = np.concatenate(right_inds) if right_inds else np.array([], dtype=np.int64)

    left_x_at_bottom = right_x_at_bottom = None
    if li.size >= MIN_LANE_PIXELS_FOR_FIT:
        lf = np.polyfit(nonzero_y[li].astype(np.float32), nonzero_x[li].astype(np.float32), 2)
        ys = np.linspace(0, h - 1, h)
        xs = lf[0] * ys * ys + lf[1] * ys + lf[2]
        pts = np.array([np.column_stack([xs, ys])], dtype=np.int32)
        cv2.polylines(canvas, pts, False, (0, 255, 255), 2)
        left_x_at_bottom = float(lf[0] * h * h + lf[1] * h + lf[2])
    if ri.size >= MIN_LANE_PIXELS_FOR_FIT:
        rf = np.polyfit(nonzero_y[ri].astype(np.float32), nonzero_x[ri].astype(np.float32), 2)
        ys = np.linspace(0, h - 1, h)
        xs = rf[0] * ys * ys + rf[1] * ys + rf[2]
        pts = np.array([np.column_stack([xs, ys])], dtype=np.int32)
        cv2.polylines(canvas, pts, False, (0, 0, 255), 2)
        right_x_at_bottom = float(rf[0] * h * h + rf[1] * h + rf[2])

    status_text = f'L={left_x_at_bottom}  R={right_x_at_bottom}'
    cv2.putText(canvas, status_text, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)
    return canvas


class DebugViewer(Node):

    def __init__(self):
        super().__init__('perception_debug_viewer')
        self.bridge = CvBridge()
        self.src, self.dst, self.M = build_ipm(IMAGE_W, IMAGE_H)
        self.create_subscription(
            Image, '/qcar2/front_camera/image', self._on_image, 10
        )
        self.get_logger().info(
            'debug viewer subscribed to /qcar2/front_camera/image '
            "(press 'q' or ESC in any window to quit)"
        )

    def _on_image(self, msg: Image):
        bgr = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        if bgr.shape[1] != IMAGE_W or bgr.shape[0] != IMAGE_H:
            bgr = cv2.resize(bgr, (IMAGE_W, IMAGE_H))

        raw = bgr.copy()
        cv2.polylines(raw, [self.src.astype(np.int32).reshape(-1, 1, 2)],
                      True, (0, 255, 0), 2)
        cv2.line(raw, (IMAGE_W // 2, 0), (IMAGE_W // 2, IMAGE_H), (0, 0, 255), 1)

        binary = binary_mask(bgr)
        warped = cv2.warpPerspective(binary, self.M, (IMAGE_W, IMAGE_H))
        windows = sliding_window_overlay(warped)

        cv2.imshow('1. raw + IPM polygon', raw)
        cv2.imshow('2. binary', binary)
        cv2.imshow('3. warped', warped)
        cv2.imshow('4. sliding windows + fit', windows)
        if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
            rclpy.shutdown()


def main():
    rclpy.init()
    n = DebugViewer()
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        n.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
