#!/usr/bin/env python3
"""
Click-to-calibrate tool for the Inverse Perspective Mapping (IPM) transform.

Workflow
--------
1.  Start the simulation (Terminal 1).
2.  Pause Gazebo (spacebar in the GUI) with the car on a STRAIGHT section so
    the lane markings are roughly straight in the camera view.
3.  Run this tool:
        ros2 run qcar2_line_tracker ipm_calibrate
4.  A live preview window opens. Press 's' to capture a frame.
5.  Click 4 points outlining a known rectangle on the road, in this order:
        1) bottom-left   (closer to camera, left side)
        2) bottom-right  (closer, right side)
        3) top-right     (further away, right side)
        4) top-left      (further away, left side)

    On our track the natural rectangle is bounded by the two SOLID white lane
    lines (centre dashes are inside, you click on the outer lane lines):
        - real width  = 0.80 m  (lane line to lane line — from ±0.40)
        - real length = whatever distance ahead you cover (1.0–2.0 m typical)

6.  After clicking the 4th point the warped bird's-eye preview opens. If it
    looks like a clean rectangle with parallel lane lines, press 's' to save.
    Otherwise press 'r' to redo the 4 clicks.

Parameters
----------
  real_width_m   (default 0.80)  real-world width of the clicked rectangle
  real_length_m  (default 1.50)  real-world length of the clicked rectangle
  output_yaml    (default: <pkg_share>/config/ipm.yaml)

Tip
---
If you're not sure what 'real_length_m' to use, just pick a value that makes
the warped preview look proportional (the lane should appear as a long
straight strip, not wildly stretched).
"""
import os
import sys
import time
import cv2
import numpy as np
import yaml
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


def default_yaml_path() -> str:
    try:
        from ament_index_python.packages import get_package_share_directory
        return os.path.join(
            get_package_share_directory('qcar2_line_tracker'),
            'config', 'ipm.yaml'
        )
    except Exception:
        return '/tmp/ipm.yaml'


def _find_source_config_path() -> str:
    """
    Walk up from this file's location to find the workspace src/ layout.
    Returns the path where the source-tree copy of ipm.yaml should live,
    or None if we can't figure out where the source tree is.
    """
    here = os.path.realpath(__file__)
    # We expect a path like .../qcar2_line_tracker/qcar2_line_tracker/ipm_calibrate.py
    # in either an installed or source layout. Walk up looking for a
    # directory whose name is 'qcar2_line_tracker' that contains a 'config' dir.
    cur = os.path.dirname(here)
    for _ in range(6):
        candidate = os.path.join(cur, 'config', 'ipm.yaml')
        # Heuristic: source-tree path contains 'src/qcar2_line_tracker'
        if 'src/qcar2_line_tracker' in cur and os.path.isdir(os.path.dirname(candidate)):
            return candidate
        # Heuristic: look one level up for a package.xml typical of src
        pkg_xml = os.path.join(cur, 'package.xml')
        if os.path.exists(pkg_xml) and 'src/qcar2_line_tracker' in cur:
            return candidate
        cur = os.path.dirname(cur)
    # Fall back: guess based on $HOME
    workspace = os.environ.get('COLCON_PREFIX_PATH', '')
    if workspace:
        ws_root = os.path.dirname(workspace.split(':')[0])
        candidate = os.path.join(
            ws_root, 'src', 'qcar2_line_tracker', 'config', 'ipm.yaml'
        )
        if os.path.isdir(os.path.dirname(candidate)):
            return candidate
    return None


class FrameGrabber(Node):
    def __init__(self):
        super().__init__('ipm_calibrate')
        self.declare_parameter('real_width_m', 0.80)
        self.declare_parameter('real_length_m', 1.50)
        self.declare_parameter('output_yaml', default_yaml_path())
        self.declare_parameter('image_topic', '/qcar2/front_camera/image')

        self.real_width_m = float(self.get_parameter('real_width_m').value)
        self.real_length_m = float(self.get_parameter('real_length_m').value)
        self.output_yaml = self.get_parameter('output_yaml').value
        topic = self.get_parameter('image_topic').value

        self.bridge = CvBridge()
        self.latest_frame = None
        self.create_subscription(Image, topic, self._on_image, 10)

    def _on_image(self, msg):
        self.latest_frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')


def save_calibration(node: FrameGrabber, src_points, frame_shape):
    h_img, w_img = frame_shape[:2]
    warped_w, warped_h = 640, 480

    # Destination rectangle: centred horizontally, fills the warped image vertically.
    # Width = 60% of warped width represents the clicked real_width_m.
    margin = 0.20 * warped_w
    dst = np.float32([
        [margin,           warped_h - 1],     # BL
        [warped_w - margin, warped_h - 1],    # BR
        [warped_w - margin, 0],               # TR
        [margin,           0],                # TL
    ])
    src = np.float32(src_points)

    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)

    dst_width_px = warped_w - 2 * margin
    m_per_px_x = node.real_width_m / dst_width_px
    m_per_px_y = node.real_length_m / warped_h

    cfg = {
        'src_points': [[float(p[0]), float(p[1])] for p in src_points],
        'dst_points': dst.tolist(),
        'warped_size': [warped_w, warped_h],
        'real_width_m': node.real_width_m,
        'real_length_m': node.real_length_m,
        'm_per_px_x': float(m_per_px_x),
        'm_per_px_y': float(m_per_px_y),
        'M': M.tolist(),
        'Minv': Minv.tolist(),
        'camera_image_size': [w_img, h_img],
    }

    os.makedirs(os.path.dirname(node.output_yaml), exist_ok=True)
    with open(node.output_yaml, 'w') as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    node.get_logger().info(f'Saved calibration → {node.output_yaml}')

    # ALSO write a copy back to the source tree so the calibration
    # survives `colcon build` (which wipes the install/ directory).
    src_yaml = _find_source_config_path()
    if src_yaml and src_yaml != node.output_yaml:
        try:
            os.makedirs(os.path.dirname(src_yaml), exist_ok=True)
            with open(src_yaml, 'w') as f:
                yaml.safe_dump(cfg, f, sort_keys=False)
            node.get_logger().info(f'         (source copy → {src_yaml})')
        except Exception as e:
            node.get_logger().warn(f'Could not write source copy: {e}')
    node.get_logger().info(
        f'  m_per_px_x = {m_per_px_x:.5f}  ({node.real_width_m:.2f} m / {dst_width_px:.0f} px)'
    )
    node.get_logger().info(
        f'  m_per_px_y = {m_per_px_y:.5f}  ({node.real_length_m:.2f} m / {warped_h} px)'
    )


# Mouse-click state for the OpenCV window
_clicks: list = []


def _mouse_cb(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(_clicks) < 4:
        _clicks.append((x, y))


def _draw_clicks_overlay(frame, clicks):
    img = frame.copy()
    for i, (x, y) in enumerate(clicks):
        cv2.circle(img, (x, y), 8, (0, 255, 0), -1)
        cv2.circle(img, (x, y), 9, (0, 0, 0), 1)
        cv2.putText(img, str(i + 1), (x + 12, y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
                    cv2.LINE_AA)
    if len(clicks) >= 2:
        pts = np.array(clicks, dtype=np.int32)
        cv2.polylines(img, [pts], len(clicks) == 4, (0, 255, 255), 2)
    return img


def _draw_text(img, lines, y0=24, scale=0.55):
    for i, line in enumerate(lines):
        cv2.putText(img, line, (10, y0 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4,
                    cv2.LINE_AA)
        cv2.putText(img, line, (10, y0 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 0), 1,
                    cv2.LINE_AA)


def main(args=None):
    rclpy.init(args=args)
    node = FrameGrabber()

    node.get_logger().info('Waiting for first camera frame...')
    timeout_s = 10.0
    t0 = time.time()
    while node.latest_frame is None:
        rclpy.spin_once(node, timeout_sec=0.1)
        if time.time() - t0 > timeout_s:
            node.get_logger().error('No camera frame received in 10s — is sim running?')
            node.destroy_node()
            rclpy.shutdown()
            return

    # ── Live preview, wait for capture ─────────────────────────────
    while True:
        live = node.latest_frame.copy()
        _draw_text(live, [
            f"LIVE  topic={node.get_parameter('image_topic').value}",
            f"real_width={node.real_width_m:.2f} m  real_length={node.real_length_m:.2f} m",
            "[s] capture frame   [q] quit",
        ])
        cv2.imshow('IPM calibrate', live)
        key = cv2.waitKey(30) & 0xFF
        rclpy.spin_once(node, timeout_sec=0)
        if key == ord('s'):
            captured = node.latest_frame.copy()
            break
        if key == ord('q'):
            cv2.destroyAllWindows()
            node.destroy_node()
            rclpy.shutdown()
            return

    # ── Click 4 points, with warped-preview confirm step ──────────
    _clicks.clear()
    cv2.setMouseCallback('IPM calibrate', _mouse_cb)
    while True:
        overlay = _draw_clicks_overlay(captured, _clicks)
        if len(_clicks) < 4:
            _draw_text(overlay, [
                "CAPTURE  click 4 points in order:",
                "  1) bottom-left   2) bottom-right",
                "  3) top-right     4) top-left",
                "[r] reset   [q] quit",
            ])
            cv2.imshow('IPM calibrate', overlay)
        else:
            # Compute the warped preview
            src = np.float32(_clicks)
            warped_w, warped_h = 640, 480
            margin = 0.20 * warped_w
            dst = np.float32([
                [margin,           warped_h - 1],
                [warped_w - margin, warped_h - 1],
                [warped_w - margin, 0],
                [margin,           0],
            ])
            M = cv2.getPerspectiveTransform(src, dst)
            preview = cv2.warpPerspective(captured, M, (warped_w, warped_h))

            # Draw vertical centre line so user can visually check parallelism
            cv2.line(preview, (warped_w // 2, 0), (warped_w // 2, warped_h),
                     (0, 255, 0), 1)
            cv2.line(preview, (int(margin), 0), (int(margin), warped_h),
                     (0, 255, 255), 1)
            cv2.line(preview, (int(warped_w - margin), 0),
                     (int(warped_w - margin), warped_h),
                     (0, 255, 255), 1)
            _draw_text(preview, [
                "WARPED PREVIEW",
                "Lane lines should be roughly parallel + vertical here.",
                "[s] save   [r] redo clicks   [q] quit",
            ])

            # Show both: original (with clicks) on the left, warped on the right
            disp = np.hstack([
                cv2.resize(overlay, (warped_w, warped_h)),
                preview,
            ])
            cv2.imshow('IPM calibrate', disp)

        key = cv2.waitKey(30) & 0xFF
        rclpy.spin_once(node, timeout_sec=0)

        if key == ord('q'):
            break
        if key == ord('r'):
            _clicks.clear()
            cv2.destroyAllWindows()
            cv2.imshow('IPM calibrate', captured)
            cv2.setMouseCallback('IPM calibrate', _mouse_cb)
        if key == ord('s') and len(_clicks) == 4:
            save_calibration(node, _clicks, captured.shape)
            node.get_logger().info("Press 'q' to quit, or 'r' to recalibrate.")

    cv2.destroyAllWindows()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
