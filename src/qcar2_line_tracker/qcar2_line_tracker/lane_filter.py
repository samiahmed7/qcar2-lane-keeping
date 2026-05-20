#!/usr/bin/env python3
"""
Lane-filter node — IPM warp + sliding-window detection + Clothoid EKF.

This node runs in PARALLEL with the existing lane_follower. It does NOT
publish cmd_vel. Its purpose is to verify the EKF produces sensible lane
estimates before we (Phase A3) switch steering to use it.

Subscribes
----------
  /qcar2/front_camera/image     sensor_msgs/Image   (BGR via cv_bridge)
  /model/qcar2/odometry         nav_msgs/Odometry   (for v, omega)

Publishes
---------
  /qcar2/lane_filter/debug      sensor_msgs/Image
        Composite: camera view + warped overlay + EKF projected lane
  /qcar2/lane_filter/state      std_msgs/Float32MultiArray
        [c0, c1, c2, c3, lane_width_m, age_s]
"""
import os
from pathlib import Path
import math
import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge

from qcar2_line_tracker.ipm import IPM, default_config_path
from qcar2_line_tracker.lane_detector import LaneDetector
from qcar2_line_tracker.clothoid_ekf import ClothoidEKF


class LaneFilterNode(Node):
    def __init__(self):
        super().__init__('lane_filter')

        self.declare_parameter('ipm_yaml', default_config_path())
        self.declare_parameter('image_topic', '/qcar2/front_camera/image')
        self.declare_parameter('odom_topic', '/model/qcar2/odometry')
        self.declare_parameter('predict_only_max_age', 3.0)   # seconds

        self.declare_parameter('white_v_low', 180)
        self.declare_parameter('n_windows', 9)
        self.declare_parameter('window_margin', 50)
        self.declare_parameter('min_pixels_per_side', 100)

        # Confidence thresholds (above min_pixels_per_side, "trust this side")
        self.declare_parameter('confident_pixels_per_side', 600)
        self.declare_parameter('min_windows_confident', 3)
        self.declare_parameter('default_lane_full_width_m', 0.80)
        self.declare_parameter('lane_width_alpha', 0.10)   # EMA factor for width

        # EKF tuning
        self.declare_parameter('q_c0', 0.001)
        self.declare_parameter('q_c1', 0.001)
        self.declare_parameter('q_c2', 0.0001)
        self.declare_parameter('q_c3', 0.00001)
        self.declare_parameter('r_meas', 0.01)

        # Steering control (drive_enable: opt-in. Stops the centroid driver
        # and uses EKF state as the truth signal instead.)
        self.declare_parameter('drive_enable', False)
        self.declare_parameter('drive_speed', 0.15)
        self.declare_parameter('drive_crawl_speed', 0.05)
        self.declare_parameter('drive_k_lat', 1.5)    # Stanley cross-track gain
        self.declare_parameter('drive_k_head', 1.0)   # heading-error gain
        self.declare_parameter('drive_k_ff_curv', 0.5)  # curvature feed-forward
        self.declare_parameter('drive_steering_limit', 0.5)   # rad
        self.declare_parameter('drive_rate_limit', 0.06)      # rad / frame
        self.declare_parameter('drive_lookahead_m', 0.7)      # for curvature FF

        ipm_yaml = self.get_parameter('ipm_yaml').value
        if not os.path.exists(ipm_yaml):
            self.get_logger().fatal(
                f'IPM calibration not found at {ipm_yaml}. '
                f'Run `ros2 launch qcar2_line_tracker ipm_calibrate.launch.py` first.'
            )
            raise SystemExit(2)
        self.ipm = IPM(ipm_yaml)
        self.get_logger().info(
            f'IPM loaded: warped {self.ipm.warped_w}x{self.ipm.warped_h}, '
            f'm/px_x={self.ipm.m_per_px_x:.5f}  m/px_y={self.ipm.m_per_px_y:.5f}'
        )

        self.detector = LaneDetector(
            n_windows=int(self.get_parameter('n_windows').value),
            window_margin=int(self.get_parameter('window_margin').value),
            min_pixels_per_side=int(self.get_parameter('min_pixels_per_side').value),
            white_v_low=int(self.get_parameter('white_v_low').value),
        )

        self.ekf = ClothoidEKF(
            q_diag=(
                float(self.get_parameter('q_c0').value),
                float(self.get_parameter('q_c1').value),
                float(self.get_parameter('q_c2').value),
                float(self.get_parameter('q_c3').value),
            ),
            r_meas=float(self.get_parameter('r_meas').value),
        )
        self.predict_only_max_age = float(
            self.get_parameter('predict_only_max_age').value
        )
        self.confident_pixels = int(
            self.get_parameter('confident_pixels_per_side').value
        )
        self.min_windows_confident = int(
            self.get_parameter('min_windows_confident').value
        )
        self.lane_full_width_m = float(
            self.get_parameter('default_lane_full_width_m').value
        )
        self.lane_width_alpha = float(self.get_parameter('lane_width_alpha').value)

        self.drive_enable        = bool(self.get_parameter('drive_enable').value)
        self.drive_speed         = float(self.get_parameter('drive_speed').value)
        self.drive_crawl_speed   = float(self.get_parameter('drive_crawl_speed').value)
        self.drive_k_lat         = float(self.get_parameter('drive_k_lat').value)
        self.drive_k_head        = float(self.get_parameter('drive_k_head').value)
        self.drive_k_ff_curv     = float(self.get_parameter('drive_k_ff_curv').value)
        self.drive_steering_limit = float(self.get_parameter('drive_steering_limit').value)
        self.drive_rate_limit    = float(self.get_parameter('drive_rate_limit').value)
        self.drive_lookahead_m   = float(self.get_parameter('drive_lookahead_m').value)

        self.bridge = CvBridge()
        self.last_odom = None       # (v, omega, stamp_sec)
        self.last_predict_stamp = None
        self.frame_count = 0
        self.last_mode = 'INIT'
        self.last_steering_cmd = 0.0

        self.create_subscription(
            Image, self.get_parameter('image_topic').value,
            self._on_image, 10
        )
        self.create_subscription(
            Odometry, self.get_parameter('odom_topic').value,
            self._on_odom, 10
        )
        self.debug_pub = self.create_publisher(
            Image, '/qcar2/lane_filter/debug', 10
        )
        self.state_pub = self.create_publisher(
            Float32MultiArray, '/qcar2/lane_filter/state', 10
        )
        self.cmd_pub = self.create_publisher(
            Twist, '/model/qcar2/cmd_vel', 10
        ) if self.drive_enable else None

        self.get_logger().info('Lane filter (IPM + sliding-window + Clothoid EKF) up.')
        if self.drive_enable:
            self.get_logger().warn(
                '** drive_enable=TRUE — this node WILL publish to /model/qcar2/cmd_vel. **'
            )
            self.get_logger().warn(
                '   STOP any other driver (lane_follower) to avoid command fights.'
            )

    # ────────────────────────────────────────────────────────────
    def _on_odom(self, msg: Odometry):
        v = float(msg.twist.twist.linear.x)
        omega = float(msg.twist.twist.angular.z)
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.last_odom = (v, omega, stamp)

    def _stamp_to_sec(self, stamp):
        return stamp.sec + stamp.nanosec * 1e-9

    def _predict_to_now(self, now_sec):
        if self.last_predict_stamp is None or self.last_odom is None:
            self.last_predict_stamp = now_sec
            return
        dt = now_sec - self.last_predict_stamp
        if dt <= 0:
            return
        v, omega, _ = self.last_odom
        self.ekf.predict(v=v, omega=omega, dt=dt)
        self.last_predict_stamp = now_sec

    # ────────────────────────────────────────────────────────────
    def _on_image(self, msg: Image):
        self.frame_count += 1
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

        # 1. Warp to bird's-eye
        bev = self.ipm.warp(frame)

        # 2. Detect lane lines in BEV.
        #    If the EKF already has a state estimate, use it to PREDICT where
        #    each lane line should be at the bottom of the BEV and seed the
        #    sliding window starts there. This prevents the detector from
        #    splitting one curved line into two phantom halves on tight
        #    corners — the windows look in the right places, not just in
        #    "the left half" and "the right half" of the image.
        seed_left = seed_right = None
        if self.ekf.initialized:
            # Bottom of BEV image corresponds to s ≈ 0 (right under the car).
            # Use a small forward offset to avoid the very edge.
            s_bottom_m = 0.10
            y_centre_m = self.ekf.evaluate(s_bottom_m)
            half = self.lane_full_width_m / 2.0
            # left line is at y_centre + half (left is +y in our convention)
            seed_left,  _ = self.ipm.metres_to_px(s_bottom_m, y_centre_m + half)
            seed_right, _ = self.ipm.metres_to_px(s_bottom_m, y_centre_m - half)

        det = self.detector.detect(bev, seed_left=seed_left, seed_right=seed_right)
        left = det['left_centres']
        right = det['right_centres']

        # 3. Confidence per side. A side is "confident" only if:
        #    (a) total tracked pixels exceed threshold,
        #    (b) at least N windows recentred (not just got pixels),
        #    (c) the recentred window x-positions form a coherent line
        #        (linear fit residuals are small — the windows aren't
        #         scattered all over the place).
        def _side_confident(centres, total_pixels):
            if total_pixels < self.confident_pixels:
                return False
            if len(centres) < self.min_windows_confident:
                return False
            xs = np.array([c[0] for c in centres], dtype=np.float64)
            ys = np.array([c[1] for c in centres], dtype=np.float64)
            if len(xs) >= 2 and np.ptp(ys) > 0:
                # Fit x = a*y + b and check residual std
                a, b = np.polyfit(ys, xs, 1)
                residuals = xs - (a * ys + b)
                # Phantom lines have very large residuals;
                # real lines stay within ~20 px of the fit.
                if float(np.std(residuals)) > 30.0:
                    return False
            return True

        left_confident  = _side_confident(left,  det['left_count'])
        right_confident = _side_confident(right, det['right_count'])

        # 4. Build CENTRELINE measurements. We use whichever sides are
        #    confident; if only one side is, infer the centreline by
        #    projecting that side inward by half the known lane width.
        centreline_meas = []
        lane_width_obs = None
        half = self.lane_full_width_m / 2.0

        if left_confident and right_confident:
            # Pair by window index (both walked bottom-up). Real centreline.
            n = min(len(left), len(right))
            widths = []
            pair_meas = []
            for i in range(n):
                lx, ly = left[i]
                rx, ry = right[i]
                cy_px = (ly + ry) / 2.0
                cx_px = (lx + rx) / 2.0
                s_m, y_m = self.ipm.px_to_metres(cx_px, cy_px)
                pair_meas.append((s_m, y_m))
                left_y_m = self.ipm.px_to_metres(lx, ly)[1]
                right_y_m = self.ipm.px_to_metres(rx, ry)[1]
                widths.append(left_y_m - right_y_m)   # +left → left > right
            if widths:
                lane_width_obs = float(np.median(widths))

            # SANITY: if the two "lines" are too close in metres (or even
            # crossed, i.e. left < right) they are the same physical line
            # split into two halves by the detector. Drop the weaker side
            # and downgrade to single-line mode.
            min_plausible = 0.5 * self.lane_full_width_m   # half of expected
            if lane_width_obs is None or lane_width_obs < min_plausible:
                if det['left_count'] >= det['right_count']:
                    # Use LEFT as the real line, project inward
                    centreline_meas = []
                    for lx, ly in left:
                        s_m, y_left_m = self.ipm.px_to_metres(lx, ly)
                        centreline_meas.append((s_m, y_left_m - half))
                    mode = 'AMBIG→L'
                else:
                    centreline_meas = []
                    for rx, ry in right:
                        s_m, y_right_m = self.ipm.px_to_metres(rx, ry)
                        centreline_meas.append((s_m, y_right_m + half))
                    mode = 'AMBIG→R'
            else:
                centreline_meas = pair_meas
                # EMA-update running width estimate only when we trust both
                if 0.4 < lane_width_obs < 1.5:
                    self.lane_full_width_m = (
                        (1.0 - self.lane_width_alpha) * self.lane_full_width_m
                        + self.lane_width_alpha * lane_width_obs
                    )
                    half = self.lane_full_width_m / 2.0
                mode = 'BOTH'
        elif left_confident:
            # Only LEFT line is trustworthy. Project it inward (to the right,
            # since left line is to the left of centreline) by half-width.
            for lx, ly in left:
                s_m, y_left_m = self.ipm.px_to_metres(lx, ly)
                centreline_meas.append((s_m, y_left_m - half))
            mode = 'LEFT_ONLY'
        elif right_confident:
            # Only RIGHT line is trustworthy. Project inward (to the left).
            for rx, ry in right:
                s_m, y_right_m = self.ipm.px_to_metres(rx, ry)
                centreline_meas.append((s_m, y_right_m + half))
            mode = 'RIGHT_ONLY'
        else:
            mode = 'COAST'   # no confident detection — EKF predict-only

        self.last_mode = mode

        # 4. EKF predict (using last odom) + update (if we have measurements)
        now_sec = self._stamp_to_sec(msg.header.stamp)
        self._predict_to_now(now_sec)
        if len(centreline_meas) >= 2:
            self.ekf.update(centreline_meas)

        # 5. Stanley-style steering controller (only if drive_enable=True).
        #    Uses EKF state directly in METRIC space:
        #        c0  — lateral offset of lane centre (+ left)
        #        c1  — lane heading at vehicle (+ left)
        #        c2  — curvature ahead (1/m)
        #    Control law: δ = k_head·c1 + atan(k_lat·c0/v) + k_ff·κ_lookahead
        c0, c1, c2, c3 = self.ekf.state_tuple()
        if self.drive_enable and self.cmd_pub is not None:
            self._publish_drive_cmd(mode, c0, c1, c2)

        # 6. Publish state
        sm = Float32MultiArray()
        age = (now_sec - self.last_predict_stamp) if self.last_predict_stamp else 0.0
        sm.data = [float(c0), float(c1), float(c2), float(c3),
                   float(lane_width_obs if lane_width_obs else 0.0),
                   float(age)]
        self.state_pub.publish(sm)

        # 6. Build debug composite
        self._publish_debug(frame, bev, det, centreline_meas)

        if self.frame_count % 30 == 1:
            width_str = f'{lane_width_obs:.3f}' if lane_width_obs else 'n/a'
            self.get_logger().info(
                f'#{self.frame_count} mode={mode} '
                f'L_n={det["left_count"]} R_n={det["right_count"]} '
                f'meas={len(centreline_meas)} '
                f'c0={c0:+.3f}m c1={c1:+.3f}rad κ={c2:+.3f}/m  '
                f'width_obs={width_str} width_est={self.lane_full_width_m:.3f}'
            )

    # ────────────────────────────────────────────────────────────
    def _publish_drive_cmd(self, mode, c0, c1, c2):
        """Stanley-style controller using EKF state."""
        twist = Twist()

        # Cap speed by mode: crawl when we're coasting on prediction alone
        if mode == 'COAST' or not self.ekf.initialized:
            twist.linear.x = self.drive_crawl_speed
            # Stanley would still produce a steering command, but if perception
            # is dark we'd rather hold the wheel close to last good state.
            steering = 0.5 * self.last_steering_cmd
        else:
            v = self.drive_speed

            # Stanley control law in radians.
            # Heading term: align car with lane direction (+c1 → lane turns left → steer left).
            head_term = self.drive_k_head * c1

            # Cross-track term: pull back to lane centre.
            # +c0 → lane centre is to the LEFT → steer left (positive δ).
            lat_term = math.atan2(self.drive_k_lat * c0, max(v, 0.05))

            # Curvature feed-forward: anticipate upcoming curve by κ at lookahead.
            kappa_ahead = self.ekf.evaluate_curvature(self.drive_lookahead_m)
            ff_term = self.drive_k_ff_curv * kappa_ahead

            steering = head_term + lat_term + ff_term
            twist.linear.x = v

        # Saturate
        steering = float(np.clip(
            steering, -self.drive_steering_limit, self.drive_steering_limit))

        # Rate limit on steering (smooths actuator + survives status flips)
        max_step = self.drive_rate_limit
        delta = steering - self.last_steering_cmd
        delta = float(np.clip(delta, -max_step, max_step))
        steering = self.last_steering_cmd + delta
        self.last_steering_cmd = steering

        # ROS convention: +angular.z = turn LEFT, which matches our +δ sign.
        twist.angular.z = steering

        self.cmd_pub.publish(twist)

    # ────────────────────────────────────────────────────────────
    def _publish_debug(self, cam_frame, bev, det, meas):
        """Composite:  camera-with-overlay  |  BEV with windows + EKF curve."""
        h, w = cam_frame.shape[:2]
        bev_h, bev_w = bev.shape[:2]

        # ---- BEV panel: detector overlay + EKF projected centreline ----
        bev_dbg = det['overlay'].copy()
        bev_h_px, bev_w_px = bev_dbg.shape[:2]

        # Mark where the sliding windows were SEEDED (predicted by EKF)
        for base, colour in (
            (det.get('left_base'),  (255, 255, 0)),    # cyan-ish
            (det.get('right_base'), (0, 255, 255)),    # yellow
        ):
            if base is not None:
                cv2.drawMarker(bev_dbg, (int(base), bev_h_px - 5),
                               colour, cv2.MARKER_TRIANGLE_UP, 12, 2)

        # Draw the EKF-projected centreline across the BEV
        if self.ekf.initialized:
            pts = []
            for s_m in np.linspace(0.1, 2.0, 30):
                y_m = self.ekf.evaluate(s_m)
                u, v = self.ipm.metres_to_px(s_m, y_m)
                if 0 <= u < bev_w and 0 <= v < bev_h:
                    pts.append((int(u), int(v)))
            if len(pts) >= 2:
                cv2.polylines(bev_dbg, [np.array(pts, dtype=np.int32)],
                              False, (0, 255, 0), 3)
            # Mark lateral offset at vehicle position (s=0)
            u0, v0 = self.ipm.metres_to_px(0.05, self.ekf.lateral_offset())
            cv2.circle(bev_dbg, (int(u0), int(v0)), 6, (0, 255, 0), -1)
            cv2.line(bev_dbg, (bev_w // 2, bev_h - 5),
                     (int(u0), bev_h - 5), (0, 255, 0), 2)

        # ---- Camera panel: unwarp EKF centreline back and overlay ----
        cam_dbg = cam_frame.copy()
        if self.ekf.initialized:
            bev_pts = []
            for s_m in np.linspace(0.1, 2.0, 30):
                y_m = self.ekf.evaluate(s_m)
                bev_pts.append(self.ipm.metres_to_px(s_m, y_m))
            bev_pts_np = np.array(bev_pts, dtype=np.float32)
            cam_pts = self.ipm.unwarp_points(bev_pts_np).astype(np.int32)
            cv2.polylines(cam_dbg, [cam_pts], False, (0, 255, 0), 3)

        # Draw the source trapezoid on the camera view (faint, for reference)
        src = self.ipm.src.astype(np.int32)
        cv2.polylines(cam_dbg, [src], True, (255, 0, 255), 1)

        # HUD on the camera view
        c0, c1, c2, c3 = self.ekf.state_tuple()
        def put(img, text, y, scale=0.6, colour=(0, 255, 255)):
            cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                        scale, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                        scale, colour, 2, cv2.LINE_AA)
        mode_colour = {
            'BOTH':       (0, 255, 0),
            'LEFT_ONLY':  (0, 200, 255),
            'RIGHT_ONLY': (0, 200, 255),
            'AMBIG→L':    (0, 165, 255),    # orange — saw "two" lines too close
            'AMBIG→R':    (0, 165, 255),
            'COAST':      (0, 0, 255),
            'INIT':       (200, 200, 200),
        }.get(self.last_mode, (255, 255, 255))
        put(cam_dbg, f'mode: {self.last_mode}', 28, colour=mode_colour)
        put(cam_dbg, f'offset  c0 = {c0:+.3f} m',       54)
        put(cam_dbg, f'heading c1 = {math.degrees(c1):+.2f}°', 78)
        put(cam_dbg, f'curvature c2 = {c2:+.3f} /m',    102)
        put(cam_dbg, f'lane width est = {self.lane_full_width_m:.3f} m', 126, scale=0.5)
        put(cam_dbg, f'detections: L={det["left_count"]} R={det["right_count"]}',
            148, scale=0.5)
        if self.drive_enable:
            drive_colour = (0, 255, 0) if self.last_mode != 'COAST' else (0, 165, 255)
            put(cam_dbg,
                f'DRIVING  steer={self.last_steering_cmd:+.3f} rad',
                172, scale=0.55, colour=drive_colour)

        # ---- Compose side-by-side (camera | BEV), match height ----
        scale = h / float(bev_h)
        bev_resized = cv2.resize(bev_dbg, (int(bev_w * scale), h))
        composite = np.hstack([cam_dbg, bev_resized])

        self.debug_pub.publish(self.bridge.cv2_to_imgmsg(composite, encoding='bgr8'))


def main(args=None):
    rclpy.init(args=args)
    try:
        node = LaneFilterNode()
    except SystemExit:
        rclpy.shutdown()
        return
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
