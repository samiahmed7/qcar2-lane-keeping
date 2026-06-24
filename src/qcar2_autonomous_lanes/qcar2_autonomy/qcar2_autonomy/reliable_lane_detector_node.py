#!/usr/bin/env python3
"""Robust two-line HSV+BEV right-lane detector — sliding window + polynomial fit.

Perception half of the "MPC guides, vision centres" split:
    * mpc_reference_planner_node owns WHERE to go (waypoints, handles
      T-junction / roundabout where vision is unreliable).
    * This node owns staying CENTRED in the right lane on every section,
      including curves, by fitting a 2nd-order polynomial to each line.

Pipeline (every frame):
    camera -> HSV white mask -> morphological cleanup
           -> IPM bird's-eye warp
           -> sliding windows (N strips from bottom to top)
              -> centroids of white pixels in each window
           -> 2nd-order polynomial fit to centroids (per line)
           -> evaluate polynomials at a fixed near-car y
           -> right-lane centre = midpoint(dash_x, edge_x)
           -> error_px = centre - image_centre -> LaneModel

The polynomial adapts to curves: each window strip searches near the
previous detection, so the tracked position naturally follows a curved line
rather than relying on a single column histogram.

Publishes (drop-in for bev_lane_detector_node):
    /qcar2/lane/model         qcar2_msgs/LaneModel
    /qcar2/lane/debug_image   sensor_msgs/Image
"""
import math

import cv2
import numpy as np
import rclpy
from qcar2_msgs.msg import LaneModel
from rclpy.node import Node
from sensor_msgs.msg import Image

from qcar2_autonomy.image_msg_utils import bgr_to_image_msg, image_msg_to_bgr

IPM_SRC_RATIOS = [
    (0.003, 0.502), (0.336, 0.294), (0.666, 0.296), (0.994, 0.504),
]
IPM_DST_RATIOS = [
    (0.200, 1.000), (0.200, 0.000), (0.800, 0.000), (0.800, 1.000),
]


class ReliableLaneDetectorNode(Node):
    def __init__(self):
        super().__init__("reliable_lane_detector_node")

        self.declare_parameter("image_topic",      "/qcar2/front_camera/image")
        self.declare_parameter("lane_model_topic", "/qcar2/lane/model")
        self.declare_parameter("debug_topic",      "/qcar2/lane/debug_image")
        self.declare_parameter("image_width",  640)
        self.declare_parameter("image_height", 480)
        self.declare_parameter("debug_enabled", True)
        self.declare_parameter("hsv_lo", [0, 0, 180])
        self.declare_parameter("hsv_hi", [180, 80, 255])
        self.declare_parameter("morph_kernel", 5)
        # sliding window
        self.declare_parameter("n_windows",          10)
        self.declare_parameter("window_half_w_px",   80)
        self.declare_parameter("min_pix_per_window", 30)
        self.declare_parameter("hist_band_frac",     0.50)   # bottom fraction for init hist
        self.declare_parameter("eval_y_frac",        0.80)   # where to eval polynomial (0=top,1=bottom)
        # lane geometry
        self.declare_parameter("nominal_lane_width_px", 394.0)
        self.declare_parameter("min_lane_width_px",     220.0)
        self.declare_parameter("max_lane_width_px",     540.0)
        self.declare_parameter("target_ema_alpha",   0.30)
        self.declare_parameter("width_ema_alpha",    0.20)
        self.declare_parameter("confidence_both",    0.70)
        self.declare_parameter("confidence_single",  0.50)

        self.W = int(self.get_parameter("image_width").value)
        self.H = int(self.get_parameter("image_height").value)
        self.center_px  = 0.5 * self.W
        self.debug_en   = bool(self.get_parameter("debug_enabled").value)
        self.hsv_lo     = np.array(self.get_parameter("hsv_lo").value, dtype=np.uint8)
        self.hsv_hi     = np.array(self.get_parameter("hsv_hi").value, dtype=np.uint8)
        k = max(1, int(self.get_parameter("morph_kernel").value))
        self.morph_k    = np.ones((k, k), np.uint8)
        self.n_windows  = int(self.get_parameter("n_windows").value)
        self.win_hw     = int(self.get_parameter("window_half_w_px").value)
        self.min_pix    = int(self.get_parameter("min_pix_per_window").value)
        self.band_frac  = float(self.get_parameter("hist_band_frac").value)
        self.eval_y_fr  = float(self.get_parameter("eval_y_frac").value)
        self.nom_width  = float(self.get_parameter("nominal_lane_width_px").value)
        self.min_width  = float(self.get_parameter("min_lane_width_px").value)
        self.max_width  = float(self.get_parameter("max_lane_width_px").value)
        self.tgt_alpha  = float(self.get_parameter("target_ema_alpha").value)
        self.w_alpha    = float(self.get_parameter("width_ema_alpha").value)
        self.conf_both  = float(self.get_parameter("confidence_both").value)
        self.conf_single= float(self.get_parameter("confidence_single").value)

        src = np.float32([[rx*self.W, ry*self.H] for rx, ry in IPM_SRC_RATIOS])
        dst = np.float32([[rx*self.W, ry*self.H] for rx, ry in IPM_DST_RATIOS])
        self.M = cv2.getPerspectiveTransform(src, dst)
        self.ipm_src = src   # for debug overlay

        self.lane_width_ema = self.nom_width
        self.target_ema     = None

        self.lane_pub  = self.create_publisher(LaneModel, self.get_parameter("lane_model_topic").value, 10)
        self.debug_pub = self.create_publisher(Image,     self.get_parameter("debug_topic").value,      10)
        self.create_subscription(Image, self.get_parameter("image_topic").value, self._on_image, 10)
        self.get_logger().info("Sliding-window HSV+BEV lane detector ready")

    # ------------------------------------------------------------------ #
    def _on_image(self, msg: Image):
        try:
            bgr = image_msg_to_bgr(msg)
        except Exception as exc:
            self.get_logger().warn(f"image decode: {exc}", throttle_duration_sec=2.0)
            return
        if bgr.shape[1] != self.W or bgr.shape[0] != self.H:
            bgr = cv2.resize(bgr, (self.W, self.H))

        # 1. HSV white mask + morphological open (kill speckle)
        mask = cv2.inRange(cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV), self.hsv_lo, self.hsv_hi)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.morph_k)
        # 2. bird's-eye warp
        bev = cv2.warpPerspective(mask, self.M, (self.W, self.H), flags=cv2.INTER_LINEAR)
        # 3. sliding windows -> polynomial fits
        (d_poly, d_n, d_pts,
         e_poly, e_n, e_pts) = self._sliding_window_fit(bev)
        # 4. build lane model
        lane = self._build_model(msg.header, d_poly, d_n, d_pts, e_poly, e_n, e_pts)
        self.lane_pub.publish(lane)

        self.get_logger().info(
            f"dash_windows={d_n}/{self.n_windows} edge_windows={e_n}/{self.n_windows} "
            f"err={'nan' if not math.isfinite(lane.error_px) else f'{lane.error_px:.1f}'}px "
            f"conf={lane.confidence:.2f} width={self.lane_width_ema:.0f}px",
            throttle_duration_sec=0.5,
        )
        if self.debug_en:
            self._publish_debug(bgr, bev, d_pts, e_pts, d_poly, e_poly, lane)

    # ------------------------------------------------------------------ #
    def _sliding_window_fit(self, bev):
        """Sliding window search + 2nd-order polynomial fit for both lines.

        Returns:
            dash_poly  (ndarray|None): polyfit [a,b,c] for centre dash
            dash_n     (int): windows with valid detections
            dash_pts   (ndarray|None): (N,2) array of [y, x] centroids
            edge_poly  (ndarray|None): polyfit for right edge
            edge_n     (int)
            edge_pts   (ndarray|None)
        """
        h, w = bev.shape
        band0 = int(h * self.band_frac)

        # Initial peak positions from bottom histogram
        hist = bev[band0:, :].sum(axis=0).astype(np.float32)
        hist = np.convolve(hist, np.ones(21, np.float32) / 21.0, mode="same")
        mid = w // 2
        left_base  = int(np.argmax(hist[:mid])) if hist[:mid].max() > 0 else mid // 2
        right_base = int(np.argmax(hist[mid:])) + mid if hist[mid:].max() > 0 else mid + mid // 2

        win_h = max(1, (h - band0) // self.n_windows)
        cur_l, cur_r = left_base, right_base
        l_xs, l_ys, r_xs, r_ys = [], [], [], []

        for i in range(self.n_windows):
            y_hi = h - i * win_h
            y_lo = max(band0, h - (i + 1) * win_h)
            if y_lo >= y_hi:
                continue
            y_mid = (y_lo + y_hi) // 2

            # Left window (dash)
            xl0 = max(0, cur_l - self.win_hw); xl1 = min(w, cur_l + self.win_hw)
            strip = bev[y_lo:y_hi, xl0:xl1]
            nz = np.nonzero(strip)
            if len(nz[1]) >= self.min_pix:
                cx = int(np.mean(nz[1])) + xl0
                l_xs.append(cx); l_ys.append(y_mid)
                cur_l = cx

            # Right window (edge)
            xr0 = max(0, cur_r - self.win_hw); xr1 = min(w, cur_r + self.win_hw)
            strip = bev[y_lo:y_hi, xr0:xr1]
            nz = np.nonzero(strip)
            if len(nz[1]) >= self.min_pix:
                cx = int(np.mean(nz[1])) + xr0
                r_xs.append(cx); r_ys.append(y_mid)
                cur_r = cx

        def _fit(ys, xs):
            if len(ys) < 3:
                return None
            try:
                return np.polyfit(ys, xs, 2)
            except Exception:
                return None

        d_poly = _fit(l_ys, l_xs)
        e_poly = _fit(r_ys, r_xs)
        d_pts  = np.column_stack([l_ys, l_xs]) if l_ys else None
        e_pts  = np.column_stack([r_ys, r_xs]) if r_ys else None
        return d_poly, len(l_ys), d_pts, e_poly, len(r_ys), e_pts

    # ------------------------------------------------------------------ #
    def _build_model(self, header, d_poly, d_n, d_pts, e_poly, e_n, e_pts):
        lane = LaneModel()
        lane.header     = header
        lane.target_lane= "RIGHT"
        lane.curvature  = 0.0
        lane.left_x = lane.middle_x = lane.right_x = math.nan

        eval_y = int(self.H * self.eval_y_fr)

        def eval_poly(poly):
            if poly is None:
                return None
            v = float(np.polyval(poly, eval_y))
            return v if math.isfinite(v) else None

        dash_x = eval_poly(d_poly)
        edge_x = eval_poly(e_poly)

        # Reject if polynomial evaluation is outside plausible half
        if dash_x is not None and not (self.W * 0.02 < dash_x < self.center_px + 0.20 * self.W):
            dash_x = None
        if edge_x is not None and not (self.center_px - 0.20 * self.W < edge_x < self.W * 0.98):
            edge_x = None

        # The SOLID right edge is the stable anchor for the lateral target:
        # it is continuous, so it does not flicker like the dashed centre line.
        # The dash is only used to MEASURE the lane width (and add confidence)
        # when it sits a plausible distance to the LEFT of the edge. This kills
        # the dash-flicker oscillation and enforces the rule that the dash can
        # never coincide with the right line (width must exceed min_width).
        target = None
        confidence = 0.0
        half_w = 0.5 * self.lane_width_ema

        if edge_x is not None:
            cov = e_n / float(self.n_windows)
            base = self.conf_single
            if dash_x is not None:
                width = edge_x - dash_x
                if self.min_width <= width <= self.max_width:
                    # valid pair: dash is genuinely left of the edge
                    self.lane_width_ema = (
                        (1.0 - self.w_alpha) * self.lane_width_ema + self.w_alpha * width
                    )
                    half_w = 0.5 * self.lane_width_ema
                    base = self.conf_both
                    cov = (d_n + e_n) / (2.0 * self.n_windows)
                    lane.middle_visible = True;  lane.middle_x = float(dash_x)
                # else: dash too close to / far from the edge -> bogus, ignore it
                #       (this is the "dash on the right line" case)
            target = edge_x - half_w                       # anchor to the solid edge
            confidence = base * min(1.0, cov * 1.5)
            lane.right_visible = True;  lane.right_x = float(edge_x)

        elif dash_x is not None:
            # No edge visible: fall back to the dash (less reliable, intermittent).
            edge_est = dash_x + self.lane_width_ema
            if self.center_px < edge_est < self.W * 0.96:
                target = dash_x + half_w
                cov = d_n / float(self.n_windows)
                confidence = self.conf_single * 0.8 * min(1.0, cov * 1.5)
                lane.middle_visible = True;  lane.middle_x = float(dash_x)
                lane.right_x = float(edge_est)

        if target is None:
            self.target_ema = None
            lane.target_center_x = math.nan
            lane.error_px        = math.nan
            lane.estimated_lane_width_px = float(self.lane_width_ema)
            lane.confidence = 0.0
            return lane

        self.target_ema = (
            target if self.target_ema is None
            else (1.0 - self.tgt_alpha) * self.target_ema + self.tgt_alpha * target
        )
        lane.target_center_x   = float(self.target_ema)
        lane.right_lane_center_x = float(self.target_ema)
        lane.error_px          = float(self.target_ema - self.center_px)
        lane.estimated_lane_width_px = float(self.lane_width_ema)
        lane.confidence        = float(confidence)
        return lane

    # ------------------------------------------------------------------ #
    def _publish_debug(self, camera_bgr, bev, d_pts, e_pts, d_poly, e_poly, lane):
        # BEV with window centroids + polynomial curves + target
        vis = cv2.cvtColor(bev, cv2.COLOR_GRAY2BGR)
        h, w = vis.shape[:2]

        # Draw window centroids
        if d_pts is not None:
            for y, x in d_pts:
                cv2.circle(vis, (int(x), int(y)), 5, (0, 255, 255), -1)  # yellow = dash
        if e_pts is not None:
            for y, x in e_pts:
                cv2.circle(vis, (int(x), int(y)), 5, (0, 255, 0), -1)    # green = edge

        # Draw fitted polynomial curves
        ys = np.linspace(int(h * self.band_frac), h - 1, 60).astype(int)
        for poly, color in [(d_poly, (0, 220, 220)), (e_poly, (0, 220, 0))]:
            if poly is not None:
                xs = np.polyval(poly, ys).astype(int)
                for j in range(len(ys) - 1):
                    if 0 <= xs[j] < w and 0 <= xs[j+1] < w:
                        cv2.line(vis, (xs[j], ys[j]), (xs[j+1], ys[j+1]), color, 2)

        # Vertical markers
        eval_y = int(h * self.eval_y_fr)

        def vline(x, color, label):
            if x is None or (isinstance(x, float) and not math.isfinite(x)):
                return
            xi = int(round(x))
            cv2.line(vis, (xi, 0), (xi, h), color, 2)
            cv2.putText(vis, label, (max(2, xi - 28), 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

        vline(self.center_px, (255, 255, 0), "car")
        if math.isfinite(lane.target_center_x):
            vline(lane.target_center_x, (255, 0, 255), "target")

        # Eval-y horizontal guide
        cv2.line(vis, (0, eval_y), (w, eval_y), (100, 100, 100), 1)

        cv2.putText(vis,
            f"conf={lane.confidence:.2f} err="
            f"{'nan' if not math.isfinite(lane.error_px) else f'{lane.error_px:.1f}'}px",
            (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # Camera view with IPM polygon
        cam = camera_bgr.copy()
        pts = self.ipm_src.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(cam, [pts], True, (0, 255, 0), 2)
        combined = np.hstack((cam, vis))
        try:
            self.debug_pub.publish(bgr_to_image_msg(combined, header=lane.header))
        except (ValueError, cv2.error) as exc:
            self.get_logger().warn(f"debug: {exc}", throttle_duration_sec=2.0)


def main(args=None):
    rclpy.init(args=args)
    node = ReliableLaneDetectorNode()
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
