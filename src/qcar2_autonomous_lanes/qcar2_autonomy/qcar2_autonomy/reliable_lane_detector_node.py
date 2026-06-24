#!/usr/bin/env python3
"""Reliable two-line HSV+BEV right-lane detector for QCar2.

Perception half of the "MPC guides, vision centres" split:
    * mpc_reference_planner_node owns WHERE to go (waypoints through the
      T-junction / roundabout, where vision is unreliable), and
    * this node owns staying CENTRED in the right lane on the straights.

Pipeline, every frame, no classifier / memory / synthesis chains:
    camera -> HSV white mask -> IPM bird's-eye warp -> column histogram
           -> two peaks (dashed centre line + solid right edge)
           -> target = midpoint -> publish LaneModel(error_px, confidence...)

The planner consumes /qcar2/lane/model and shifts its waypoint reference
laterally by  -error_px * (lane_width_m / estimated_lane_width_px), clamped and
confidence/heading gated. So this node only reports a clean error_px
(target - image_centre, +ve = lane centre to the camera's right) and an honest
confidence; the planner disengages fusion itself at curves/junctions when
confidence drops, letting the pure waypoint MPC drive through.

Drop-in source for the MPC lane fusion (same topics as bev_lane_detector_node):
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

# Calibrated IPM, identical to bev_lane_detector_node. src corners are
# [near-left, far-left, far-right, near-right]; each maps to the matching dst
# corner so the warp is a clean rectification (NOT a scrambled correspondence).
IPM_SRC_RATIOS = [
    (0.003, 0.502),  # near left
    (0.336, 0.294),  # far left
    (0.666, 0.296),  # far right
    (0.994, 0.504),  # near right
]
IPM_DST_RATIOS = [
    (0.200, 1.000),  # near left
    (0.200, 0.000),  # far left
    (0.800, 0.000),  # far right
    (0.800, 1.000),  # near right
]


class ReliableLaneDetectorNode(Node):
    def __init__(self):
        super().__init__("reliable_lane_detector_node")

        self.declare_parameter("image_topic", "/qcar2/front_camera/image")
        self.declare_parameter("lane_model_topic", "/qcar2/lane/model")
        self.declare_parameter("debug_topic", "/qcar2/lane/debug_image")
        self.declare_parameter("hsv_lo", [0, 0, 180])
        self.declare_parameter("hsv_hi", [180, 80, 255])
        self.declare_parameter("image_width", 640)
        self.declare_parameter("image_height", 480)
        self.declare_parameter("debug_enabled", True)
        # histogram band: bottom fraction of the warp (nearest road)
        self.declare_parameter("hist_band_frac", 0.40)
        self.declare_parameter("min_col_pixels", 1500.0)
        self.declare_parameter("peak_merge_px", 40.0)
        # nominal dashed-centre -> right-edge spacing in BEV px (= 0.5 m road
        # half-width). Self-corrects to the measured value when both are seen.
        self.declare_parameter("nominal_lane_width_px", 394.0)
        self.declare_parameter("min_lane_width_px", 250.0)
        self.declare_parameter("max_lane_width_px", 520.0)
        self.declare_parameter("target_ema_alpha", 0.4)
        self.declare_parameter("width_ema_alpha", 0.2)
        self.declare_parameter("confidence_both", 0.70)
        self.declare_parameter("confidence_single", 0.55)
        # measured-confidence shaping
        self.declare_parameter("width_tolerance_px", 80.0)   # width-consistency falloff
        self.declare_parameter("stability_jump_px", 60.0)    # target jump that zeroes stability
        self.declare_parameter("morph_kernel", 5)            # HSV mask open/close kernel

        self.W = int(self.get_parameter("image_width").value)
        self.H = int(self.get_parameter("image_height").value)
        self.center_px = 0.5 * self.W
        self.hsv_lo = np.array(self.get_parameter("hsv_lo").value, dtype=np.uint8)
        self.hsv_hi = np.array(self.get_parameter("hsv_hi").value, dtype=np.uint8)
        self.debug_enabled = bool(self.get_parameter("debug_enabled").value)
        self.hist_band_frac = float(self.get_parameter("hist_band_frac").value)
        self.min_col_pixels = float(self.get_parameter("min_col_pixels").value)
        self.peak_merge_px = float(self.get_parameter("peak_merge_px").value)
        self.nominal_width = float(self.get_parameter("nominal_lane_width_px").value)
        self.min_width = float(self.get_parameter("min_lane_width_px").value)
        self.max_width = float(self.get_parameter("max_lane_width_px").value)
        self.target_alpha = float(self.get_parameter("target_ema_alpha").value)
        self.width_alpha = float(self.get_parameter("width_ema_alpha").value)
        self.conf_both = float(self.get_parameter("confidence_both").value)
        self.conf_single = float(self.get_parameter("confidence_single").value)
        self.width_tol_px = float(self.get_parameter("width_tolerance_px").value)
        self.stability_jump_px = float(self.get_parameter("stability_jump_px").value)
        k = max(1, int(self.get_parameter("morph_kernel").value))
        self.morph_kernel = np.ones((k, k), np.uint8)

        self.ipm_src = np.float32([[rx * self.W, ry * self.H] for rx, ry in IPM_SRC_RATIOS])
        self.ipm_dst = np.float32([[rx * self.W, ry * self.H] for rx, ry in IPM_DST_RATIOS])
        self.ipm_matrix = cv2.getPerspectiveTransform(self.ipm_src, self.ipm_dst)

        self.lane_width_ema = self.nominal_width
        self.target_ema = None

        self.lane_pub = self.create_publisher(
            LaneModel, self.get_parameter("lane_model_topic").value, 10
        )
        self.debug_pub = self.create_publisher(
            Image, self.get_parameter("debug_topic").value, 10
        )
        self.create_subscription(
            Image, self.get_parameter("image_topic").value, self._on_image, 10
        )
        self.get_logger().info("Reliable two-line BEV lane detector ready (feeds MPC lane fusion)")

    def _on_image(self, msg: Image):
        try:
            bgr = image_msg_to_bgr(msg)
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f"image conversion failed: {exc}", throttle_duration_sec=2.0)
            return
        if bgr.shape[1] != self.W or bgr.shape[0] != self.H:
            bgr = cv2.resize(bgr, (self.W, self.H))

        # 1. HSV white mask + morphological cleanup (kill speckle, fill gaps)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.hsv_lo, self.hsv_hi)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.morph_kernel)
        # 2. bird's-eye warp
        bev = cv2.warpPerspective(mask, self.ipm_matrix, (self.W, self.H), flags=cv2.INTER_LINEAR)

        # 3. column histogram over the bottom band -> 4. peaks
        band0 = int(self.H * self.hist_band_frac)
        hist = bev[band0:, :].sum(axis=0).astype(np.float32)
        hs = np.convolve(hist, np.ones(15, np.float32) / 15.0, mode="same")
        peaks = self._find_peaks(hs)
        dash, edge = self._classify(peaks, hs)

        lane = self._build_model(msg.header, dash, edge, hs, peaks)
        self.lane_pub.publish(lane)

        self.get_logger().info(
            f"dash={None if dash is None else int(dash)} "
            f"edge={None if edge is None else int(edge)} "
            f"err={'nan' if not math.isfinite(lane.error_px) else ('%.1f' % lane.error_px)}px "
            f"conf={lane.confidence:.2f} width={self.lane_width_ema:.0f}px",
            throttle_duration_sec=0.5,
        )

        if self.debug_enabled:
            self._publish_debug(bgr, bev, dash, edge, lane)

    def _find_peaks(self, hs):
        thr = max(hs.max() * 0.25, self.min_col_pixels)
        peaks = []
        for x in range(2, self.W - 2):
            if hs[x] >= thr and hs[x] >= hs[x - 1] and hs[x] >= hs[x + 1]:
                if not peaks or x - peaks[-1] > self.peak_merge_px:
                    peaks.append(x)
                elif hs[x] > hs[peaks[-1]]:
                    peaks[-1] = x
        return peaks

    def _classify(self, peaks, hs):
        """Pick dashed-centre (left of car) + right-edge (right of car).

        Robust to spurious peaks: among all straddling pairs, choose the one
        whose spacing best matches the tracked lane width (with a small bonus
        for strong peaks). Only if no valid-width pair exists do we fall back
        to the single most prominent peak, labelled by which side it is on.
        """
        if not peaks:
            return None, None
        left = [p for p in peaks if p < self.center_px]
        right = [p for p in peaks if p >= self.center_px]

        best = None
        for lp in left:
            for rp in right:
                w = rp - lp
                if not (self.min_width <= w <= self.max_width):
                    continue
                # lower width error is better; tiny prominence tie-breaker
                score = -abs(w - self.lane_width_ema) + 1e-4 * (hs[lp] + hs[rp])
                if best is None or score > best[0]:
                    best = (score, lp, rp)
        if best is not None:
            return best[1], best[2]

        # no valid-width pair: keep only the single strongest peak
        allp = left + right
        strongest = max(allp, key=lambda p: hs[p])
        if strongest < self.center_px:
            return strongest, None
        return None, strongest

    def _build_model(self, header, dash, edge, hs, peaks):
        lane = LaneModel()
        lane.header = header
        lane.target_lane = "RIGHT"
        lane.curvature = 0.0  # curves are MPC's job (map waypoints); see node docstring
        lane.left_x = math.nan
        lane.middle_x = math.nan
        lane.right_x = math.nan

        both = dash is not None and edge is not None
        width = None
        target = None
        if both:
            width = float(edge - dash)
            target = 0.5 * (dash + edge)
            lane.middle_visible = True
            lane.right_visible = True
            lane.middle_x = float(dash)
            lane.right_x = float(edge)
        elif dash is not None:
            edge_est = dash + self.lane_width_ema
            # Reject if estimated edge is off-screen or left of centre — the
            # single peak is probably the right edge mislabelled as dash,
            # which would project the target outside the road.
            if self.center_px < edge_est < self.W * 0.95:
                target = 0.5 * (dash + edge_est)
                lane.middle_visible = True
                lane.middle_x = float(dash)
                lane.right_x = float(edge_est)
        elif edge is not None:
            dash_est = edge - self.lane_width_ema
            # Reject if estimated dash is off-screen or right of centre.
            if self.W * 0.05 < dash_est < self.center_px:
                target = 0.5 * (dash_est + edge)
                lane.right_visible = True
                lane.middle_x = float(dash_est)
                lane.right_x = float(edge)

        if target is None:
            self.target_ema = None
            lane.target_center_x = math.nan
            lane.error_px = math.nan
            lane.estimated_lane_width_px = float(self.lane_width_ema)
            lane.confidence = 0.0
            return lane

        # MEASURED confidence (uses the previous target_ema for stability scoring,
        # so compute it BEFORE updating the EMAs).
        confidence = self._quality_confidence(dash, edge, hs, peaks, both, width, target)

        # Update width EMA only on geometrically valid pairs.
        if both and self.min_width <= width <= self.max_width:
            self.lane_width_ema = (
                (1.0 - self.width_alpha) * self.lane_width_ema
                + self.width_alpha * width
            )
        # Update target EMA (temporal smoothing of the lateral target).
        self.target_ema = (
            target if self.target_ema is None
            else (1.0 - self.target_alpha) * self.target_ema + self.target_alpha * target
        )

        lane.target_center_x = float(self.target_ema)
        lane.right_lane_center_x = float(self.target_ema)
        lane.error_px = float(self.target_ema - self.center_px)
        lane.estimated_lane_width_px = float(self.lane_width_ema)
        lane.confidence = float(confidence)
        return lane

    def _quality_confidence(self, dash, edge, hs, peaks, both, width, target):
        """Confidence = base(both/single) scaled by four quality factors in [0,1]:
        peak prominence, lane-width consistency, peak ambiguity, temporal stability.
        A clean detection stays ~0.60; a weak/unstable/ill-shaped one drops below
        the planner's 0.30 fusion gate and is ignored. Honest confidence is what
        keeps the MPC from trusting bad frames.
        """
        hmax = float(hs.max()) if hs.size else 0.0
        ref = max(self.min_col_pixels, 0.5 * hmax + 1e-6)

        # 1. peak prominence: how strong the used peak(s) are vs a solid reference
        used = [p for p in (dash, edge) if p is not None]
        proms = [min(1.0, float(hs[int(p)]) / ref) for p in used]
        prominence = float(np.mean(proms)) if proms else 0.0

        # 2. lane-width consistency (only verifiable when both lines are seen)
        if both:
            if not (self.min_width <= width <= self.max_width):
                return 0.0  # geometry rejected outright
            width_score = float(np.exp(-abs(width - self.lane_width_ema) / self.width_tol_px))
        else:
            width_score = 0.85  # single line: spacing unverified, mild penalty

        # 3. ambiguity: >2 strong peaks => unsure which pair is the lane
        strong = sum(1 for p in peaks if float(hs[p]) >= 0.6 * hmax)
        ambiguity = 1.0 if strong <= 2 else max(0.4, 1.0 - 0.2 * (strong - 2))

        # 4. temporal stability: a big jump from the last target => low trust
        if self.target_ema is None:
            stability = 1.0
        else:
            jump = abs(target - self.target_ema)
            stability = float(np.clip(1.0 - jump / self.stability_jump_px, 0.3, 1.0))

        base = self.conf_both if both else self.conf_single
        return float(np.clip(base * prominence * width_score * ambiguity * stability, 0.0, 1.0))

    def _publish_debug(self, camera_bgr, bev_mask, dash, edge, lane):
        vis = cv2.cvtColor(bev_mask, cv2.COLOR_GRAY2BGR)
        h = vis.shape[0]

        def vline(x, color, label):
            if x is None or (isinstance(x, float) and not math.isfinite(x)):
                return
            xi = int(round(x))
            cv2.line(vis, (xi, 0), (xi, h), color, 2)
            cv2.putText(vis, label, (max(2, xi - 30), 28), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 1, cv2.LINE_AA)

        vline(self.center_px, (255, 255, 0), "car")
        vline(dash, (0, 255, 255), "dash")
        vline(edge, (0, 255, 0), "edge")
        if math.isfinite(lane.target_center_x):
            vline(lane.target_center_x, (255, 0, 255), "target")
        cv2.putText(
            vis,
            f"conf={lane.confidence:.2f} err="
            f"{'nan' if not math.isfinite(lane.error_px) else ('%.1f' % lane.error_px)}px",
            (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA,
        )

        camera_debug = camera_bgr.copy()
        pts = self.ipm_src.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(camera_debug, [pts], True, (0, 255, 0), 2)
        combined = np.hstack((camera_debug, vis))
        try:
            self.debug_pub.publish(bgr_to_image_msg(combined, header=lane.header))
        except (ValueError, cv2.error) as exc:
            self.get_logger().warn(f"debug conversion failed: {exc}", throttle_duration_sec=2.0)


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
