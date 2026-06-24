#!/usr/bin/env python3
"""Robust right-lane detector with PERSISTENT lane-identity tracking.

Perception half of the "MPC guides, vision centres" split:
    * mpc_reference_planner_node owns WHERE to go (waypoints, junction/roundabout).
    * This node owns staying CENTRED in the right lane on every section.

Instead of re-classifying lines from scratch each frame (which let the
right-edge line be mislabelled as the middle dash whenever the dash vanished),
the detector keeps two TRACKED identities — `middle` (dashed centre) and
`right` (solid edge) — and:

  1. Seeds each frame's sliding-window search from the line's LAST tracked x
     (not from an image-centre split). Identity therefore persists.
  2. Associates a new detection to a track only if it lies within a distance
     GATE of the track's last position. A far detection cannot capture a track.
  3. Enforces that `middle` and `right` are DISTINCT lines: the middle must sit
     a plausible lane-width to the LEFT of the right edge with dark road between
     them. A single wide white line can never become both.
  4. Handles lost tracks explicitly: if a line is unseen for max_missing_frames
     it is marked lost (stays missing) rather than being replaced by the other.

Pipeline: camera -> HSV white mask -> morphological clean -> IPM warp ->
per-identity sliding-window + polynomial fit -> associate/gate -> edge-anchored
right-lane centre -> LaneModel.

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
        # BEV mask cleanup. This rejects small flecks plus horizontal/compact
        # road markings before they can seed the lane identity tracker.
        self.declare_parameter("component_filter_enabled", False)
        self.declare_parameter("min_component_area_px", 60)
        self.declare_parameter("max_component_area_px", 50000)
        self.declare_parameter("min_component_height_px", 18)
        self.declare_parameter("max_component_width_px", 170)
        self.declare_parameter("min_component_aspect", 1.20)
        self.declare_parameter("min_filtered_pixels", 250)
        self.declare_parameter("min_filtered_pixel_ratio", 0.08)
        # sliding window
        self.declare_parameter("n_windows",          10)
        self.declare_parameter("window_half_w_px",   80)
        self.declare_parameter("min_pix_per_window", 30)
        self.declare_parameter("hist_band_frac",     0.50)
        self.declare_parameter("eval_y_frac",        0.80)
        self.declare_parameter("min_line_span_frac", 0.18)
        self.declare_parameter("max_line_rms_px", 42.0)
        # lane geometry
        self.declare_parameter("nominal_lane_width_px", 394.0)
        self.declare_parameter("min_lane_width_px",     220.0)
        self.declare_parameter("max_lane_width_px",     540.0)
        self.declare_parameter("max_target_jump_px", 80.0)
        # tracking / identity
        self.declare_parameter("assoc_gate_px",       90.0)   # max jump to keep identity
        self.declare_parameter("max_missing_frames",  8)      # frames before a track is lost
        self.declare_parameter("track_x_alpha",       0.45)   # per-track position smoothing
        self.declare_parameter("target_ema_alpha",    0.30)
        self.declare_parameter("width_ema_alpha",     0.10)
        self.declare_parameter("confidence_both",     0.70)
        self.declare_parameter("confidence_single",   0.50)

        self.W = int(self.get_parameter("image_width").value)
        self.H = int(self.get_parameter("image_height").value)
        self.center_px  = 0.5 * self.W
        self.debug_en   = bool(self.get_parameter("debug_enabled").value)
        self.hsv_lo     = np.array(self.get_parameter("hsv_lo").value, dtype=np.uint8)
        self.hsv_hi     = np.array(self.get_parameter("hsv_hi").value, dtype=np.uint8)
        k = max(1, int(self.get_parameter("morph_kernel").value))
        self.morph_k    = np.ones((k, k), np.uint8)
        self.comp_filter = bool(self.get_parameter("component_filter_enabled").value)
        self.min_comp_area = int(self.get_parameter("min_component_area_px").value)
        self.max_comp_area = int(self.get_parameter("max_component_area_px").value)
        self.min_comp_h = int(self.get_parameter("min_component_height_px").value)
        self.max_comp_w = int(self.get_parameter("max_component_width_px").value)
        self.min_comp_aspect = float(self.get_parameter("min_component_aspect").value)
        self.min_filtered_pixels = int(self.get_parameter("min_filtered_pixels").value)
        self.min_filtered_ratio = float(
            self.get_parameter("min_filtered_pixel_ratio").value
        )
        self.n_windows  = int(self.get_parameter("n_windows").value)
        self.win_hw     = int(self.get_parameter("window_half_w_px").value)
        self.min_pix    = int(self.get_parameter("min_pix_per_window").value)
        self.band_frac  = float(self.get_parameter("hist_band_frac").value)
        self.eval_y_fr  = float(self.get_parameter("eval_y_frac").value)
        self.min_line_span_frac = float(self.get_parameter("min_line_span_frac").value)
        self.max_line_rms_px = float(self.get_parameter("max_line_rms_px").value)
        self.nom_width  = float(self.get_parameter("nominal_lane_width_px").value)
        self.min_width  = float(self.get_parameter("min_lane_width_px").value)
        self.max_width  = float(self.get_parameter("max_lane_width_px").value)
        self.max_target_jump = float(self.get_parameter("max_target_jump_px").value)
        self.assoc_gate = float(self.get_parameter("assoc_gate_px").value)
        self.max_missing= int(self.get_parameter("max_missing_frames").value)
        self.track_alpha= float(self.get_parameter("track_x_alpha").value)
        self.tgt_alpha  = float(self.get_parameter("target_ema_alpha").value)
        self.w_alpha    = float(self.get_parameter("width_ema_alpha").value)
        self.conf_both  = float(self.get_parameter("confidence_both").value)
        self.conf_single= float(self.get_parameter("confidence_single").value)

        src = np.float32([[rx*self.W, ry*self.H] for rx, ry in IPM_SRC_RATIOS])
        dst = np.float32([[rx*self.W, ry*self.H] for rx, ry in IPM_DST_RATIOS])
        self.M = cv2.getPerspectiveTransform(src, dst)
        self.ipm_src = src

        self.eval_y = int(self.H * self.eval_y_fr)
        self.lane_width_ema = self.nom_width
        self.target_ema     = None
        # persistent lane identities
        self.track = {
            "middle": {"x": None, "poly": None, "n": 0, "miss": self.max_missing + 1},
            "right":  {"x": None, "poly": None, "n": 0, "miss": self.max_missing + 1},
        }

        self.lane_pub  = self.create_publisher(LaneModel, self.get_parameter("lane_model_topic").value, 10)
        self.debug_pub = self.create_publisher(Image,     self.get_parameter("debug_topic").value,      10)
        self.create_subscription(Image, self.get_parameter("image_topic").value, self._on_image, 10)
        self.get_logger().info("Identity-tracking HSV+BEV lane detector ready")

    # ------------------------------------------------------------------ #
    def _on_image(self, msg: Image):
        try:
            bgr = image_msg_to_bgr(msg)
        except Exception as exc:
            self.get_logger().warn(f"image decode: {exc}", throttle_duration_sec=2.0)
            return
        if bgr.shape[1] != self.W or bgr.shape[0] != self.H:
            bgr = cv2.resize(bgr, (self.W, self.H))

        mask = cv2.inRange(cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV), self.hsv_lo, self.hsv_hi)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.morph_k)
        bev = cv2.warpPerspective(mask, self.M, (self.W, self.H), flags=cv2.INTER_LINEAR)
        if self.comp_filter:
            filtered = self._filter_components(bev)
            raw_px = int(np.count_nonzero(bev))
            kept_px = int(np.count_nonzero(filtered))
            min_px = max(
                self.min_filtered_pixels,
                int(self.min_filtered_ratio * max(1, raw_px)),
            )
            if raw_px > 0 and kept_px < min_px:
                self.get_logger().warn(
                    f"BEV component filter rejected too much "
                    f"({kept_px}/{raw_px}px); using raw mask",
                    throttle_duration_sec=2.0,
                )
            else:
                bev = filtered

        self._update_tracks(bev)
        lane = self._build_model(msg.header)
        self.lane_pub.publish(lane)

        tr_m, tr_r = self.track["middle"], self.track["right"]
        self.get_logger().info(
            f"middle={'--' if tr_m['x'] is None else int(tr_m['x'])}(miss{tr_m['miss']}) "
            f"right={'--' if tr_r['x'] is None else int(tr_r['x'])}(miss{tr_r['miss']}) "
            f"err={'nan' if not math.isfinite(lane.error_px) else f'{lane.error_px:.1f}'}px "
            f"conf={lane.confidence:.2f} width={self.lane_width_ema:.0f}px",
            throttle_duration_sec=0.5,
        )
        if self.debug_en:
            self._publish_debug(bgr, bev, lane)

    # ------------------------------------------------------------------ #
    def _filter_components(self, bev):
        n, labels, stats, _ = cv2.connectedComponentsWithStats(
            (bev > 0).astype(np.uint8), connectivity=8
        )
        out = np.zeros_like(bev)
        for cid in range(1, n):
            x, y, w, h, area = stats[cid]
            if area < self.min_comp_area or area > self.max_comp_area:
                continue
            if h < self.min_comp_h or w > self.max_comp_w:
                continue
            aspect = h / float(max(1, w))
            if aspect < self.min_comp_aspect:
                continue
            out[labels == cid] = 255
        return out

    # ------------------------------------------------------------------ #
    def _histogram_peaks(self, bev):
        """All prominent white-column peaks in the near band (>=40px apart)."""
        band0 = int(self.H * self.band_frac)
        hist = bev[band0:, :].sum(axis=0).astype(np.float32)
        hist = np.convolve(hist, np.ones(21, np.float32) / 21.0, mode="same")
        thr = max(hist.max() * 0.30, 255.0 * 6)
        peaks = []
        for x in range(2, self.W - 2):
            if hist[x] >= thr and hist[x] >= hist[x - 1] and hist[x] >= hist[x + 1]:
                if not peaks or x - peaks[-1] > 40:
                    peaks.append(x)
                elif hist[x] > hist[peaks[-1]]:
                    peaks[-1] = x
        return peaks, hist

    def _track_line(self, bev, base_x):
        """Slide windows bottom->top from base_x; return (eval_x, poly, n, pts)."""
        h, w = bev.shape
        band0 = int(h * self.band_frac)
        win_h = max(1, (h - band0) // self.n_windows)
        cur = int(base_x)
        xs, ys = [], []
        for i in range(self.n_windows):
            y_hi = h - i * win_h
            y_lo = max(band0, h - (i + 1) * win_h)
            if y_lo >= y_hi:
                continue
            x0 = max(0, cur - self.win_hw); x1 = min(w, cur + self.win_hw)
            nz = np.nonzero(bev[y_lo:y_hi, x0:x1])
            if len(nz[1]) >= self.min_pix:
                cx = int(np.mean(nz[1])) + x0
                xs.append(cx); ys.append((y_lo + y_hi) // 2)
                cur = cx
        if len(ys) < 3:
            return None, None, len(ys), None
        try:
            poly = np.polyfit(ys, xs, 2)
        except Exception:
            return None, None, len(ys), None
        if not self._line_fit_ok(ys, xs, poly):
            return None, None, len(ys), None
        eval_x = float(np.polyval(poly, self.eval_y))
        pts = np.column_stack([ys, xs])
        return eval_x, poly, len(ys), pts

    def _line_fit_ok(self, ys, xs, poly):
        ys = np.asarray(ys, dtype=np.float32)
        xs = np.asarray(xs, dtype=np.float32)
        if ys.size < 3:
            return False
        y_span = float(ys.max() - ys.min())
        if y_span < max(6.0, self.min_line_span_frac * self.H):
            return False
        pred = np.polyval(poly, ys)
        rms = float(np.sqrt(np.mean((xs - pred) ** 2)))
        return rms <= self.max_line_rms_px

    def _has_dark_gap(self, bev, x1, x2, min_dark_frac=0.40):
        lo, hi = int(min(x1, x2)), int(max(x1, x2))
        if hi - lo < 10:
            return False
        y0 = max(0, self.eval_y - 8); y1 = min(bev.shape[0], self.eval_y + 9)
        band = bev[y0:y1, lo:hi]
        return band.size > 0 and float((band == 0).mean()) >= min_dark_frac

    # ------------------------------------------------------------------ #
    def _update_tracks(self, bev):
        """Seed from last tracked positions, detect, associate with gating,
        enforce distinctness, and age out lost tracks."""
        peaks, _ = self._histogram_peaks(bev)
        tr_m, tr_r = self.track["middle"], self.track["right"]

        # --- RIGHT edge (anchor): seed from its track, else rightmost peak ---
        if tr_r["x"] is not None:
            right_seed = tr_r["x"]
        else:
            right_cands = [p for p in peaks if p >= self.center_px] or peaks
            right_seed = max(right_cands) if right_cands else int(self.center_px + 0.25 * self.W)
        rx, rpoly, rn, _ = self._track_line(bev, right_seed)
        # plausible-half guard
        if rx is not None and not (0.10 * self.W < rx < self.W * 0.99):
            rx = None
        self._associate("right", rx, rpoly, rn)

        # --- MIDDLE dash: seed from its track, else a peak that is a valid lane
        #     width LEFT of the right edge (so it can never seed onto the edge) ---
        if tr_m["x"] is not None:
            middle_seed = tr_m["x"]
        else:
            anchor = tr_r["x"]
            middle_seed = None
            if anchor is not None:
                # candidate peaks left of the edge by a plausible lane width
                cands = [p for p in peaks
                         if self.min_width <= (anchor - p) <= self.max_width]
                if cands:
                    middle_seed = max(cands)          # nearest valid dash left of edge
            else:
                left_cands = [p for p in peaks if p < self.center_px]
                if left_cands:
                    middle_seed = max(left_cands)
        if middle_seed is None:
            self._associate("middle", None, None, 0)
        else:
            mx, mpoly, mn, _ = self._track_line(bev, middle_seed)
            # plausible-half guard
            if mx is not None and not (self.W * 0.02 < mx < self.center_px + 0.20 * self.W):
                mx = None
            # distinctness vs the right edge: real lane width + dark road between
            if mx is not None and tr_r["x"] is not None:
                if not (self.min_width <= (tr_r["x"] - mx) <= self.max_width
                        and self._has_dark_gap(bev, mx, tr_r["x"])):
                    mx = None     # same blob as the edge -> not a distinct middle
            self._associate("middle", mx, mpoly, mn)

    def _associate(self, name, det_x, det_poly, det_n):
        """Update a track only if det_x is within the gate; else age it out."""
        tr = self.track[name]
        if det_x is not None and (
            tr["x"] is None or abs(det_x - tr["x"]) <= self.assoc_gate
        ):
            tr["x"] = det_x if tr["x"] is None else (
                (1.0 - self.track_alpha) * tr["x"] + self.track_alpha * det_x
            )
            tr["poly"] = det_poly
            tr["n"] = det_n
            tr["miss"] = 0
        else:
            tr["miss"] += 1
            if tr["miss"] > self.max_missing:
                tr["x"] = None
                tr["poly"] = None
                tr["n"] = 0

    # ------------------------------------------------------------------ #
    def _build_model(self, header):
        lane = LaneModel()
        lane.header      = header
        lane.target_lane = "RIGHT"
        lane.curvature   = 0.0
        lane.left_x = lane.middle_x = lane.right_x = math.nan

        tr_m, tr_r = self.track["middle"], self.track["right"]
        edge_x = tr_r["x"]
        dash_x = tr_m["x"]

        target = None
        confidence = 0.0
        half_w = 0.5 * self.lane_width_ema

        if edge_x is not None:
            cov = tr_r["n"] / float(self.n_windows)
            base = self.conf_single
            if dash_x is not None:                       # both identities alive & distinct
                width = edge_x - dash_x
                if self.min_width <= width <= self.max_width:
                    self.lane_width_ema = (
                        (1.0 - self.w_alpha) * self.lane_width_ema + self.w_alpha * width
                    )
                    half_w = 0.5 * self.lane_width_ema
                    base = self.conf_both
                    cov = (tr_m["n"] + tr_r["n"]) / (2.0 * self.n_windows)
                    lane.middle_visible = True;  lane.middle_x = float(dash_x)
            target = edge_x - half_w                      # anchor to the solid edge
            confidence = base * min(1.0, cov * 1.5)
            lane.right_visible = True;  lane.right_x = float(edge_x)

        elif dash_x is not None:                          # only the dash (no edge anchor)
            edge_est = dash_x + self.lane_width_ema
            if self.center_px < edge_est < self.W * 0.96:
                target = dash_x + half_w
                cov = tr_m["n"] / float(self.n_windows)
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

        if (
            self.target_ema is not None
            and self.max_target_jump > 0.0
            and abs(target - self.target_ema) > self.max_target_jump
        ):
            lane.target_center_x = float(self.target_ema)
            lane.right_lane_center_x = float(self.target_ema)
            lane.error_px = float(self.target_ema - self.center_px)
            lane.estimated_lane_width_px = float(self.lane_width_ema)
            lane.confidence = float(min(confidence, 0.10))
            return lane

        self.target_ema = (
            target if self.target_ema is None
            else (1.0 - self.tgt_alpha) * self.target_ema + self.tgt_alpha * target
        )
        lane.target_center_x     = float(self.target_ema)
        lane.right_lane_center_x  = float(self.target_ema)
        lane.error_px            = float(self.target_ema - self.center_px)
        lane.estimated_lane_width_px = float(self.lane_width_ema)
        lane.confidence          = float(confidence)
        return lane

    # ------------------------------------------------------------------ #
    def _publish_debug(self, camera_bgr, bev, lane):
        vis = cv2.cvtColor(bev, cv2.COLOR_GRAY2BGR)
        h, w = vis.shape[:2]
        ys = np.linspace(int(h * self.band_frac), h - 1, 60).astype(int)

        for name, color in [("middle", (0, 220, 220)), ("right", (0, 220, 0))]:
            poly = self.track[name]["poly"]
            if poly is not None:
                xs = np.polyval(poly, ys).astype(int)
                for j in range(len(ys) - 1):
                    if 0 <= xs[j] < w and 0 <= xs[j + 1] < w:
                        cv2.line(vis, (xs[j], ys[j]), (xs[j + 1], ys[j + 1]), color, 2)
            tx = self.track[name]["x"]
            if tx is not None:
                xi = int(round(tx))
                lbl = f"{name}{'' if self.track[name]['miss'] == 0 else '?'}"
                cv2.line(vis, (xi, 0), (xi, h), color, 2)
                cv2.putText(vis, lbl, (max(2, xi - 28), 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

        def vline(x, color, label):
            if x is None or (isinstance(x, float) and not math.isfinite(x)):
                return
            xi = int(round(x))
            cv2.line(vis, (xi, 0), (xi, h), color, 2)
            cv2.putText(vis, label, (max(2, xi - 28), 44),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

        vline(self.center_px, (255, 255, 0), "car")
        if math.isfinite(lane.target_center_x):
            vline(lane.target_center_x, (255, 0, 255), "target")
        cv2.line(vis, (0, self.eval_y), (w, self.eval_y), (100, 100, 100), 1)
        cv2.putText(vis,
            f"conf={lane.confidence:.2f} err="
            f"{'nan' if not math.isfinite(lane.error_px) else f'{lane.error_px:.1f}'}px",
            (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

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
