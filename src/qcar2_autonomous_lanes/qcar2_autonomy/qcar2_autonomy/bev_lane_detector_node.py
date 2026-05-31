#!/usr/bin/env python3
"""Phase 1 camera lane detector for QCar2 lane targets."""
import math

import rclpy
from qcar2_msgs.msg import BehaviorState, LaneModel
from rclpy.node import Node
from sensor_msgs.msg import Image

import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError


HISTORY_WEIGHT = 30.0
CONTINUITY_SIGMA_PX = 40.0
MAX_FRAME_JUMP_PX = 80.0
WIDTH_PRIOR_WEIGHT = 5.0
WIDTH_BALANCE_WEIGHT = 7.0
TARGET_CENTER_WEIGHT = 4.0
LEFT = 'LEFT'
RIGHT = 'RIGHT'
STOP = 'STOP'


class BeVLaneDetectorNode(Node):
    def __init__(self):
        super().__init__('bev_lane_detector_node')

        self.declare_parameter('image_topic', '/qcar2/front_camera/image')
        self.declare_parameter('behavior_state_topic', '/qcar2/behavior/state')
        self.declare_parameter('lane_model_topic', '/qcar2/lane/model')
        self.declare_parameter('debug_topic', '/qcar2/lane/debug_image')
        self.declare_parameter('hsv_lo', [0, 0, 180])
        self.declare_parameter('hsv_hi', [180, 80, 255])
        self.declare_parameter('roi_y_start', 240)
        self.declare_parameter('roi_y_end', 480)
        self.declare_parameter('min_pixels_per_line', 40)
        self.declare_parameter('debug_enabled', True)
        self.declare_parameter('estimated_lane_width_px', 608.0)
        self.declare_parameter('max_missing_frames', 45)
        self.declare_parameter('alpha_smoothing', 0.65)
        self.declare_parameter('confidence_threshold', 0.2)
        self.declare_parameter('min_lane_width_px', 200.0)
        self.declare_parameter('max_lane_width_px', 800.0)
        self.declare_parameter('peak_merge_distance_px', 25.0)
        self.declare_parameter('histogram_smoothing_kernel', 9)
        self.declare_parameter('lane_change_smoothing_alpha', 0.85)
        self.declare_parameter('max_target_shift_px_per_frame', 35.0)
        self.declare_parameter('lane_change_min_confidence', 0.45)
        self.declare_parameter('lane_change_duration_sec', 3.0)

        self.hsv_lo = np.array(self.get_parameter('hsv_lo').value, dtype=np.uint8)
        self.hsv_hi = np.array(self.get_parameter('hsv_hi').value, dtype=np.uint8)
        self.roi_y_start = int(self.get_parameter('roi_y_start').value)
        self.roi_y_end = int(self.get_parameter('roi_y_end').value)
        self.min_pixels_per_line = int(self.get_parameter('min_pixels_per_line').value)
        self.debug_enabled = bool(self.get_parameter('debug_enabled').value)
        self.estimated_lane_width_px = float(
            self.get_parameter('estimated_lane_width_px').value
        )
        self.max_missing_frames = int(self.get_parameter('max_missing_frames').value)
        self.alpha_smoothing = float(self.get_parameter('alpha_smoothing').value)
        self.confidence_threshold = float(self.get_parameter('confidence_threshold').value)
        self.min_lane_width_px = float(self.get_parameter('min_lane_width_px').value)
        self.max_lane_width_px = float(self.get_parameter('max_lane_width_px').value)
        self.peak_merge_distance_px = float(
            self.get_parameter('peak_merge_distance_px').value
        )
        self.histogram_smoothing_kernel = max(
            1, int(self.get_parameter('histogram_smoothing_kernel').value)
        )
        if self.histogram_smoothing_kernel % 2 == 0:
            self.histogram_smoothing_kernel += 1
        self.lane_change_smoothing_alpha = min(
            0.99,
            max(0.0, float(self.get_parameter('lane_change_smoothing_alpha').value)),
        )
        self.max_target_shift_px_per_frame = max(
            1.0,
            float(self.get_parameter('max_target_shift_px_per_frame').value),
        )
        self.lane_change_min_confidence = min(
            1.0,
            max(0.0, float(self.get_parameter('lane_change_min_confidence').value)),
        )
        self.lane_change_duration_sec = max(
            0.5,
            float(self.get_parameter('lane_change_duration_sec').value),
        )

        self.last_left_x = None
        self.last_middle_x = None
        self.last_right_x = None
        self.active_target_center_x = None
        self.desired_target_center_x = None
        self.requested_behavior_state = 'KEEP_RIGHT'
        self.transition_state = None
        self.transition_start_time_sec = None
        self.transition_start_center_x = None
        self.previous_target_center_x = None
        self.previous_target_lane = RIGHT
        self.requested_target_lane = RIGHT
        self.missing_left_count = 0
        self.missing_middle_count = 0
        self.missing_right_count = 0

        self.bridge = CvBridge()
        self.lane_pub = self.create_publisher(
            LaneModel,
            self.get_parameter('lane_model_topic').value,
            10,
        )
        self.debug_pub = self.create_publisher(
            Image,
            self.get_parameter('debug_topic').value,
            10,
        )
        self.create_subscription(
            Image,
            self.get_parameter('image_topic').value,
            self._on_image,
            10,
        )
        self.create_subscription(
            BehaviorState,
            self.get_parameter('behavior_state_topic').value,
            self._on_behavior,
            10,
        )

        self.get_logger().info('Phase 1 BEV lane detector ready')

    def _on_behavior(self, msg: BehaviorState):
        requested = msg.target_lane.strip().upper()
        state = msg.state.strip().upper()
        self.requested_behavior_state = state
        if msg.emergency_stop or requested == STOP or state == 'EMERGENCY_STOP':
            new_lane = STOP
        elif requested == LEFT:
            new_lane = LEFT
        else:
            new_lane = RIGHT
        # When the requested lane flips, the camera perspective shifts dramatically
        # and the old per-line history (last_left_x / last_middle_x / last_right_x)
        # becomes stale — the classifier's history bonus then keeps labeling new
        # peaks as if they were the old ones, which is what produced the "left
        # treated as middle" and "right both middle and right" misclassifications
        # observed during CHANGE_LEFT / RETURN_RIGHT. Flush memory on the change.
        if new_lane != self.requested_target_lane and new_lane in (LEFT, RIGHT):
            self.last_left_x = None
            self.last_middle_x = None
            self.last_right_x = None
            self.missing_left_count = 0
            self.missing_middle_count = 0
            self.missing_right_count = 0
        self.requested_target_lane = new_lane

    def _on_image(self, msg: Image):
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError as exc:
            self.get_logger().warn(
                f'cv_bridge image conversion failed: {exc}',
                throttle_duration_sec=2.0,
            )
            return

        height, width = bgr.shape[:2]
        y0 = max(0, min(height - 1, self.roi_y_start))
        y1 = max(y0 + 1, min(height, self.roi_y_end))
        roi = bgr[y0:y1, :]

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.hsv_lo, self.hsv_hi)

        kernel = np.ones((3, 3), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        histogram = np.sum(mask > 0, axis=0).astype(np.float32)
        raw_peaks = self._detect_peaks(histogram)
        peaks = self._merge_close_peaks(raw_peaks)

        lane, debug_state = self._build_lane_model(msg.header, peaks, width)
        self.lane_pub.publish(lane)

        if self.debug_enabled:
            self._publish_debug(bgr, mask, lane, peaks, raw_peaks, y0, y1, debug_state)

    def _detect_peaks(self, histogram):
        """Return contiguous bright bands as {'x', 'score'}, sorted by x."""
        if histogram.size == 0:
            return []

        k = self.histogram_smoothing_kernel
        smoothing = np.ones(k, dtype=np.float32) / float(k)
        smooth = np.convolve(histogram, smoothing, mode='same')
        active = smooth >= max(1, self.min_pixels_per_line)

        bands = []
        start = None
        for idx, is_active in enumerate(active):
            if is_active and start is None:
                start = idx
            elif not is_active and start is not None:
                self._collect_band(bands, histogram, start, idx - 1)
                start = None
        if start is not None:
            self._collect_band(bands, histogram, start, len(active) - 1)

        return sorted(bands, key=lambda b: b['x'])

    def _collect_band(self, bands, histogram, start, end):
        weights = histogram[start:end + 1]
        score = float(np.sum(weights))
        if score < self.min_pixels_per_line:
            return
        xs = np.arange(start, end + 1, dtype=np.float32)
        bands.append({'x': float(np.average(xs, weights=weights)), 'score': score})

    def _merge_close_peaks(self, peaks):
        """Merge adjacent peaks whose centers fall within peak_merge_distance_px."""
        if not peaks:
            return []
        merged = [dict(peaks[0])]
        for peak in peaks[1:]:
            last = merged[-1]
            if peak['x'] - last['x'] <= self.peak_merge_distance_px:
                total = last['score'] + peak['score']
                last['x'] = (last['x'] * last['score'] + peak['x'] * peak['score']) / total
                last['score'] = total
            else:
                merged.append(dict(peak))
        return merged

    def _build_lane_model(self, header, peaks, image_width):
        lane = LaneModel()
        lane.header = header
        lane.left_visible = False
        lane.middle_visible = False
        lane.right_visible = False
        lane.left_x = math.nan
        lane.middle_x = math.nan
        lane.right_x = math.nan
        lane.left_lane_center_x = math.nan
        lane.right_lane_center_x = math.nan
        lane.target_center_x = math.nan
        lane.error_px = math.nan
        lane.confidence = 0.0
        lane.curvature = 0.0
        lane.estimated_lane_width_px = math.nan
        requested_target_lane = self.requested_target_lane
        lane.target_lane = requested_target_lane

        left_x, middle_x, right_x, selected_indices, rejected_indices = self._select_lines(
            peaks,
            image_width,
            requested_target_lane,
        )
        # Single-frame outlier rejection: if the newly-assigned x is far from
        # the last-known position AND last position is fresh (<= max_missing),
        # ignore the new value and let the memory recovery step below restore
        # the stable label. This kills the M-oscillation we saw alternating
        # between the real dashed middle and a noise peak ~60 px away.
        left_x = self._reject_jump(left_x, self.last_left_x, self.missing_left_count)
        middle_x = self._reject_jump(middle_x, self.last_middle_x, self.missing_middle_count)
        right_x = self._reject_jump(right_x, self.last_right_x, self.missing_right_count)
        left_status = 'detected' if self._is_finite(left_x) else 'lost'
        middle_status = 'detected' if self._is_finite(middle_x) else 'lost'
        right_status = 'detected' if self._is_finite(right_x) else 'lost'
        predicted_target = False

        if self._is_finite(left_x):
            self.last_left_x = left_x
            self.missing_left_count = 0
        else:
            self.missing_left_count += 1

        if self._is_finite(middle_x):
            self.last_middle_x = middle_x
            self.missing_middle_count = 0
        else:
            self.missing_middle_count += 1

        if self._is_finite(right_x):
            self.last_right_x = right_x
            self.missing_right_count = 0
        else:
            self.missing_right_count += 1

        measured_widths = []
        if self._is_finite(left_x) and self._is_finite(middle_x):
            measured_widths.append(middle_x - left_x)
        if self._is_finite(middle_x) and self._is_finite(right_x):
            measured_widths.append(right_x - middle_x)
        valid_widths = [
            width for width in measured_widths
            if self.min_lane_width_px <= width <= self.max_lane_width_px
        ]
        if valid_widths:
            measured_width = float(np.mean(valid_widths))
            self.estimated_lane_width_px = (
                self.alpha_smoothing * self.estimated_lane_width_px
                + (1.0 - self.alpha_smoothing) * measured_width
            )

        # Prefer temporal memory of each line's last known position over
        # cross-line synthesis. We also REQUIRE the restored value to keep the
        # left < middle < right ordering — otherwise stale memory of a
        # previous-position line can be placed to the wrong side of a freshly
        # detected line, which is the "left mistaken for middle" failure
        # observed when the dashed middle was intermittent.
        def _consistent(candidate, side):
            if side == 'left':
                if self._is_finite(middle_x) and candidate >= middle_x:
                    return False
                if self._is_finite(right_x) and candidate >= right_x:
                    return False
            elif side == 'middle':
                if self._is_finite(left_x) and candidate <= left_x:
                    return False
                if self._is_finite(right_x) and candidate >= right_x:
                    return False
            elif side == 'right':
                if self._is_finite(middle_x) and candidate <= middle_x:
                    return False
                if self._is_finite(left_x) and candidate <= left_x:
                    return False
            return True

        if (
            not self._is_finite(middle_x)
            and self.last_middle_x is not None
            and self.missing_middle_count <= self.max_missing_frames
            and _consistent(self.last_middle_x, 'middle')
        ):
            middle_x = self.last_middle_x
            middle_status = 'memory'
        if (
            not self._is_finite(right_x)
            and self.last_right_x is not None
            and self.missing_right_count <= self.max_missing_frames
            and _consistent(self.last_right_x, 'right')
        ):
            right_x = self.last_right_x
            right_status = 'memory'
        if (
            not self._is_finite(left_x)
            and self.last_left_x is not None
            and self.missing_left_count <= self.max_missing_frames
            and _consistent(self.last_left_x, 'left')
        ):
            left_x = self.last_left_x
            left_status = 'memory'

        # Cross-line synthesis is the last resort and only fires from a
        # REAL detection (never from another synthesized value) for the line
        # the current target lane actually depends on. This prevents the
        # phantom-line chains we observed where M synthesized from R, then
        # L synthesized from synthesized M, produced targets at x=-617.
        if requested_target_lane == RIGHT:
            if not self._is_finite(middle_x) and right_status == 'detected':
                candidate = right_x - self.estimated_lane_width_px
                if self._line_in_image(candidate, image_width):
                    middle_x = candidate
                    middle_status = 'estimated'
            elif not self._is_finite(right_x) and middle_status == 'detected':
                candidate = middle_x + self.estimated_lane_width_px
                if self._line_in_image(candidate, image_width):
                    right_x = candidate
                    right_status = 'estimated'
        elif requested_target_lane == LEFT:
            if not self._is_finite(left_x) and middle_status == 'detected':
                candidate = middle_x - self.estimated_lane_width_px
                if self._line_in_image(candidate, image_width):
                    left_x = candidate
                    left_status = 'estimated'
            elif not self._is_finite(middle_x) and left_status == 'detected':
                candidate = left_x + self.estimated_lane_width_px
                if self._line_in_image(candidate, image_width):
                    middle_x = candidate
                    middle_status = 'estimated'

        if self._is_finite(left_x):
            lane.left_visible = True
            lane.left_x = float(left_x)

        if self._is_finite(middle_x):
            lane.middle_visible = True
            lane.middle_x = float(middle_x)

        if self._is_finite(right_x):
            lane.right_visible = True
            lane.right_x = float(right_x)

        if (
            lane.left_visible
            and lane.middle_visible
            and self._valid_lane_width(lane.middle_x - lane.left_x)
        ):
            lane.left_lane_center_x = 0.5 * (lane.left_x + lane.middle_x)

        if (
            lane.middle_visible
            and lane.right_visible
            and self._valid_lane_width(lane.right_x - lane.middle_x)
        ):
            lane.right_lane_center_x = 0.5 * (lane.middle_x + lane.right_x)

        lane.estimated_lane_width_px = float(self.estimated_lane_width_px)

        desired_target = math.nan
        if requested_target_lane == STOP:
            lane.confidence = 0.0
            desired_target = self._current_active_target()
        else:
            desired_target, lane.confidence, predicted_target = self._choose_target(
                lane,
                requested_target_lane,
                left_status,
                middle_status,
                right_status,
                image_width,
            )
            if self._is_finite(desired_target) and not predicted_target:
                lane.confidence = max(lane.confidence, self.lane_change_min_confidence)
            if not self._is_finite(desired_target):
                self.desired_target_center_x = None
                self.previous_target_center_x = None

        if self._is_finite(desired_target):
            active_target = self._advance_active_target(
                desired_target,
                image_width,
                freeze=requested_target_lane == STOP,
                initial_target=self._initial_active_target(lane, image_width),
                transition_state=self.requested_behavior_state,
            )
            lane.target_center_x = float(active_target)
            lane.error_px = float(active_target - 0.5 * image_width)
            if requested_target_lane != STOP:
                self.previous_target_center_x = active_target
                self.previous_target_lane = requested_target_lane

        debug_state = {
            'left_status': left_status,
            'middle_status': middle_status if not predicted_target else 'predicted',
            'right_status': right_status if not predicted_target else 'predicted',
            'predicted_target': predicted_target,
            'selected_indices': selected_indices,
            'rejected_indices': rejected_indices,
            'requested_target_lane': requested_target_lane,
            'desired_target_center_x': self.desired_target_center_x,
            'active_target_center_x': self.active_target_center_x,
        }

        return lane, debug_state

    def _choose_target(
        self,
        lane,
        target_lane,
        left_status,
        middle_status,
        right_status,
        image_width,
    ):
        # 'memory' = the line wasn't detected this frame but its last-known
        # position is recent (within max_missing_frames). That's a real
        # observation just slightly stale — treat it like 'detected' for
        # confidence purposes, distinct from cross-line 'estimated' synthesis
        # which is much less reliable.
        observed = ('detected', 'memory')
        if target_lane == LEFT:
            # Prefer a real visual lock on the left lane center.
            real_left_lock = (
                math.isfinite(lane.left_lane_center_x)
                and left_status in observed
                and middle_status in observed
            )
            if real_left_lock:
                if left_status == 'detected' and middle_status == 'detected':
                    confidence = 1.0
                else:
                    confidence = 0.8  # at least one from memory
                return self._clamp_target(lane.left_lane_center_x, image_width), confidence, False
            # Blind-plan offset: shift one lane width left of the visible right lane.
            # Lets the car commit to the change before the left line enters the FOV.
            # Accept right_status of detected OR estimated — we only need *some*
            # right-lane reference to plan the cross-track shift. Bound the target
            # so the initial error doesn't drive the controller into a max-rate pivot.
            if math.isfinite(lane.right_lane_center_x):
                # Gentle left bias rather than a full lane-width offset. A full
                # lane-width target lands well off the left edge of the image and
                # drives the controller into a max-rate left turn — the car
                # overshoots into the left curb before the FSM can call RETURN.
                # A modest target (around image_center − 100 px) gives a smooth
                # arc that physically reaches the left lane over a few seconds.
                gentle_offset = 0.18 * image_width
                planned = (0.5 * image_width) - gentle_offset
                # Also pull a little toward the cross-track interpretation of the
                # visible right lane so the target tracks the road geometry as
                # the car turns (right_lane_center_x shifts in image as we arc).
                aim = lane.right_lane_center_x - 0.5 * self.estimated_lane_width_px
                planned = 0.5 * (planned + aim)
                planned = self._clamp_target(planned, image_width)
                conf = 0.6 if right_status == 'detected' else 0.4
                return planned, conf, False
            if middle_status in observed and math.isfinite(lane.middle_x):
                planned = lane.middle_x - 0.5 * self.estimated_lane_width_px
                confidence = 0.45 if middle_status == 'detected' else 0.35
                return self._clamp_target(planned, image_width), confidence, True
            if left_status in observed and math.isfinite(lane.left_x):
                planned = lane.left_x + 0.5 * self.estimated_lane_width_px
                confidence = 0.45 if left_status == 'detected' else 0.35
                return self._clamp_target(planned, image_width), confidence, True
            if (
                self.previous_target_lane == LEFT
                and self.previous_target_center_x is not None
                and self.missing_left_count <= self.max_missing_frames
                and self.missing_middle_count <= self.max_missing_frames
            ):
                return self._clamp_target(
                    self.previous_target_center_x,
                    image_width,
                ), 0.3, True
            # Drive-straight fallback: no perception lock at all but we're
            # commanded to LEFT. Returning image-center keeps the controller
            # creeping forward at low confidence (just above min_confidence),
            # so the car can clear the obstacle zone instead of stalling.
            return self._image_center_target_fallback(image_width), 0.31, True

        if math.isfinite(lane.right_lane_center_x):
            both_detected = middle_status == 'detected' and right_status == 'detected'
            both_observed = middle_status in observed and right_status in observed
            if both_detected:
                confidence = 1.0
            elif both_observed:
                confidence = 0.8  # at least one from memory but neither synthesized
            else:
                confidence = 0.6  # one side synthesized — weaker target
            return self._clamp_target(lane.right_lane_center_x, image_width), confidence, False
        if math.isfinite(lane.left_lane_center_x):
            planned = lane.left_lane_center_x + self.estimated_lane_width_px
            planned = self._clamp_target(planned, image_width)
            both_detected = left_status == 'detected' and middle_status == 'detected'
            conf = 0.6 if both_detected else 0.4
            return planned, conf, False
        if middle_status in observed and math.isfinite(lane.middle_x):
            planned = lane.middle_x + 0.5 * self.estimated_lane_width_px
            confidence = 0.45 if middle_status == 'detected' else 0.35
            return self._clamp_target(planned, image_width), confidence, True
        if right_status in observed and math.isfinite(lane.right_x):
            planned = lane.right_x - 0.5 * self.estimated_lane_width_px
            confidence = 0.45 if right_status == 'detected' else 0.35
            return self._clamp_target(planned, image_width), confidence, True
        if (
            self.previous_target_lane == RIGHT
            and self.previous_target_center_x is not None
            and self.missing_middle_count <= self.max_missing_frames
            and self.missing_right_count <= self.max_missing_frames
        ):
            return self._clamp_target(self.previous_target_center_x, image_width), 0.3, True
        return math.nan, 0.0, False

    def _current_active_target(self):
        if self._is_finite(self.active_target_center_x):
            return self.active_target_center_x
        if self._is_finite(self.previous_target_center_x):
            return self.previous_target_center_x
        return math.nan

    @staticmethod
    def _image_center_target_fallback(image_width):
        # Used as a "drive straight" fallback when we're commanded to LEFT but
        # have no usable perception lock. Image center keeps the controller
        # error near zero so the car coasts forward through a perception dropout.
        return 0.5 * image_width

    def _reject_jump(self, new_x, last_x, missing_count):
        """Reject single-frame outliers that would label-swap a stable line."""
        if new_x is None or last_x is None:
            return new_x
        if missing_count > self.max_missing_frames:
            return new_x  # history is stale, accept whatever was found
        if abs(new_x - last_x) > MAX_FRAME_JUMP_PX:
            return None  # caller will recover from memory in the next stage
        return new_x

    def _initial_active_target(self, lane, image_width):
        if self._is_finite(self.previous_target_center_x):
            return self.previous_target_center_x
        if math.isfinite(lane.right_lane_center_x):
            return lane.right_lane_center_x
        if math.isfinite(lane.left_lane_center_x):
            return lane.left_lane_center_x
        return 0.5 * image_width

    def _advance_active_target(
        self,
        desired_target,
        image_width,
        freeze=False,
        initial_target=None,
        transition_state=None,
    ):
        desired_target = self._clamp_target(desired_target, image_width)
        initial_target = self._clamp_target(initial_target, image_width)
        self.desired_target_center_x = float(desired_target)

        if freeze:
            if self._is_finite(self.active_target_center_x):
                self.active_target_center_x = self._clamp_target(
                    self.active_target_center_x,
                    image_width,
                )
                return self.active_target_center_x
            self.active_target_center_x = float(desired_target)
            return self.active_target_center_x

        if transition_state in ('CHANGE_LEFT', 'RETURN_RIGHT'):
            return self._advance_quintic_target(
                desired_target,
                transition_state,
                initial_target,
                image_width,
            )

        self.transition_state = None
        self.transition_start_time_sec = None
        self.transition_start_center_x = None
        return self._advance_step_limited_target(
            desired_target,
            initial_target,
            image_width,
        )

    def _advance_quintic_target(
        self,
        desired_target,
        transition_state,
        initial_target,
        image_width,
    ):
        now_sec = self._now_sec()
        if (
            self.transition_state != transition_state
            or self.transition_start_time_sec is None
            or not self._is_finite(self.transition_start_center_x)
        ):
            self.transition_state = transition_state
            self.transition_start_time_sec = now_sec
            if self._is_finite(self.active_target_center_x):
                self.transition_start_center_x = float(self.active_target_center_x)
            elif self._is_finite(initial_target):
                self.transition_start_center_x = float(initial_target)
            else:
                self.transition_start_center_x = float(desired_target)
            self.active_target_center_x = self.transition_start_center_x

        elapsed = max(0.0, now_sec - self.transition_start_time_sec)
        progress = min(1.0, elapsed / self.lane_change_duration_sec)
        eased = self._quintic_smoothstep(progress)
        ideal = self.transition_start_center_x + eased * (
            desired_target - self.transition_start_center_x
        )
        ideal = self._clamp_target(ideal, image_width)
        return self._advance_step_limited_target(ideal, initial_target, image_width)

    def _advance_step_limited_target(
        self,
        desired_target,
        initial_target=None,
        image_width=None,
    ):
        desired_target = self._clamp_target(desired_target, image_width)
        initial_target = self._clamp_target(initial_target, image_width)
        if not self._is_finite(self.active_target_center_x):
            if self._is_finite(initial_target):
                self.active_target_center_x = float(initial_target)
            else:
                self.active_target_center_x = float(desired_target)
            if abs(desired_target - self.active_target_center_x) <= self.max_target_shift_px_per_frame:
                self.active_target_center_x = float(desired_target)
            return self.active_target_center_x

        delta = desired_target - self.active_target_center_x
        max_step = self.max_target_shift_px_per_frame
        if abs(delta) <= max_step:
            self.active_target_center_x = float(desired_target)
        else:
            direction = 1.0 if delta > 0.0 else -1.0
            self.active_target_center_x += direction * max_step
        self.active_target_center_x = self._clamp_target(
            self.active_target_center_x,
            image_width,
        )
        return self.active_target_center_x

    @staticmethod
    def _clamp_target(value, image_width):
        if value is None or not math.isfinite(value):
            return math.nan
        if image_width is None or image_width <= 1:
            return float(value)
        margin = max(8.0, 0.04 * float(image_width))
        lo = margin
        hi = float(image_width) - margin
        return max(lo, min(hi, float(value)))

    @staticmethod
    def _line_in_image(value, image_width):
        return (
            value is not None
            and math.isfinite(value)
            and 0.0 <= float(value) <= float(image_width)
        )

    def _valid_lane_width(self, width):
        return (
            math.isfinite(width)
            and self.min_lane_width_px <= width <= self.max_lane_width_px
        )

    @staticmethod
    def _quintic_smoothstep(progress):
        t = max(0.0, min(1.0, progress))
        return 10.0 * t ** 3 - 15.0 * t ** 4 + 6.0 * t ** 5

    def _now_sec(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _select_lines(self, peaks, image_width, target_lane):
        """Pick up to 3 peaks as (left, middle, right) using geometry + history."""
        if not peaks:
            return None, None, None, set(), set()

        xs = [p['x'] for p in peaks]
        n = len(peaks)
        min_w = self.min_lane_width_px
        max_w = self.max_lane_width_px

        best_triple = None
        # Semantic-geometry priors used to reject obviously-wrong triples:
        # the actual RIGHT solid line must lie in the right portion of the
        # image (xs[k] >= right_min), and the actual LEFT solid (if visible)
        # must be in the left portion (xs[i] <= left_max). Without these,
        # a triple like (95, 123, 265) — where 95 is the real middle dashed
        # and 265 is just an artefact closer to image center than the real
        # right solid — gets labelled L=95, M=123, R=265, exactly the
        # "middle treated as left, right treated as middle" pattern.
        right_must_be_geq = 0.45 * image_width
        left_must_be_leq = 0.55 * image_width
        for i in range(n):
            for j in range(i + 1, n):
                w_lm = xs[j] - xs[i]
                if w_lm < min_w or w_lm > max_w:
                    continue
                for k in range(j + 1, n):
                    w_mr = xs[k] - xs[j]
                    if w_mr < min_w or w_mr > max_w:
                        continue
                    # Hard semantic prior: refuse to call something "right" if
                    # it sits in the left half of the image. The leftmost peak
                    # must likewise not be in the right half.
                    if xs[k] < right_must_be_geq:
                        continue
                    if xs[i] > left_must_be_leq:
                        continue
                    left_lane_center = 0.5 * (xs[i] + xs[j])
                    right_lane_center = 0.5 * (xs[j] + xs[k])
                    target_center = (
                        left_lane_center
                        if target_lane == LEFT
                        else right_lane_center
                    )
                    score = (
                        peaks[i]['score'] + peaks[j]['score'] + peaks[k]['score']
                        + self._continuity_bonus(xs[i], self.last_left_x)
                        + self._continuity_bonus(xs[j], self.last_middle_x)
                        + self._continuity_bonus(xs[k], self.last_right_x)
                        + self._lane_width_bonus(w_lm)
                        + self._lane_width_bonus(w_mr)
                        + self._width_balance_bonus(w_lm, w_mr)
                        + self._target_center_bonus(target_center, image_width)
                    )
                    if best_triple is None or score > best_triple[0]:
                        best_triple = (score, i, j, k)
        if best_triple is not None:
            _, i, j, k = best_triple
            selected = {i, j, k}
            return xs[i], xs[j], xs[k], selected, set(range(n)) - selected

        best_pair = None  # (score, i, j, kind)
        for i in range(n):
            for j in range(i + 1, n):
                w = xs[j] - xs[i]
                if w < min_w or w > max_w:
                    continue
                base = peaks[i]['score'] + peaks[j]['score']
                pair_center = 0.5 * (xs[i] + xs[j])
                mr_desired_target = (
                    pair_center - self.estimated_lane_width_px
                    if target_lane == LEFT
                    else pair_center
                )
                # Semantic prior: only consider MR if the right peak is
                # actually in the right half of the image. This is the key
                # fix that prevents a left-half peak from being labelled
                # "right" — the failure mode the user observed at startup.
                if xs[j] >= right_must_be_geq:
                    self._pair_peaks_xs = (xs[i], xs[j])
                    mr = (
                        base
                        + self._continuity_bonus(xs[i], self.last_middle_x)
                        + self._continuity_bonus(xs[j], self.last_right_x)
                        + self._lane_width_bonus(w)
                        + self._target_center_bonus(mr_desired_target, image_width)
                        + self._pair_kind_bonus('mr', pair_center, image_width, target_lane)
                    )
                    if best_pair is None or mr > best_pair[0]:
                        best_pair = (mr, i, j, 'mr')
                lm_desired_target = (
                    pair_center
                    if target_lane == LEFT
                    else pair_center + self.estimated_lane_width_px
                )
                # Semantic prior for LM: leftmost must not be in the right half
                # (it must plausibly be a "left" line).
                if xs[i] <= left_must_be_leq:
                    self._pair_peaks_xs = (xs[i], xs[j])
                    lm = (
                        base
                        + self._continuity_bonus(xs[i], self.last_left_x)
                        + self._continuity_bonus(xs[j], self.last_middle_x)
                        + self._lane_width_bonus(w)
                        + self._target_center_bonus(lm_desired_target, image_width)
                        + self._pair_kind_bonus('lm', pair_center, image_width, target_lane)
                    )
                    if best_pair is None or lm > best_pair[0]:
                        best_pair = (lm, i, j, 'lm')
        if best_pair is not None:
            _, i, j, kind = best_pair
            selected = {i, j}
            if kind == 'mr':
                return None, xs[i], xs[j], selected, set(range(n)) - selected
            return xs[i], xs[j], None, selected, set(range(n)) - selected

        # Single-peak fallback: assign to nearest tracked line, else split by image center.
        single_best = None  # (score, idx, kind)
        for idx, peak in enumerate(peaks):
            for kind, ref in (
                ('middle', self.last_middle_x),
                ('right', self.last_right_x),
                ('left', self.last_left_x),
            ):
                if ref is None:
                    continue
                if abs(peak['x'] - ref) > self.max_lane_width_px:
                    continue
                score = peak['score'] + self._continuity_bonus(peak['x'], ref)
                if single_best is None or score > single_best[0]:
                    single_best = (score, idx, kind)
        if single_best is not None:
            _, idx, kind = single_best
            x = xs[idx]
            selected = {idx}
            if kind == 'left':
                return x, None, None, selected, set(range(n)) - selected
            if kind == 'middle':
                return None, x, None, selected, set(range(n)) - selected
            return None, None, x, selected, set(range(n)) - selected

        x = xs[0]
        if x < 0.5 * image_width:
            return None, x, None, {0}, set(range(1, n))
        return None, None, x, {0}, set(range(1, n))

    def _lane_width_bonus(self, width):
        if width < self.min_lane_width_px or width > self.max_lane_width_px:
            return -math.inf
        width_range = max(1.0, self.max_lane_width_px - self.min_lane_width_px)
        sigma = max(60.0, 0.35 * width_range)
        return (
            WIDTH_PRIOR_WEIGHT
            * math.exp(-abs(width - self.estimated_lane_width_px) / sigma)
            * 100.0
        )

    def _width_balance_bonus(self, width_a, width_b):
        sigma = max(50.0, 0.25 * self.estimated_lane_width_px)
        return WIDTH_BALANCE_WEIGHT * math.exp(-abs(width_a - width_b) / sigma) * 100.0

    def _target_center_bonus(self, target_center_x, image_width):
        reference = (
            self.previous_target_center_x
            if self.previous_target_center_x is not None
            else 0.5 * image_width
        )
        sigma = max(80.0, 0.25 * image_width)
        return (
            TARGET_CENTER_WEIGHT
            * math.exp(-abs(target_center_x - reference) / sigma)
            * 100.0
        )

    def _pair_kind_bonus(self, kind, pair_center_x, image_width, target_lane):
        """Bias two-line classification based on PHYSICAL position, not on the
        FSM's lane request. The classifier must agree with what is actually
        visible in the image — getting this wrong (e.g., labelling a peak at
        x=540 as 'middle' just because the FSM wants LEFT) cascades into the
        controller chasing a phantom target.

        Note: pair_center_x is unused here — we score the kind purely by the
        positions of the two peaks accessed by the caller.
        """
        if kind not in ('lm', 'mr'):
            return 0.0
        # The caller passes the two peak xs via self._pair_peaks_xs as a tuple
        # right before each scoring call (set just below). This sidesteps a
        # broader signature change.
        peaks = getattr(self, '_pair_peaks_xs', None)
        if peaks is None:
            return 0.0
        xs_i, xs_j = peaks
        score = 0.0
        if kind == 'mr':
            # The RIGHT solid line should be in the right portion of the image.
            if xs_j > 0.70 * image_width:
                score += 400.0
            elif xs_j > 0.50 * image_width:
                score += 200.0
            # The MIDDLE in right-lane view sits left of image centre.
            if xs_i < 0.50 * image_width:
                score += 150.0
        else:  # kind == 'lm'
            # The LEFT solid line in left-lane view sits in the far left.
            if xs_i < 0.30 * image_width:
                score += 300.0
            elif xs_i < 0.40 * image_width:
                score += 150.0
            # The MIDDLE in left-lane view sits right of image centre, but not
            # at the far right (that would be the right solid).
            if 0.40 * image_width <= xs_j <= 0.75 * image_width:
                score += 200.0
            elif xs_j > 0.85 * image_width:
                score -= 200.0  # peak at far right is *almost certainly* the right solid

        # Light bias to the FSM-requested pair when the geometric scores are tied,
        # but small enough that geometry dominates.
        requested_pair = 'lm' if target_lane == LEFT else 'mr'
        if kind == requested_pair:
            score += 40.0
        return score

    @staticmethod
    def _continuity_bonus(x, last_x):
        if last_x is None:
            return 0.0
        return HISTORY_WEIGHT * math.exp(-abs(x - last_x) / CONTINUITY_SIGMA_PX) * 100.0

    @staticmethod
    def _is_finite(value):
        return value is not None and math.isfinite(value)

    def _publish_debug(self, bgr, mask, lane, peaks, raw_peaks, y0, y1, debug_state):
        debug = bgr.copy()
        image_center_x = debug.shape[1] * 0.5

        cv2.rectangle(debug, (0, y0), (debug.shape[1] - 1, y1 - 1), (80, 80, 80), 1)
        self._draw_vertical(debug, image_center_x, y0, y1, (255, 255, 0), 'image_center')

        for peak in raw_peaks:
            x_i = int(round(peak['x']))
            cv2.circle(debug, (x_i, y1 - 6), 3, (180, 180, 180), -1)

        selected = debug_state.get('selected_indices', set())
        rejected = debug_state.get('rejected_indices', set())
        for idx, peak in enumerate(peaks):
            x_i = int(round(peak['x']))
            if idx in selected:
                cv2.line(debug, (x_i, y0 + 4), (x_i, y0 + 14), (255, 255, 255), 2)
            elif idx in rejected:
                cv2.drawMarker(
                    debug, (x_i, y0 + 10), (0, 0, 255),
                    markerType=cv2.MARKER_TILTED_CROSS, markerSize=12, thickness=2,
                )
            else:
                cv2.drawMarker(
                    debug, (x_i, y0 + 10), (0, 165, 255),
                    markerType=cv2.MARKER_TILTED_CROSS, markerSize=12, thickness=2,
                )

        if lane.left_visible:
            left_label = f"left_{debug_state['left_status']}"
            self._draw_vertical(debug, lane.left_x, y0, y1, (255, 0, 0), left_label)
        if lane.middle_visible:
            middle_label = f"middle_{debug_state['middle_status']}"
            self._draw_vertical(debug, lane.middle_x, y0, y1, (0, 255, 255), middle_label)
        if lane.right_visible:
            right_label = f"right_{debug_state['right_status']}"
            self._draw_vertical(debug, lane.right_x, y0, y1, (0, 255, 0), right_label)
        desired_target = debug_state.get('desired_target_center_x', math.nan)
        if self._is_finite(desired_target):
            self._draw_vertical(
                debug,
                desired_target,
                y0,
                y1,
                (0, 165, 255),
                f"desired_{debug_state['requested_target_lane'].lower()}",
            )
        if math.isfinite(lane.target_center_x):
            self._draw_vertical(
                debug,
                lane.target_center_x,
                y0,
                y1,
                (255, 0, 255),
                f"active_{lane.target_lane.lower()}",
            )

        error_text = 'nan' if not math.isfinite(lane.error_px) else f'{lane.error_px:+.1f}px'
        desired_text = (
            'nan' if not self._is_finite(desired_target) else f'{desired_target:.1f}'
        )
        active_target = debug_state.get('active_target_center_x', math.nan)
        active_text = (
            'nan' if not self._is_finite(active_target) else f'{active_target:.1f}'
        )
        cv2.putText(
            debug,
            (
                f"raw={len(raw_peaks)} kept={len(peaks)} "
                f"target={lane.target_lane} "
                f"desired={desired_text} active={active_text} "
                f"middle={debug_state['middle_status']} "
                f"right={debug_state['right_status']} "
                f"conf={lane.confidence:.2f} error={error_text}"
            ),
            (12, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        debug[y0:y1, :] = cv2.addWeighted(debug[y0:y1, :], 0.75, mask_bgr, 0.25, 0.0)

        try:
            debug_msg = self.bridge.cv2_to_imgmsg(debug, encoding='bgr8')
            debug_msg.header = lane.header
            self.debug_pub.publish(debug_msg)
        except CvBridgeError as exc:
            self.get_logger().warn(
                f'cv_bridge debug conversion failed: {exc}',
                throttle_duration_sec=2.0,
            )

    @staticmethod
    def _draw_vertical(image, x, y0, y1, color, label):
        if not math.isfinite(x):
            return
        x_i = int(round(x))
        cv2.line(image, (x_i, y0), (x_i, y1 - 1), color, 2)
        cv2.putText(
            image,
            label,
            (max(0, x_i + 4), max(18, y0 + 18)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )


def main(args=None):
    rclpy.init(args=args)
    node = BeVLaneDetectorNode()
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
