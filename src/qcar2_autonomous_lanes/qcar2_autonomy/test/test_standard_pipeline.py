"""Unit tests for the MPC overtake stack.

The old vision/FSM/DWA nodes were removed (MPC-only cleanup), so only the MPC
reference-planner tests remain. These exercise the progress-window path search
and reference generation without needing ROS or a running sim.
"""
import math
import unittest

import cv2
import numpy as np

from qcar2_autonomy.bev_lane_detector_node import BeVLaneDetectorNode
from qcar2_autonomy.mpc_reference_planner_node import (
    LANE_KEEP_RIGHT,
    PASS_OBSTACLE,
    MpcReferencePlannerNode,
)
from qcar2_autonomy.mpc_drive_node import MpcDriveNode
from qcar2_autonomy.mpc.path_utils import compute_path_from_wp


# A simple out-and-back path: 6 points east along y=0, then 6 points west along
# y=2. Index 2 is (2,0) on the outbound leg; index 9 is (2,2) on the return leg.
_TEST_PATH = np.array(
    [
        [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, math.pi, math.pi, math.pi, math.pi, math.pi, math.pi],
    ],
    dtype=float,
)


class _Published:
    def __init__(self):
        self.messages = []

    def publish(self, msg):
        self.messages.append(msg)


class _FakeDuration:
    def __init__(self, seconds):
        self.nanoseconds = int(seconds * 1e9)


class _FakeTime:
    def __init__(self, seconds):
        self.seconds = seconds

    def __sub__(self, other):
        return _FakeDuration(self.seconds - other.seconds)


class _FakeClock:
    def now(self):
        return _FakeTime(10.0)


class _FakeLogger:
    def info(self, *args, **kwargs):
        pass

    def warn(self, *args, **kwargs):
        pass


class _LaneFusionHarness:
    def __init__(self):
        self.lane_fusion_enabled = True
        self.mode = LANE_KEEP_RIGHT
        self.lane_fusion_alpha = 1.0
        self.lane_fusion_correction = 0.0
        self.lane_fusion_pub = _Published()
        self.lane_fusion_status_pub = _Published()
        self.lane_fusion_status = "disabled"
        self.lane_fusion_disable_heading_delta = 0.35
        self.lane_model = None
        self.lane_model_time = None
        self.lane_target_x = 520.0
        self.lane_target_time = _FakeTime(9.9)
        self.lane_fusion_timeout_sec = 0.6
        self.lane_fusion_min_confidence = 0.30
        self.lane_image_center_px = 320.0
        self.lane_width_m = 0.50
        self.lane_px_to_m = 0.0010
        self.lane_fusion_gain = 0.70
        self.lane_fusion_max_correction = 0.12
        self.lane_fusion_max_step = 0.0
        self.lane_fusion_heading_full_gain = 0.25
        self.lane_fusion_heading_zero_gain = 1.20
        self.lane_fusion_confidence_scaled = True
        self.lane_fusion_min_confidence_gain = 0.25
        self.lane_fusion_max_error_px = 220.0
        self.lane_fusion_max_error_jump_px = 110.0
        self.lane_fusion_obstacle_gate_distance = 1.20
        self.lane_fusion_uncertainty_gate = 0.12
        self.state_uncertainty_m = 0.0
        self.last_accepted_lane_error_px = None
        self.adaptive_bias_enabled = False
        self.adaptive_bias_max = 0.035
        self.adaptive_lane_bias = 0.0

    def get_clock(self):
        return _FakeClock()

    def _lane_fusion_target(self, ref):
        return MpcReferencePlannerNode._lane_fusion_target(self, ref)

    def _reference_heading_change(self, ref):
        return MpcReferencePlannerNode._reference_heading_change(self, ref)

    def _lane_fusion_route_gain(self, heading_change):
        return MpcReferencePlannerNode._lane_fusion_route_gain(self, heading_change)

    def _lane_confidence_gain(self, confidence):
        return MpcReferencePlannerNode._lane_confidence_gain(self, confidence)

    def _lane_uncertainty_gain(self):
        return MpcReferencePlannerNode._lane_uncertainty_gain(self)

    def _lane_fusion_obstacle_gate_active(self):
        return False

    def _adaptive_bias_correction(self):
        return MpcReferencePlannerNode._adaptive_bias_correction(self)


class _FakeCurvedPlanner:
    def __init__(self):
        self.M = 4
        self.s = np.array([0.0, 1.0, 2.0, 3.0], dtype=float)
        self.psip = np.array([0.0, math.pi / 2.0, math.pi / 2.0, math.pi / 2.0])
        self.total_len = 3.0
        self.dl = 1.0

    def _idx_at_s(self, s_query, loop):
        if loop and self.total_len > 0.0:
            s_query = s_query % self.total_len
        return int(np.clip(np.searchsorted(self.s, s_query), 0, self.M - 1))


class _SpeedProfileHarness:
    def __init__(self):
        self.mode = LANE_KEEP_RIGHT
        self.normal_speed = 1.0
        self.lane_change_speed = 0.5
        self.return_speed = 0.4
        self.speed_profile_enabled = True
        self.max_straight_speed = 1.0
        self.min_curve_speed = 0.18
        self.max_lateral_accel = 0.45
        self.curvature_lookahead = 1.0
        self.speed_smoothing_alpha = 1.0
        self.smoothed_speed = None
        self.path_index = 0
        self.loop = False
        self.planner = _FakeCurvedPlanner()
        self.uncertainty_speed_gain = 0.0
        self.uncertainty_speed_min_factor = 0.45
        self.state_uncertainty_m = 0.0

    def _curvature_limited_speed(self, requested_speed):
        return MpcReferencePlannerNode._curvature_limited_speed(self, requested_speed)

    def _path_curvature_ahead(self):
        return MpcReferencePlannerNode._path_curvature_ahead(self)

    def _uncertainty_speed_factor(self):
        return MpcReferencePlannerNode._uncertainty_speed_factor(self)


class _FakeOvertakePlanner:
    def __init__(self):
        self.s = np.array([0.0, 0.6, 1.2, 1.8], dtype=float)
        self.total_len = 4.0
        self.hold_pad = 0.25
        self.ramp_out = 1.0

    def nearest_index(self, x, y):
        return 1

    def project_obstacle(self, x, y):
        return float(x), float(y), 0


class _ChainOvertakeHarness:
    def __init__(self):
        self.path_index = 1
        self.state = np.zeros(4, dtype=float)
        self.loop = False
        self.planner = _FakeOvertakePlanner()
        self.overtake = {
            "s_obs": 0.0,
            "e_obs": 0.0,
            "radius": 0.10,
            "side": 1.0,
        }
        self.mode = LANE_KEEP_RIGHT
        self.obstacle = {
            "front_distance": 1.2,
            "x": 1.4,
            "y": 0.0,
            "radius": 0.12,
            "left_clear": math.inf,
            "right_clear": math.inf,
        }
        self.obstacle_time = _FakeTime(9.9)
        self.obstacle_timeout_sec = 0.8
        self.return_obstacle_hold_distance = 2.0
        self.return_obstacle_min_arc_delta = -0.25
        self.current_lane_half_width = 0.38

    def get_clock(self):
        return _FakeClock()

    def get_logger(self):
        return _FakeLogger()

    def _signed_arc_delta(self, s_value, s_reference):
        return MpcReferencePlannerNode._signed_arc_delta(self, s_value, s_reference)

    def _obstacle_is_fresh(self):
        return MpcReferencePlannerNode._obstacle_is_fresh(self)

    def _extend_overtake_for_chained_obstacle(self):
        return MpcReferencePlannerNode._extend_overtake_for_chained_obstacle(self)


class _FakeLaneModel:
    def __init__(self, confidence, error_px, estimated_lane_width_px=300.0):
        self.confidence = confidence
        self.error_px = error_px
        self.estimated_lane_width_px = estimated_lane_width_px


class _LowConfidenceLaneHarness(_LaneFusionHarness):
    def __init__(self):
        super().__init__()
        self.lane_target_x = None
        self.lane_target_time = None
        self.lane_model = _FakeLaneModel(confidence=0.20, error_px=120.0)
        self.lane_model_time = _FakeTime(9.9)
        self.lane_fusion_min_confidence = 0.30
        self.lane_fusion_hold_low_confidence = True
        self.lane_fusion_correction = -0.22


class _BevHarness:
    def __init__(self):
        self.min_component_area_px = 10
        self.max_component_area_px = 20000
        self.max_component_width_px = 80
        self.min_component_fill = 0.15
        self.min_component_height_px = 8
        self.min_component_elongation = 1.10
        self.max_component_angle_from_vertical_deg = 72.0
        self.line_validation_enabled = True
        self.min_pixels_per_line = 5
        self.histogram_smoothing_kernel = 3
        self.min_line_support_px = 20
        self.max_line_fit_residual_px = 6.0
        self.min_line_inlier_ratio = 0.50
        self.max_line_angle_from_vertical_deg = 75.0
        self.min_peak_line_quality = 0.22

    def _component_line_metrics(self, labels, component_id):
        return BeVLaneDetectorNode._component_line_metrics(self, labels, component_id)

    def _filter_components(self, mask):
        return BeVLaneDetectorNode._filter_components(self, mask)

    def _detect_peaks(self, histogram, mask=None):
        return BeVLaneDetectorNode._detect_peaks(self, histogram, mask)

    def _collect_band(self, bands, histogram, start, end, mask=None):
        return BeVLaneDetectorNode._collect_band(self, bands, histogram, start, end, mask)

    def _band_line_quality(self, mask, start, end):
        return BeVLaneDetectorNode._band_line_quality(self, mask, start, end)


class _AdaptiveBiasHarness:
    def __init__(self):
        self.adaptive_bias_enabled = True
        self.adaptive_bias_alpha = 1.0
        self.adaptive_bias_gain = 0.5
        self.adaptive_bias_max = 0.04
        self.adaptive_bias_deadband = 0.01
        self.adaptive_bias_min_speed = 0.05
        self.adaptive_bias_max_uncertainty = 0.08
        self.adaptive_lane_bias = 0.0
        self.state_uncertainty_m = 0.0
        self.mode = LANE_KEEP_RIGHT
        self.overtake = None
        self.path_index = 0
        self.state = np.array([0.0, 0.08, 0.3, 0.0], dtype=float)
        self.last_cross_track_error = 0.0
        self.planner = type(
            "Planner",
            (),
            {
                "M": 1,
                "xp": np.array([0.0], dtype=float),
                "yp": np.array([0.0], dtype=float),
                "psip": np.array([0.0], dtype=float),
            },
        )()

    def _decay_adaptive_bias(self):
        return MpcReferencePlannerNode._decay_adaptive_bias(self)

    def _cross_track_error(self):
        return MpcReferencePlannerNode._cross_track_error(self)


class MpcPlannerTest(unittest.TestCase):
    def test_progress_window_rejects_nearby_wrong_track_branch(self):
        """Near a point where two legs of the path are close, the global nearest
        index can jump to the wrong leg; the progress-window search must stay on
        the leg the car is actually progressing along."""
        from qcar2_autonomy.mpc.overtake_planner import OvertakePlanner

        planner = OvertakePlanner(_TEST_PATH)
        global_idx = planner.nearest_index(2.0, 1.9)
        window_idx = planner.nearest_index_near(
            2.0, 1.9, center_idx=2, back_m=0.2, ahead_m=1.5, loop=False,
        )
        self.assertEqual(global_idx, 9)   # naive nearest -> return leg (wrong)
        self.assertEqual(window_idx, 2)   # windowed -> stays on outbound leg

    def test_reference_can_start_from_progress_index(self):
        from qcar2_autonomy.mpc.overtake_planner import OvertakePlanner

        planner = OvertakePlanner(_TEST_PATH)
        ref = planner.reference(
            state=np.array([2.0, 1.9, 0.0, 0.0], dtype=float),
            target_v=0.3, horizon_steps=2, dt=0.2, loop=False, start_index=2,
        )
        self.assertAlmostEqual(ref[0, 0], 2.0)
        self.assertAlmostEqual(ref[1, 0], 0.0)

    def test_lane_fusion_target_is_signed_and_capped_in_metres(self):
        harness = _LaneFusionHarness()
        ref = np.zeros((4, 5), dtype=float)

        target = MpcReferencePlannerNode._lane_fusion_target(harness, ref)
        shifted = MpcReferencePlannerNode._apply_lane_fusion(harness, ref)

        self.assertAlmostEqual(target, -0.12)
        self.assertAlmostEqual(harness.lane_fusion_correction, -0.12)
        self.assertAlmostEqual(harness.lane_fusion_pub.messages[-1].data, -0.12)
        self.assertIn("active:ml_target", harness.lane_fusion_status_pub.messages[-1].data)
        self.assertTrue(np.allclose(shifted[0, :], ref[0, :]))
        self.assertTrue(np.allclose(shifted[1, :], -0.12))

    def test_lane_fusion_holds_offset_on_short_low_confidence_drop(self):
        harness = _LowConfidenceLaneHarness()
        ref = np.zeros((4, 5), dtype=float)

        target = MpcReferencePlannerNode._lane_fusion_target(harness, ref)

        self.assertAlmostEqual(target, -0.22)
        self.assertIn("hold_low_model_conf", harness.lane_fusion_status)

    def test_lane_fusion_slew_limit_damps_offset_step(self):
        harness = _LaneFusionHarness()
        harness.lane_fusion_max_step = 0.01
        ref = np.zeros((4, 5), dtype=float)

        shifted = MpcReferencePlannerNode._apply_lane_fusion(harness, ref)

        self.assertAlmostEqual(harness.lane_fusion_correction, -0.01)
        self.assertTrue(np.allclose(shifted[1, :], -0.01))

    def test_lane_fusion_rejects_implausible_error_jump(self):
        harness = _LaneFusionHarness()
        harness.lane_target_x = None
        harness.lane_target_time = None
        harness.lane_model = _FakeLaneModel(confidence=0.95, error_px=140.0)
        harness.lane_model_time = _FakeTime(9.9)
        harness.last_accepted_lane_error_px = 0.0
        harness.lane_fusion_max_error_jump_px = 40.0
        ref = np.zeros((4, 5), dtype=float)

        target = MpcReferencePlannerNode._lane_fusion_target(harness, ref)

        self.assertAlmostEqual(target, 0.0)
        self.assertIn("error_jump_gate", harness.lane_fusion_status)

    def test_lane_fusion_scales_with_model_confidence_and_uncertainty(self):
        harness = _LaneFusionHarness()
        harness.lane_target_x = None
        harness.lane_target_time = None
        harness.lane_model = _FakeLaneModel(
            confidence=0.65,
            error_px=60.0,
            estimated_lane_width_px=300.0,
        )
        harness.lane_model_time = _FakeTime(9.9)
        harness.state_uncertainty_m = 0.03
        ref = np.zeros((4, 5), dtype=float)

        target = MpcReferencePlannerNode._lane_fusion_target(harness, ref)

        conf_gain = harness._lane_confidence_gain(0.65)
        uncertainty_gain = harness._lane_uncertainty_gain()
        expected = -60.0 * (0.50 / 300.0) * 0.70 * conf_gain * uncertainty_gain
        self.assertAlmostEqual(target, expected)
        self.assertIn("conf_gain", harness.lane_fusion_status)
        self.assertIn("unc=", harness.lane_fusion_status)

    def test_curvature_speed_profile_slows_for_sharp_turn(self):
        harness = _SpeedProfileHarness()

        speed = MpcReferencePlannerNode._target_speed(harness)

        self.assertLess(speed, harness.normal_speed)
        self.assertGreaterEqual(speed, harness.min_curve_speed)
        self.assertAlmostEqual(speed, math.sqrt(0.45 / (math.pi / 2.0)))

    def test_uncertainty_speed_factor_slows_when_pose_covariance_grows(self):
        harness = _SpeedProfileHarness()
        harness.uncertainty_speed_gain = 2.0
        harness.state_uncertainty_m = 0.20

        factor = MpcReferencePlannerNode._uncertainty_speed_factor(harness)

        self.assertAlmostEqual(factor, 0.60)

    def test_return_right_holds_left_for_chained_obstacle(self):
        harness = _ChainOvertakeHarness()

        MpcReferencePlannerNode._update_active_overtake(harness, _FakeTime(10.0))

        self.assertEqual(harness.mode, PASS_OBSTACLE)
        self.assertAlmostEqual(harness.overtake["s_obs"], 1.4)
        self.assertAlmostEqual(harness.overtake["radius"], 0.12)

    def test_adaptive_bias_learns_against_cross_track_drift(self):
        harness = _AdaptiveBiasHarness()

        MpcReferencePlannerNode._update_adaptive_bias(harness)

        self.assertAlmostEqual(harness.last_cross_track_error, 0.08)
        self.assertAlmostEqual(harness.adaptive_lane_bias, -0.04)

    def test_bev_component_filter_rejects_horizontal_blob(self):
        harness = _BevHarness()
        mask = np.zeros((120, 180), dtype=np.uint8)
        cv2.line(mask, (88, 8), (98, 112), 255, 3)
        cv2.rectangle(mask, (10, 55), (70, 64), 255, -1)

        filtered = BeVLaneDetectorNode._filter_components(harness, mask)

        self.assertGreater(np.count_nonzero(filtered[:, 80:110]), 0)
        self.assertEqual(np.count_nonzero(filtered[55:65, 10:60]), 0)

    def test_bev_peak_detection_uses_line_quality(self):
        harness = _BevHarness()
        mask = np.zeros((120, 180), dtype=np.uint8)
        cv2.line(mask, (90, 8), (96, 112), 255, 3)
        histogram = np.sum(mask > 0, axis=0).astype(np.float32)

        peaks = BeVLaneDetectorNode._detect_peaks(harness, histogram, mask)

        self.assertEqual(len(peaks), 1)
        self.assertGreater(peaks[0]["line_quality"], 0.3)

    def test_drive_command_rate_limiter_caps_omega_step(self):
        limited = MpcDriveNode._rate_limit(
            value=1.0,
            previous=0.0,
            max_rate=2.0,
            dt=0.05,
        )

        self.assertAlmostEqual(limited, 0.10)

    def test_dense_waypoint_heading_uses_local_tangent_window(self):
        path = compute_path_from_wp(
            np.array([0.0, 1.0, 1.0]),
            np.array([0.0, 0.0, 1.0]),
            step=0.1,
            heading_window_m=0.35,
        )
        heading = np.unwrap(path[2])
        raw_step = math.pi / 2.0
        largest_step = float(np.max(np.abs(np.diff(heading))))

        self.assertLess(largest_step, raw_step)


if __name__ == "__main__":
    unittest.main()
