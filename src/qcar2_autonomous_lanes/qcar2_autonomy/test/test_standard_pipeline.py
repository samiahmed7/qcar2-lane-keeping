"""Unit tests for the MPC overtake stack.

The old vision/FSM/DWA nodes were removed (MPC-only cleanup), so only the MPC
reference-planner tests remain. These exercise the progress-window path search
and reference generation without needing ROS or a running sim.
"""
import math
import unittest

import numpy as np

from qcar2_autonomy.mpc_reference_planner_node import (
    LANE_KEEP_RIGHT,
    MpcReferencePlannerNode,
)


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
        self.lane_image_center_px = 320.0
        self.lane_px_to_m = 0.0010
        self.lane_fusion_gain = 0.70
        self.lane_fusion_max_correction = 0.12

    def get_clock(self):
        return _FakeClock()

    def _lane_fusion_target(self, ref):
        return MpcReferencePlannerNode._lane_fusion_target(self, ref)

    def _reference_heading_change(self, ref):
        return MpcReferencePlannerNode._reference_heading_change(self, ref)


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


if __name__ == "__main__":
    unittest.main()
