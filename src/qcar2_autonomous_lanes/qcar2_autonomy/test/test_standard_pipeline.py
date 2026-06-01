"""Unit tests for the MPC overtake stack.

The old vision/FSM/DWA nodes were removed (MPC-only cleanup), so only the MPC
reference-planner tests remain. These exercise the progress-window path search
and reference generation without needing ROS or a running sim.
"""
import math
import unittest

import numpy as np


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


if __name__ == "__main__":
    unittest.main()
