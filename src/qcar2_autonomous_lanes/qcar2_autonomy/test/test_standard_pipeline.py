import math
import unittest

import cv2
import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.duration import Duration
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Header
from std_msgs.msg import Float32MultiArray, String


def setUpModule():
    if not rclpy.ok():
        rclpy.init()


def tearDownModule():
    if rclpy.ok():
        rclpy.shutdown()


def _set_odom_pose(odom, y=0.0, yaw=0.0, wz=0.0):
    odom.pose.pose.position.y = float(y)
    odom.pose.pose.orientation.z = math.sin(0.5 * yaw)
    odom.pose.pose.orientation.w = math.cos(0.5 * yaw)
    odom.twist.twist.angular.z = float(wz)


class StandardPipelineTest(unittest.TestCase):
    def test_perception_detects_synthetic_two_lane_frame(self):
        from qcar2_autonomy.perception_node import AdvancedPerceptionNode

        node = AdvancedPerceptionNode()
        try:
            width = node.image_width
            height = node.image_height
            frame = np.zeros((height, width, 3), dtype=np.uint8)

            # Draw the synthetic lane boundaries through the same source points
            # used by the default IPM. This keeps the test tied to the actual
            # bird's-eye-view geometry instead of stale hand-picked pixels.
            src = node.src_points.astype(np.int32)
            cv2.line(
                frame,
                tuple(src[0]),
                tuple(src[1]),
                (255, 255, 255),
                16,
            )
            cv2.line(
                frame,
                tuple(src[3]),
                tuple(src[2]),
                (255, 255, 255),
                16,
            )

            left_x, right_x, status = node._process_frame(frame)

            self.assertEqual(status, 2.0)
            self.assertGreaterEqual(left_x, 110.0)
            self.assertLessEqual(left_x, 180.0)
            self.assertGreaterEqual(right_x, 460.0)
            self.assertLessEqual(right_x, 530.0)
        finally:
            node.destroy_node()

    def test_perception_reports_no_lines_on_blank_frame(self):
        from qcar2_autonomy.perception_node import AdvancedPerceptionNode

        node = AdvancedPerceptionNode()
        try:
            frame = np.zeros((node.image_height, node.image_width, 3), dtype=np.uint8)
            self.assertEqual(node._process_frame(frame), (-1.0, -1.0, 0.0))
        finally:
            node.destroy_node()

    def test_perception_debug_image_draws_ipm_polygon(self):
        from qcar2_autonomy.perception_node import AdvancedPerceptionNode

        class FakePublisher:
            def __init__(self):
                self.last_msg = None

            def publish(self, msg):
                self.last_msg = msg

        node = AdvancedPerceptionNode()
        try:
            fake_pub = FakePublisher()
            node.debug_image_pub = fake_pub

            frame = np.zeros((node.image_height, node.image_width, 3), dtype=np.uint8)
            header = Header()
            header.frame_id = 'camera'
            node._publish_debug_image(frame, -1.0, -1.0, 0.0, header)

            self.assertIsNotNone(fake_pub.last_msg)
            self.assertEqual(fake_pub.last_msg.header.frame_id, 'camera')

            from qcar2_autonomy.image_msg_utils import image_msg_to_bgr

            debug = image_msg_to_bgr(fake_pub.last_msg)
            self.assertGreater(int(np.count_nonzero(debug[:, :, 1])), 0)
        finally:
            node.destroy_node()

    def test_validation_updates_ema_and_midpoint_target(self):
        from qcar2_autonomy.validation_node import ValidationNode

        node = ValidationNode()
        try:
            target = node._target_from_raw_lane_data([100.0, 500.0, 2.0])
            self.assertAlmostEqual(target, 300.0)
            self.assertAlmostEqual(node.dynamic_width, 305.0)
        finally:
            node.destroy_node()

    def test_validation_pauses_ema_during_lane_change(self):
        from qcar2_autonomy.validation_node import LANE_CHANGE, ValidationNode

        node = ValidationNode()
        try:
            node.current_state = LANE_CHANGE
            node.dynamic_width = 320.0
            target = node._target_from_raw_lane_data([100.0, 500.0, 2.0])
            self.assertAlmostEqual(target, 300.0)
            self.assertAlmostEqual(node.dynamic_width, 320.0)
        finally:
            node.destroy_node()

    def test_validation_pauses_ema_during_lane_return(self):
        from qcar2_autonomy.validation_node import ValidationNode

        node = ValidationNode()
        try:
            node.current_state = 'LANE_RETURN'
            node.dynamic_width = 320.0
            target = node._target_from_raw_lane_data([100.0, 500.0, 2.0])
            self.assertAlmostEqual(target, 300.0)
            self.assertAlmostEqual(node.dynamic_width, 320.0)
        finally:
            node.destroy_node()

    def test_validation_overrides_impossibly_narrow_two_line_report(self):
        from qcar2_autonomy.validation_node import ValidationNode

        node = ValidationNode()
        try:
            node.dynamic_width = 300.0

            # Width is only 60 px, which is below 60% of the trusted 300 px
            # lane width. Treat the midpoint, x=360, as one right-side line
            # and shift left by half the trusted width.
            target = node._target_from_raw_lane_data([330.0, 390.0, 2.0])

            self.assertAlmostEqual(target, 210.0)
            self.assertAlmostEqual(node.dynamic_width, 300.0)
        finally:
            node.destroy_node()

    def test_validation_single_right_line_uses_half_dynamic_width(self):
        from qcar2_autonomy.validation_node import ValidationNode

        node = ValidationNode()
        try:
            node.dynamic_width = 300.0
            target = node._target_from_raw_lane_data([-1.0, 500.0, 1.0])
            self.assertAlmostEqual(target, 350.0)
        finally:
            node.destroy_node()

    def test_validation_state_callback_normalizes_state(self):
        from qcar2_autonomy.validation_node import ValidationNode

        node = ValidationNode()
        try:
            msg = String()
            msg.data = ' lane_change '
            node._on_system_state(msg)
            self.assertEqual(node.current_state, 'LANE_CHANGE')
        finally:
            node.destroy_node()

    def test_kinematics_formula_without_ros_message_dependency(self):
        from qcar2_autonomy.kinematics_node import KinematicsNode

        steering = KinematicsNode.compute_steering_angle(
            v=0.5,
            omega=1.0,
            wheelbase=0.256,
        )
        self.assertAlmostEqual(steering, math.atan(0.256 / 0.5))
        self.assertEqual(KinematicsNode.compute_steering_angle(0.0, 1.0, 0.256), 0.0)

    def test_state_machine_detects_front_cone_obstacle(self):
        from qcar2_autonomy.state_machine_node import LANE_CHANGE, StateMachineNode

        node = StateMachineNode()
        try:
            scan = LaserScan()
            scan.angle_min = -1.0
            scan.angle_increment = 0.1
            scan.range_min = 0.05
            scan.range_max = 10.0
            scan.ranges = [math.inf] * 21
            scan.ranges[10] = 1.0  # angle 0.0 rad, straight ahead

            node._on_scan(scan)

            self.assertEqual(node.current_state, LANE_CHANGE)
            self.assertIsNone(node.clear_time)
        finally:
            node.destroy_node()

    def test_state_machine_returns_to_start_lane_before_lane_keep(self):
        from qcar2_autonomy.state_machine_node import (
            LANE_CHANGE,
            LANE_KEEP,
            LANE_RETURN,
            StateMachineNode,
        )

        node = StateMachineNode()
        try:
            node.return_settle_sec = 0.0
            odom = Odometry()
            _set_odom_pose(odom, y=0.0, yaw=0.0, wz=0.0)
            node._on_odom(odom)
            node._update_state(obstacle_present=True)
            self.assertEqual(node.current_state, LANE_CHANGE)

            node.clear_time = node.get_clock().now() - Duration(nanoseconds=5_000_000_000)
            node._update_state(obstacle_present=False)
            self.assertEqual(node.current_state, LANE_CHANGE)

            _set_odom_pose(odom, y=0.35, yaw=0.20, wz=0.10)
            node._on_odom(odom)
            node._update_state(obstacle_present=False)
            self.assertEqual(node.current_state, LANE_RETURN)

            # Lateral position alone is no longer enough. If the car is still
            # yawed while crossing back over the lane markers, vision can see
            # the wrong middle/right-line pair and steer out of the first lane.
            _set_odom_pose(odom, y=0.03, yaw=0.25, wz=0.0)
            node._on_odom(odom)
            node._update_state(obstacle_present=False)
            self.assertEqual(node.current_state, LANE_RETURN)

            _set_odom_pose(odom, y=0.03, yaw=0.02, wz=0.20)
            node._on_odom(odom)
            node._update_state(obstacle_present=False)
            self.assertEqual(node.current_state, LANE_RETURN)

            _set_odom_pose(odom, y=0.03, yaw=0.02, wz=0.0)
            node._on_odom(odom)
            node._update_state(obstacle_present=False)
            self.assertEqual(node.current_state, LANE_KEEP)
        finally:
            node.destroy_node()

    def test_state_machine_front_cone_handles_backwards_lidar_wrap(self):
        from qcar2_autonomy.state_machine_node import StateMachineNode

        node = StateMachineNode()
        try:
            node.forward_angle_rad = math.pi
            node.lidar_fov_cone = 0.2

            scan = LaserScan()
            scan.angle_min = -math.pi
            scan.angle_increment = (2.0 * math.pi) / 360.0
            scan.range_min = 0.05
            scan.range_max = 10.0
            scan.ranges = [math.inf] * 361

            indices = node._front_cone_indices(scan)

            self.assertIn(0, indices.tolist())
            self.assertIn(360, indices.tolist())
            self.assertNotIn(180, indices.tolist())
        finally:
            node.destroy_node()

    def test_state_machine_dodge_budget_uses_vehicle_footprint_and_box_width(self):
        from qcar2_autonomy.state_machine_node import StateMachineNode

        node = StateMachineNode()
        try:
            node.vehicle_half_width_m = 0.18
            node.clearance_margin_m = 0.20
            node.min_obstacle_width_m = 0.25
            node.lateral_per_straight_sec = 0.16
            node.min_straight_dur = 1.2
            node.max_straight_dur = 4.5
            node.default_side = 'left'
            node.prefer_side = 'auto'
            node._meas = (0.0, 0.0, math.inf, math.inf)

            direction, straight, shift = node._decide_dodge()

            self.assertEqual(direction, 1.0)
            self.assertAlmostEqual(shift, 0.505)
            self.assertAlmostEqual(straight, shift / 0.16)
        finally:
            node.destroy_node()

    def test_state_machine_offset_obstacle_only_uses_needed_clearance(self):
        from qcar2_autonomy.state_machine_node import StateMachineNode

        node = StateMachineNode()
        try:
            node.vehicle_half_width_m = 0.18
            node.clearance_margin_m = 0.20
            node.min_obstacle_width_m = 0.25
            node.center_deadband_m = 0.20
            node.prefer_side = 'auto'
            node._meas = (0.30, 0.0, math.inf, math.inf)

            direction, _, shift = node._decide_dodge()

            self.assertEqual(direction, -1.0)
            self.assertAlmostEqual(shift, 0.205)
        finally:
            node.destroy_node()

    def test_lidar_tracked_oneway_stays_in_new_lane_after_pass(self):
        from qcar2_autonomy.state_machine_node import (
            LANE_CHANGE,
            LANE_KEEP,
            StateMachineNode,
        )

        node = StateMachineNode()
        try:
            now = node.get_clock().now()
            node.oneway_lane_change = True
            node.current_state = LANE_CHANGE
            node.change_start_time = now - Duration(nanoseconds=5_000_000_000)
            node.pass_clear_time = now - Duration(nanoseconds=1_000_000_000)
            node.lane_change_min_sec = 0.0
            node.pass_confirm_sec = 0.0
            node.pass_clearance_m = 0.1
            node.obs_world_x = 0.0
            node.obs_world_y = 0.0
            node.change_yaw = 0.0
            node.odom_x = 1.0
            node.odom_y = 0.0

            node._update_lidar_tracked(
                front_blocked=False,
                beside=False,
                front_min=math.inf,
            )

            self.assertEqual(node.current_state, LANE_KEEP)
            self.assertIsNone(node.return_start_time)
        finally:
            node.destroy_node()

    def test_lidar_tracked_oneway_hands_back_on_maneuver_complete(self):
        from qcar2_autonomy.state_machine_node import (
            LANE_CHANGE,
            LANE_KEEP,
            StateMachineNode,
        )

        node = StateMachineNode()
        try:
            now = node.get_clock().now()
            node.oneway_lane_change = True
            node.current_state = LANE_CHANGE
            node.change_start_time = now - Duration(nanoseconds=2_000_000_000)
            node.lane_change_min_sec = 0.0
            node.maneuver_complete_signal = True
            node.obs_world_x = 5.0
            node.obs_world_y = 0.0
            node.change_yaw = 0.0
            node.odom_x = 0.0
            node.odom_y = 0.0

            node._update_lidar_tracked(
                front_blocked=False,
                beside=True,
                front_min=math.inf,
            )

            self.assertEqual(node.current_state, LANE_KEEP)
            self.assertFalse(node.maneuver_complete_signal)
        finally:
            node.destroy_node()

    def test_dwa_lateral_target_replaces_blind_left_bias(self):
        from qcar2_autonomy.dwa_local_planner_node import (
            DwaLocalPlannerNode,
            LANE_CHANGE,
            LANE_RETURN,
        )

        node = DwaLocalPlannerNode()
        try:
            node.current_state = LANE_CHANGE
            node.obstacle_points = np.array([[5.0, 0.0]], dtype=np.float32)
            node.lane_change_start_y = 0.0
            node.lane_change_start_yaw = 0.0
            node.odom_y = 0.0
            node.odom_yaw = 0.0
            node.target_lane_y = 0.45
            node.collision_clearance = 0.0
            node.w_clearance = 0.0
            node.w_lateral = 10.0
            node.w_heading = 2.0
            node.w_velocity = 0.0
            node.heading_gate = 0.10
            node.v_grid = np.array([0.30], dtype=np.float32)
            node.w_grid = np.array([-1.0, 0.0, 1.0], dtype=np.float32)

            _, omega_before_target = node._best_candidate()
            self.assertGreater(omega_before_target, 0.0)

            node.odom_y = 0.45
            _, omega_at_target = node._best_candidate()
            self.assertAlmostEqual(omega_at_target, 0.0)

            node.current_state = LANE_RETURN
            _, omega_returning = node._best_candidate()
            self.assertLess(omega_returning, 0.0)
        finally:
            node.destroy_node()

    def test_gazebo_tuner_rmse_from_synthetic_images(self):
        import tempfile
        from pathlib import Path

        from qcar2_autonomy.gazebo_auto_tuner import GazeboAutoTuner

        tuner = GazeboAutoTuner()
        tuner.image_width = 640
        tuner.image_height = 480

        with tempfile.TemporaryDirectory() as tmpdirname:
            tmp_path = Path(tmpdirname)
            for idx, offset in enumerate([-10, 0, 10]):
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.line(frame, (160 + offset, 240), (160 + offset, 479), (255, 255, 255), 1)
                cv2.line(frame, (480 + offset, 240), (480 + offset, 479), (255, 255, 255), 1)
                cv2.imwrite(str(tmp_path / f'frame_{idx:05d}.png'), frame)

            rmse = tuner._rmse_from_images(tmp_path)

        self.assertAlmostEqual(rmse, math.sqrt((100.0 + 0.0 + 100.0) / 3.0), delta=3.0)

    def test_ai_lane_preprocess_returns_resnet_tensor_shape(self):
        from qcar2_autonomy.ai_lane_keeping.preprocess import preprocess_bgr_for_resnet

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:, :, 1] = 128

        tensor = preprocess_bgr_for_resnet(
            frame,
            input_width=224,
            input_height=224,
            crop_top_ratio=0.25,
        )

        self.assertEqual(tensor.shape, (3, 224, 224))
        self.assertEqual(tensor.dtype, np.float32)
        self.assertTrue(np.all(np.isfinite(tensor)))

    def test_dl_lane_resolves_yolo_class_names(self):
        from qcar2_autonomy.dl_lane_segmentation_node import DLLaneSegmentationNode

        names = {0: 'lane2', 1: 'traffic_light'}

        self.assertEqual(DLLaneSegmentationNode.resolve_class_id(names, 'lane2'), 0)
        self.assertEqual(
            DLLaneSegmentationNode.resolve_class_id(names, 'traffic_light'),
            1,
        )
        self.assertEqual(DLLaneSegmentationNode.resolve_class_id(names, '0'), 0)

    def test_dl_lane_target_uses_lowest_mask_band_center(self):
        from qcar2_autonomy.dl_lane_segmentation_node import DLLaneSegmentationNode

        mask = np.zeros((100, 200), dtype=bool)
        mask[60:100, 80:121] = True

        target = DLLaneSegmentationNode.target_x_from_lane_mask(
            mask,
            min_pixels=10,
            y_start_ratio=0.5,
            band_height_px=20,
        )

        self.assertAlmostEqual(target, 100.0, delta=1.0)

    def test_dl_lane_target_uses_instances_bracketing_center(self):
        from qcar2_autonomy.dl_lane_segmentation_node import DLLaneSegmentationNode

        lane_instances = [
            {'bottom_cx': 30.0},
            {'bottom_cx': 80.0},
            {'bottom_cx': 130.0},
            {'bottom_cx': 180.0},
        ]

        target = DLLaneSegmentationNode.target_x_from_lane_instances(
            lane_instances,
            width=200,
            lane_width_px=50.0,
        )

        self.assertAlmostEqual(target, 105.0)

    def test_dl_lane_target_projects_from_single_instance(self):
        from qcar2_autonomy.dl_lane_segmentation_node import DLLaneSegmentationNode

        target = DLLaneSegmentationNode.target_x_from_lane_instances(
            [{'bottom_cx': 150.0}],
            width=200,
            lane_width_px=50.0,
        )

        self.assertAlmostEqual(target, 125.0)

    def test_dl_lane_target_applies_offset_and_clamps(self):
        from qcar2_autonomy.dl_lane_segmentation_node import DLLaneSegmentationNode

        mask = np.zeros((100, 200), dtype=bool)
        mask[70:100, 170:190] = True

        target = DLLaneSegmentationNode.target_x_from_lane_mask(
            mask,
            min_pixels=10,
            y_start_ratio=0.5,
            band_height_px=20,
            target_offset_px=80.0,
        )

        self.assertEqual(target, 199.0)

    def test_ai_lane_xy_output_maps_to_ros_turn_direction(self):
        from qcar2_autonomy.ai_lane_keeper_node import AiLaneKeeperNode

        # JetRacer x > 0 means the model target is right of image center.
        # ROS angular.z < 0 is a right turn, so the default sign should be negative.
        omega_right = AiLaneKeeperNode.steering_from_xy(
            x=0.5,
            y=1.0,
            gain=1.0,
            bias=0.0,
            sign=-1.0,
            max_angular=1.2,
        )
        omega_left = AiLaneKeeperNode.steering_from_xy(
            x=-0.5,
            y=1.0,
            gain=1.0,
            bias=0.0,
            sign=-1.0,
            max_angular=1.2,
        )

        self.assertLess(omega_right, 0.0)
        self.assertGreater(omega_left, 0.0)

    def test_ai_lane_offset_moves_learned_target_sideways(self):
        from qcar2_autonomy.ai_lane_keeper_node import AiLaneKeeperNode

        centered = AiLaneKeeperNode.steering_from_xy(
            x=0.0,
            y=1.0,
            sign=-1.0,
            max_angular=1.2,
        )
        left_bias = AiLaneKeeperNode.steering_from_xy(
            x=0.0,
            y=1.0,
            sign=-1.0,
            max_angular=1.2,
            lane_offset=-0.2,
        )
        right_bias = AiLaneKeeperNode.steering_from_xy(
            x=0.0,
            y=1.0,
            sign=-1.0,
            max_angular=1.2,
            lane_offset=0.2,
        )

        self.assertAlmostEqual(centered, 0.0)
        self.assertGreater(left_bias, 0.0)
        self.assertLess(right_bias, 0.0)

    def test_validation_callback_accepts_float32_multi_array(self):
        from qcar2_autonomy.validation_node import ValidationNode

        node = ValidationNode()
        try:
            msg = Float32MultiArray()
            msg.data = [100.0, 500.0, 2.0]
            node._on_raw_lane_data(msg)
            self.assertAlmostEqual(node.last_target_x, 300.0)
        finally:
            node.destroy_node()
