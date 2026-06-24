#!/usr/bin/env python3
"""Reference and behavior planner for the MPC stack.

The planner owns behavior-level decisions and reference generation. The MPC node
only tracks the reference published here.
"""
import math
import pathlib

import numpy as np
import rclpy
import yaml
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path
from qcar2_msgs.msg import LaneModel
from rclpy.duration import Duration
from rclpy.node import Node
from std_msgs.msg import Float32, Float32MultiArray, String

from qcar2_autonomy.mpc.overtake_planner import OvertakePlanner
from qcar2_autonomy.mpc.path_utils import compute_path_from_wp
from qcar2_autonomy.mpc_drive_node import yaw_from_quaternion


LANE_KEEP_RIGHT = "LANE_KEEP_RIGHT"
LANE_CHANGE_LEFT = "LANE_CHANGE_LEFT"
LANE_CHANGE_RIGHT = "LANE_CHANGE_RIGHT"
PASS_OBSTACLE = "PASS_OBSTACLE"
RETURN_RIGHT = "RETURN_RIGHT"
BLOCKED_STOP = "BLOCKED_STOP"


def quaternion_from_yaw(yaw):
    z = math.sin(0.5 * yaw)
    w = math.cos(0.5 * yaw)
    return z, w


class MpcReferencePlannerNode(Node):
    def __init__(self):
        super().__init__("mpc_reference_planner_node")

        ws = pathlib.Path.home() / "rosbot_ws"
        default_config = ws / "src/qcar2_autonomous_lanes/qcar2_autonomy/config/mpc.yaml"
        self.declare_parameter("config_path", str(default_config))
        self.declare_parameter("waypoints_path", str(ws / "track_waypoints.npy"))
        self.declare_parameter("odom_topic", "/model/qcar2/odometry")
        self.declare_parameter("obstacle_topic", "/mpc/obstacle")
        self.declare_parameter("reference_path_topic", "/mpc/reference_path")
        self.declare_parameter("target_speed_topic", "/mpc/target_speed")
        self.declare_parameter("mode_topic", "/mpc/mode")
        self.declare_parameter("loop", True)
        self.declare_parameter("publish_rate_hz", 10.0)
        self.declare_parameter("normal_speed", 0.30)
        self.declare_parameter("lane_change_speed", 0.18)
        self.declare_parameter("return_speed", 0.12)
        self.declare_parameter("speed_profile_enabled", True)
        self.declare_parameter("max_straight_speed", 0.80)
        self.declare_parameter("min_curve_speed", 0.18)
        self.declare_parameter("max_lateral_accel_mps2", 0.45)
        self.declare_parameter("curvature_lookahead_m", 1.20)
        self.declare_parameter("speed_smoothing_alpha", 0.35)
        self.declare_parameter("trigger_distance", 2.0)
        self.declare_parameter("obstacle_timeout_sec", 0.8)
        self.declare_parameter("cooldown_sec", 2.0)
        self.declare_parameter("prefer_side", "left")  # left | right | auto
        self.declare_parameter("current_lane_half_width_m", 0.38)
        self.declare_parameter("min_side_clearance_m", 0.6)
        self.declare_parameter("return_obstacle_hold_distance_m", 2.0)
        self.declare_parameter("return_obstacle_min_arc_delta_m", -0.25)
        # Drivable corridor: how far the car CENTRE may ride off the centreline
        # during an overtake. If both sides need more than this, the road is
        # blocked and the car stops instead of driving off the track.
        self.declare_parameter("max_lateral_offset_m", 0.65)
        # While stopped (BLOCKED_STOP): front distance at which the path counts
        # as clear again, and the minimum dwell before we re-evaluate / resume.
        self.declare_parameter("blocked_resume_distance_m", 3.0)
        self.declare_parameter("blocked_min_stop_sec", 0.6)
        self.declare_parameter("vehicle_width", 0.20)
        self.declare_parameter("clearance_margin", 0.18)
        self.declare_parameter("ramp_in", 0.9)
        self.declare_parameter("ramp_out", 0.9)
        self.declare_parameter("hold_pad", 0.25)
        self.declare_parameter("path_search_back_m", 0.5)
        self.declare_parameter("path_search_ahead_m", 2.0)
        self.declare_parameter("lane_fusion_enabled", False)
        self.declare_parameter("lane_model_topic", "/qcar2/lane/model")
        self.declare_parameter("lane_target_topic", "/planning/validated_target_x")
        self.declare_parameter("lane_image_center_px", 320.0)
        self.declare_parameter("lane_width_m", 0.50)
        self.declare_parameter("lane_px_to_m", 0.0015)
        self.declare_parameter("lane_fusion_gain", 0.35)
        self.declare_parameter("lane_fusion_alpha", 0.25)
        self.declare_parameter("lane_fusion_max_correction_m", 0.06)
        self.declare_parameter("lane_fusion_max_step_m", 0.006)
        self.declare_parameter("lane_fusion_deadband_px", 0.0)
        self.declare_parameter("lane_fusion_timeout_sec", 0.8)
        self.declare_parameter("lane_fusion_min_confidence", 0.30)
        self.declare_parameter("lane_fusion_hold_low_confidence", True)
        self.declare_parameter("lane_fusion_disable_heading_delta_rad", 3.20)
        self.declare_parameter("lane_fusion_heading_full_gain_rad", 0.60)
        self.declare_parameter("lane_fusion_heading_zero_gain_rad", 2.50)
        self.declare_parameter("lane_fusion_turn_min_gain", 0.0)
        self.declare_parameter("lane_fusion_turn_min_confidence", 0.55)
        self.declare_parameter("lane_fusion_turn_max_step_m", 0.0)
        self.declare_parameter("lane_fusion_turn_bias_m", 0.0)
        self.declare_parameter("lane_fusion_confidence_scaled", True)
        self.declare_parameter("lane_fusion_min_confidence_gain", 0.25)
        self.declare_parameter("lane_fusion_max_error_px", 220.0)
        self.declare_parameter("lane_fusion_max_error_jump_px", 110.0)
        self.declare_parameter("lane_fusion_obstacle_gate_distance_m", 1.20)
        self.declare_parameter("lane_fusion_uncertainty_gate_m", 0.12)
        self.declare_parameter("odom_uncertainty_smoothing_alpha", 0.25)
        self.declare_parameter("uncertainty_speed_gain", 2.0)
        self.declare_parameter("uncertainty_speed_min_factor", 0.45)
        self.declare_parameter("adaptive_bias_enabled", True)
        self.declare_parameter("adaptive_bias_alpha", 0.03)
        self.declare_parameter("adaptive_bias_gain", 0.35)
        self.declare_parameter("adaptive_bias_max_m", 0.035)
        self.declare_parameter("adaptive_bias_deadband_m", 0.015)
        self.declare_parameter("adaptive_bias_min_speed_mps", 0.05)
        self.declare_parameter("adaptive_bias_max_uncertainty_m", 0.08)

        self.config_path = pathlib.Path(str(self.get_parameter("config_path").value))
        config = self._load_config(self.config_path)
        prediction = config["controller"]["prediction"]
        self.dt = float(prediction["timestep"])
        self.horizon_steps = int(float(prediction["horizon_time"]) / self.dt)

        self.loop = bool(self.get_parameter("loop").value)
        self.normal_speed = float(self.get_parameter("normal_speed").value)
        self.lane_change_speed = float(self.get_parameter("lane_change_speed").value)
        self.return_speed = float(self.get_parameter("return_speed").value)
        self.speed_profile_enabled = bool(
            self.get_parameter("speed_profile_enabled").value
        )
        self.max_straight_speed = float(self.get_parameter("max_straight_speed").value)
        self.min_curve_speed = float(self.get_parameter("min_curve_speed").value)
        self.max_lateral_accel = float(
            self.get_parameter("max_lateral_accel_mps2").value
        )
        self.curvature_lookahead = float(
            self.get_parameter("curvature_lookahead_m").value
        )
        self.speed_smoothing_alpha = min(
            1.0,
            max(0.0, float(self.get_parameter("speed_smoothing_alpha").value)),
        )
        self.trigger_distance = float(self.get_parameter("trigger_distance").value)
        self.obstacle_timeout_sec = float(self.get_parameter("obstacle_timeout_sec").value)
        self.cooldown_sec = float(self.get_parameter("cooldown_sec").value)
        self.prefer_side = str(self.get_parameter("prefer_side").value).strip().lower()
        self.current_lane_half_width = float(
            self.get_parameter("current_lane_half_width_m").value
        )
        self.min_side_clearance = float(self.get_parameter("min_side_clearance_m").value)
        self.return_obstacle_hold_distance = float(
            self.get_parameter("return_obstacle_hold_distance_m").value
        )
        self.return_obstacle_min_arc_delta = float(
            self.get_parameter("return_obstacle_min_arc_delta_m").value
        )
        self.max_lateral_offset = float(self.get_parameter("max_lateral_offset_m").value)
        self.blocked_resume_distance = float(
            self.get_parameter("blocked_resume_distance_m").value
        )
        self.blocked_min_stop_sec = float(self.get_parameter("blocked_min_stop_sec").value)
        self.path_search_back_m = float(self.get_parameter("path_search_back_m").value)
        self.path_search_ahead_m = float(self.get_parameter("path_search_ahead_m").value)
        self.lane_fusion_enabled = bool(self.get_parameter("lane_fusion_enabled").value)
        self.lane_image_center_px = float(self.get_parameter("lane_image_center_px").value)
        self.lane_width_m = float(self.get_parameter("lane_width_m").value)
        self.lane_px_to_m = float(self.get_parameter("lane_px_to_m").value)
        self.lane_fusion_gain = float(self.get_parameter("lane_fusion_gain").value)
        self.lane_fusion_alpha = min(
            1.0,
            max(0.0, float(self.get_parameter("lane_fusion_alpha").value)),
        )
        self.lane_fusion_max_correction = abs(
            float(self.get_parameter("lane_fusion_max_correction_m").value)
        )
        self.lane_fusion_max_step = abs(
            float(self.get_parameter("lane_fusion_max_step_m").value)
        )
        self.lane_fusion_deadband_px = abs(
            float(self.get_parameter("lane_fusion_deadband_px").value)
        )
        self.lane_fusion_timeout_sec = float(
            self.get_parameter("lane_fusion_timeout_sec").value
        )
        self.lane_fusion_min_confidence = float(
            self.get_parameter("lane_fusion_min_confidence").value
        )
        self.lane_fusion_hold_low_confidence = bool(
            self.get_parameter("lane_fusion_hold_low_confidence").value
        )
        self.lane_fusion_disable_heading_delta = float(
            self.get_parameter("lane_fusion_disable_heading_delta_rad").value
        )
        self.lane_fusion_heading_full_gain = float(
            self.get_parameter("lane_fusion_heading_full_gain_rad").value
        )
        self.lane_fusion_heading_zero_gain = float(
            self.get_parameter("lane_fusion_heading_zero_gain_rad").value
        )
        self.lane_fusion_turn_min_gain = min(
            1.0,
            max(0.0, float(self.get_parameter("lane_fusion_turn_min_gain").value)),
        )
        self.lane_fusion_turn_min_confidence = float(
            self.get_parameter("lane_fusion_turn_min_confidence").value
        )
        self.lane_fusion_turn_max_step = abs(
            float(self.get_parameter("lane_fusion_turn_max_step_m").value)
        )
        self.lane_fusion_turn_bias = float(
            self.get_parameter("lane_fusion_turn_bias_m").value
        )
        self.lane_fusion_confidence_scaled = bool(
            self.get_parameter("lane_fusion_confidence_scaled").value
        )
        self.lane_fusion_min_confidence_gain = min(
            1.0,
            max(0.0, float(self.get_parameter("lane_fusion_min_confidence_gain").value)),
        )
        self.lane_fusion_max_error_px = float(
            self.get_parameter("lane_fusion_max_error_px").value
        )
        self.lane_fusion_max_error_jump_px = float(
            self.get_parameter("lane_fusion_max_error_jump_px").value
        )
        self.lane_fusion_obstacle_gate_distance = float(
            self.get_parameter("lane_fusion_obstacle_gate_distance_m").value
        )
        self.lane_fusion_uncertainty_gate = float(
            self.get_parameter("lane_fusion_uncertainty_gate_m").value
        )
        self.odom_uncertainty_smoothing_alpha = min(
            1.0,
            max(0.0, float(self.get_parameter("odom_uncertainty_smoothing_alpha").value)),
        )
        self.uncertainty_speed_gain = float(
            self.get_parameter("uncertainty_speed_gain").value
        )
        self.uncertainty_speed_min_factor = min(
            1.0,
            max(0.0, float(self.get_parameter("uncertainty_speed_min_factor").value)),
        )
        self.adaptive_bias_enabled = bool(
            self.get_parameter("adaptive_bias_enabled").value
        )
        self.adaptive_bias_alpha = min(
            1.0,
            max(0.0, float(self.get_parameter("adaptive_bias_alpha").value)),
        )
        self.adaptive_bias_gain = float(self.get_parameter("adaptive_bias_gain").value)
        self.adaptive_bias_max = abs(float(self.get_parameter("adaptive_bias_max_m").value))
        self.adaptive_bias_deadband = abs(
            float(self.get_parameter("adaptive_bias_deadband_m").value)
        )
        self.adaptive_bias_min_speed = float(
            self.get_parameter("adaptive_bias_min_speed_mps").value
        )
        self.adaptive_bias_max_uncertainty = float(
            self.get_parameter("adaptive_bias_max_uncertainty_m").value
        )

        wp_file = pathlib.Path(str(self.get_parameter("waypoints_path").value))
        if not wp_file.is_file():
            raise FileNotFoundError(
                f"waypoints not found: {wp_file}. Record them with record_path_node first."
            )
        waypoints = np.load(str(wp_file))
        if waypoints.ndim != 2 or waypoints.shape[0] != 2:
            raise ValueError(f"waypoints must have shape (2, N), got {waypoints.shape}")
        if self.loop and waypoints.shape[1] > 1:
            closure_gap = np.hypot(
                waypoints[0, -1] - waypoints[0, 0],
                waypoints[1, -1] - waypoints[1, 0],
            )
            if closure_gap > 0.05:
                waypoints = np.column_stack((waypoints, waypoints[:, 0]))
        self.path = compute_path_from_wp(waypoints[0], waypoints[1], step=0.05)
        self.planner = OvertakePlanner(
            self.path,
            vehicle_width=float(self.get_parameter("vehicle_width").value),
            clearance_margin=float(self.get_parameter("clearance_margin").value),
            ramp_in=float(self.get_parameter("ramp_in").value),
            ramp_out=float(self.get_parameter("ramp_out").value),
            hold_pad=float(self.get_parameter("hold_pad").value),
        )

        self.state = None
        self.obstacle = None
        self.obstacle_time = None
        self.overtake = None
        self.cooldown_until = None
        self.blocked_since = None
        self.mode = LANE_KEEP_RIGHT
        self.path_index = None
        self.lane_model = None
        self.lane_model_time = None
        self.lane_target_x = None
        self.lane_target_time = None
        self.lane_fusion_correction = 0.0
        self.lane_fusion_status = "disabled"
        self.lane_fusion_turn_active = False
        self.smoothed_speed = None
        self.state_uncertainty_m = 0.0
        self.last_accepted_lane_error_px = None
        self.adaptive_lane_bias = 0.0
        self.last_cross_track_error = 0.0

        self.reference_pub = self.create_publisher(
            Path,
            str(self.get_parameter("reference_path_topic").value),
            10,
        )
        self.lane_fusion_pub = self.create_publisher(
            Float32,
            "/mpc/lane_fusion_offset",
            10,
        )
        self.lane_fusion_status_pub = self.create_publisher(
            String,
            "/mpc/lane_fusion_status",
            10,
        )
        self.speed_pub = self.create_publisher(
            Float32,
            str(self.get_parameter("target_speed_topic").value),
            10,
        )
        self.mode_pub = self.create_publisher(
            String,
            str(self.get_parameter("mode_topic").value),
            10,
        )
        self.create_subscription(
            Odometry,
            str(self.get_parameter("odom_topic").value),
            self._on_odom,
            10,
        )
        self.create_subscription(
            Float32MultiArray,
            str(self.get_parameter("obstacle_topic").value),
            self._on_obstacle,
            10,
        )
        self.create_subscription(
            LaneModel,
            str(self.get_parameter("lane_model_topic").value),
            self._on_lane_model,
            10,
        )
        self.create_subscription(
            Float32,
            str(self.get_parameter("lane_target_topic").value),
            self._on_lane_target,
            10,
        )

        rate = max(1.0, float(self.get_parameter("publish_rate_hz").value))
        self.create_timer(1.0 / rate, self._tick)

        self.get_logger().info(
            f"MPC reference planner ready: {waypoints.shape[1]} waypoints -> "
            f"{self.path.shape[1]} path samples, horizon={self.horizon_steps}, "
            f"prefer_side={self.prefer_side}, lane_fusion={self.lane_fusion_enabled}"
        )

    @staticmethod
    def _load_config(path):
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _on_odom(self, msg: Odometry):
        p = msg.pose.pose.position
        yaw = yaw_from_quaternion(msg.pose.pose.orientation)
        self.state = np.array(
            [float(p.x), float(p.y), float(msg.twist.twist.linear.x), yaw],
            dtype=float,
        )
        self._update_state_uncertainty(msg)

    def _update_state_uncertainty(self, msg: Odometry):
        cov = np.asarray(msg.pose.covariance, dtype=float)
        if cov.size < 36:
            measured = 0.0
        else:
            vx = cov[0] if math.isfinite(cov[0]) and cov[0] > 0.0 else 0.0
            vy = cov[7] if math.isfinite(cov[7]) and cov[7] > 0.0 else 0.0
            measured = math.sqrt(max(0.0, vx + vy))
        alpha = self.odom_uncertainty_smoothing_alpha
        self.state_uncertainty_m = (
            (1.0 - alpha) * self.state_uncertainty_m
            + alpha * measured
        )

    def _on_obstacle(self, msg: Float32MultiArray):
        if len(msg.data) < 6:
            return
        ox, oy, radius, front_distance, left_clear, right_clear = [float(v) for v in msg.data[:6]]
        if math.isfinite(ox) and math.isfinite(oy) and radius > 0.0:
            self.obstacle = {
                "x": ox,
                "y": oy,
                "radius": radius,
                "front_distance": front_distance,
                "left_clear": left_clear,
                "right_clear": right_clear,
            }
        else:
            self.obstacle = None
        self.obstacle_time = self.get_clock().now()

    def _on_lane_model(self, msg: LaneModel):
        self.lane_model = msg
        self.lane_model_time = self.get_clock().now()

    def _on_lane_target(self, msg: Float32):
        if math.isfinite(msg.data):
            self.lane_target_x = float(msg.data)
            self.lane_target_time = self.get_clock().now()

    def _tick(self):
        if self.state is None:
            return

        self._update_path_index()
        self._update_behavior()
        self._update_adaptive_bias()
        speed = self._target_speed()
        ref = self.planner.reference(
            self.state,
            speed,
            self.horizon_steps,
            self.dt,
            loop=self.loop,
            overtake=self.overtake,
            start_index=self.path_index,
        )
        ref = self._apply_lane_fusion(ref)
        self._publish_reference(ref, speed)

    def _apply_lane_fusion(self, ref):
        camera_correction = 0.0
        self.lane_fusion_turn_active = False
        if not self.lane_fusion_enabled:
            self.lane_fusion_status = "disabled"
        elif self.mode == LANE_KEEP_RIGHT:
            camera_correction = self._lane_fusion_target(ref)
        else:
            self.lane_fusion_status = f"mode_gate:{self.mode}"

        bias = self._adaptive_bias_correction()
        target_correction = camera_correction + bias
        total_limit = self.lane_fusion_max_correction + self.adaptive_bias_max
        if total_limit > 0.0:
            target_correction = float(
                np.clip(target_correction, -total_limit, total_limit)
            )
        alpha = self.lane_fusion_alpha
        blended = (
            (1.0 - alpha) * self.lane_fusion_correction
            + alpha * target_correction
        )
        max_step = self.lane_fusion_max_step
        if self.lane_fusion_turn_active and self.lane_fusion_turn_max_step > 0.0:
            max_step = max(max_step, self.lane_fusion_turn_max_step)
        if max_step > 0.0:
            delta = float(np.clip(
                blended - self.lane_fusion_correction,
                -max_step,
                max_step,
            ))
            self.lane_fusion_correction += delta
        else:
            self.lane_fusion_correction = blended
        self.lane_fusion_pub.publish(Float32(data=float(self.lane_fusion_correction)))
        self.lane_fusion_status_pub.publish(String(data=(
            f"{self.lane_fusion_status} "
            f"camera={camera_correction:+.3f}m "
            f"bias={bias:+.3f}m "
            f"target={target_correction:+.3f}m "
            f"offset={self.lane_fusion_correction:+.3f}m"
        )))

        if abs(self.lane_fusion_correction) < 1e-4:
            return ref

        shifted = np.array(ref, copy=True)
        shifted[0, :] -= self.lane_fusion_correction * np.sin(shifted[3, :])
        shifted[1, :] += self.lane_fusion_correction * np.cos(shifted[3, :])
        return shifted

    def _lane_fusion_target(self, ref):
        heading_change = self._reference_heading_change(ref)
        if heading_change > self.lane_fusion_disable_heading_delta:
            self.lane_fusion_status = f"heading_gate:{heading_change:.2f}rad"
            return 0.0
        route_gain = self._lane_fusion_route_gain(heading_change)
        if self._lane_fusion_obstacle_gate_active():
            self.lane_fusion_status = "obstacle_gate"
            return 0.0
        turn_bias = MpcReferencePlannerNode._lane_fusion_turn_bias(
            self, heading_change, route_gain
        )
        if abs(turn_bias) > 1e-4:
            self.lane_fusion_turn_active = True

        self.lane_fusion_status = "no_lane_target"
        now = self.get_clock().now()
        error_px = None
        px_to_m = self.lane_px_to_m
        source = None
        source_is_model = False
        right_edge_visible = False
        model_confidence = 0.0
        confidence_gain = 1.0

        if self.lane_model is not None and self.lane_model_time is not None:
            age = (now - self.lane_model_time).nanoseconds * 1e-9
            lane = self.lane_model
            model_confidence = float(lane.confidence)
            right_edge_visible = bool(getattr(lane, "right_visible", False))
            if (
                age <= self.lane_fusion_timeout_sec
                and model_confidence >= self.lane_fusion_min_confidence
                and math.isfinite(lane.error_px)
            ):
                error_px = float(lane.error_px)
                source = f"model:conf={model_confidence:.2f},age={age:.2f}s"
                source_is_model = True
                confidence_gain = self._lane_confidence_gain(model_confidence)
                if (
                    math.isfinite(lane.estimated_lane_width_px)
                    and lane.estimated_lane_width_px > 1.0
                ):
                    px_to_m = self.lane_width_m / float(lane.estimated_lane_width_px)
            elif age > self.lane_fusion_timeout_sec:
                self.lane_fusion_status = f"stale_model:{age:.2f}s"
            elif lane.confidence < self.lane_fusion_min_confidence:
                if (
                    self.lane_fusion_hold_low_confidence
                    and abs(self.lane_fusion_correction) > 1e-4
                ):
                    self.lane_fusion_status = (
                        f"hold_low_model_conf:{lane.confidence:.2f}"
                    )
                    return float(self.lane_fusion_correction)
                self.lane_fusion_status = f"low_model_conf:{lane.confidence:.2f}"
            elif not math.isfinite(lane.error_px):
                self.lane_fusion_status = "bad_model_error"

        if error_px is None and self.lane_target_x is not None and self.lane_target_time is not None:
            age = (now - self.lane_target_time).nanoseconds * 1e-9
            if age <= self.lane_fusion_timeout_sec:
                error_px = float(self.lane_target_x - self.lane_image_center_px)
                px_to_m = self.lane_px_to_m
                source = f"ml_target:age={age:.2f}s"
            else:
                self.lane_fusion_status = f"stale_ml_target:{age:.2f}s"

        if error_px is None:
            if abs(turn_bias) > 1e-4:
                self.lane_fusion_status = (
                    f"turn_bias_no_lane:{heading_change:.2f}rad"
                )
                return turn_bias
            if self.lane_fusion_status in ("disabled", ""):
                self.lane_fusion_status = "no_lane_target"
            return 0.0
        if route_gain <= 0.0:
            if (
                source_is_model
                and right_edge_visible
                and model_confidence >= self.lane_fusion_turn_min_confidence
                and self.lane_fusion_turn_min_gain > 0.0
            ):
                route_gain = self.lane_fusion_turn_min_gain
                self.lane_fusion_turn_active = True
            else:
                self.lane_fusion_status = f"route_curve_gate:{heading_change:.2f}rad"
                return turn_bias
        elif (
            source_is_model
            and right_edge_visible
            and model_confidence >= self.lane_fusion_turn_min_confidence
            and heading_change > self.lane_fusion_heading_full_gain
        ):
            route_gain = max(route_gain, self.lane_fusion_turn_min_gain)
            self.lane_fusion_turn_active = True
        if abs(error_px) > self.lane_fusion_max_error_px:
            self.lane_fusion_status = f"error_gate:{error_px:+.1f}px"
            return 0.0
        if (
            self.last_accepted_lane_error_px is not None
            and abs(error_px - self.last_accepted_lane_error_px)
            > self.lane_fusion_max_error_jump_px
        ):
            self.lane_fusion_status = (
                f"error_jump_gate:{error_px:+.1f}px"
            )
            return 0.0
        deadband_px = float(getattr(self, "lane_fusion_deadband_px", 0.0))
        if abs(error_px) <= deadband_px:
            self.last_accepted_lane_error_px = float(error_px)
            self.lane_fusion_status = (
                f"deadband:{source},error={error_px:+.1f}px"
            )
            return 0.0

        # Positive image error means the lane target is to the camera's right,
        # so shift the reference to the vehicle's right (negative path-left).
        uncertainty_gain = self._lane_uncertainty_gain()
        if uncertainty_gain <= 0.0:
            self.lane_fusion_status = (
                f"uncertainty_gate:{self.state_uncertainty_m:.3f}m"
            )
            return 0.0
        correction = (
            -error_px
            * px_to_m
            * self.lane_fusion_gain
            * route_gain
            * confidence_gain
            * uncertainty_gain
            + turn_bias
        )
        correction = float(np.clip(
            correction,
            -self.lane_fusion_max_correction,
            self.lane_fusion_max_correction,
        ))
        self.last_accepted_lane_error_px = float(error_px)
        self.lane_fusion_status = (
            f"active:{source},error={error_px:+.1f}px,px_to_m={px_to_m:.5f},"
            f"route={route_gain:.2f},conf_gain={confidence_gain:.2f},"
            f"unc={uncertainty_gain:.2f},turn_bias={turn_bias:+.3f}"
        )
        return correction

    def _lane_fusion_turn_bias(self, heading_change, route_gain):
        """Right-lane route bias used only on curves.

        Negative correction shifts the reference to the vehicle's right. This
        helps when a recorded route runs near the dashed middle lane in a turn
        but there is clear right-lane space available.
        """
        turn_bias = float(getattr(self, "lane_fusion_turn_bias", 0.0))
        if abs(turn_bias) < 1e-6:
            return 0.0
        full_gain = float(getattr(self, "lane_fusion_heading_full_gain", 0.0))
        if heading_change <= full_gain:
            return 0.0
        fade = float(np.clip(1.0 - route_gain, 0.0, 1.0))
        return turn_bias * fade

    def _lane_fusion_route_gain(self, heading_change):
        full = max(0.0, self.lane_fusion_heading_full_gain)
        zero = max(full + 1e-3, self.lane_fusion_heading_zero_gain)
        if heading_change <= full:
            return 1.0
        if heading_change >= zero:
            return 0.0
        return float(1.0 - (heading_change - full) / (zero - full))

    def _lane_confidence_gain(self, confidence):
        if not self.lane_fusion_confidence_scaled:
            return 1.0
        if confidence <= self.lane_fusion_min_confidence:
            return 0.0
        denom = max(1e-6, 1.0 - self.lane_fusion_min_confidence)
        scaled = (confidence - self.lane_fusion_min_confidence) / denom
        return float(
            self.lane_fusion_min_confidence_gain
            + (1.0 - self.lane_fusion_min_confidence_gain)
            * np.clip(scaled, 0.0, 1.0)
        )

    def _lane_uncertainty_gain(self):
        gate = self.lane_fusion_uncertainty_gate
        if gate <= 0.0:
            return 1.0
        uncertainty = max(0.0, self.state_uncertainty_m)
        if uncertainty >= gate:
            return 0.0
        return float(1.0 - uncertainty / gate)

    def _lane_fusion_obstacle_gate_active(self):
        if self.lane_fusion_obstacle_gate_distance <= 0.0:
            return False
        if not self._obstacle_is_fresh() or self.obstacle is None:
            return False
        return self.obstacle["front_distance"] <= self.lane_fusion_obstacle_gate_distance

    def _reference_heading_change(self, ref):
        n = min(ref.shape[1], 6)
        if n < 2:
            return 0.0
        heading = np.unwrap(ref[3, :n])
        return float(np.max(np.abs(heading - heading[0])))

    def _update_path_index(self):
        if self.path_index is None:
            self.path_index = self.planner.nearest_index(self.state[0], self.state[1])
            return
        self.path_index = self.planner.nearest_index_near(
            self.state[0],
            self.state[1],
            self.path_index,
            back_m=self.path_search_back_m,
            ahead_m=self.path_search_ahead_m,
            loop=self.loop,
        )

    def _update_behavior(self):
        now = self.get_clock().now()
        if self.mode == BLOCKED_STOP:
            self._update_blocked(now)
            return
        if self.overtake is not None:
            self._update_active_overtake(now)
            return

        self.mode = LANE_KEEP_RIGHT
        if self.cooldown_until is not None and now < self.cooldown_until:
            return
        if not self._obstacle_is_fresh():
            return

        obs = self.obstacle
        if obs is None or obs["front_distance"] > self.trigger_distance:
            return

        s_obs, e_obs, _ = self.planner.project_obstacle(obs["x"], obs["y"])
        if abs(e_obs) > self.current_lane_half_width:
            self.get_logger().info(
                f"MPC planner: obstacle offset e={e_obs:+.2f} m is outside "
                f"current lane gate {self.current_lane_half_width:.2f} m.",
                throttle_duration_sec=2.0,
            )
            return

        radius = max(float(obs["radius"]), 0.05)
        side = self._select_side(e_obs, radius, obs["left_clear"], obs["right_clear"])
        if side is None:
            self._enter_blocked(now, obs, e_obs, radius)
            return

        self.overtake = {
            "s_obs": s_obs,
            "e_obs": e_obs,
            "radius": radius,
            "side": side,
        }
        self.mode = LANE_CHANGE_LEFT if side > 0.0 else LANE_CHANGE_RIGHT
        self.get_logger().warn(
            f"MPC planner: obstacle {obs['front_distance']:.2f} m ahead, "
            f"projected e={e_obs:+.2f} m, changing "
            f"{'LEFT' if side > 0.0 else 'RIGHT'}."
        )

    def _enter_blocked(self, now, obs, e_obs, radius):
        """No feasible side: stop and hold until the road clears."""
        self.mode = BLOCKED_STOP
        self.overtake = None
        if self.blocked_since is None:
            self.blocked_since = now
        self.get_logger().warn(
            f"MPC planner: road BLOCKED — obstacle {obs['front_distance']:.2f} m "
            f"ahead, e={e_obs:+.2f} m, r={radius:.2f} m, no room either side. "
            f"Stopping.",
            throttle_duration_sec=2.0,
        )

    def _update_blocked(self, now):
        """Hold while stopped; resume when the path clears or a side opens."""
        self.mode = BLOCKED_STOP
        obs = self.obstacle
        fresh = self._obstacle_is_fresh()
        held = (
            self.blocked_since is not None
            and (now - self.blocked_since).nanoseconds * 1e-9 >= self.blocked_min_stop_sec
        )

        path_clear = (
            not fresh
            or obs is None
            or obs["front_distance"] > self.blocked_resume_distance
        )
        if not path_clear and obs is not None:
            s_obs, e_obs, _ = self.planner.project_obstacle(obs["x"], obs["y"])
            if abs(e_obs) > self.current_lane_half_width:
                path_clear = True
            elif held:
                # Obstacle may have shifted since we stopped — re-check for an
                # opening and overtake if one appeared.
                radius = max(float(obs["radius"]), 0.05)
                side = self._select_side(
                    e_obs, radius, obs["left_clear"], obs["right_clear"]
                )
                if side is not None:
                    self.overtake = {
                        "s_obs": s_obs,
                        "e_obs": e_obs,
                        "radius": radius,
                        "side": side,
                    }
                    self.mode = LANE_CHANGE_LEFT if side > 0.0 else LANE_CHANGE_RIGHT
                    self.blocked_since = None
                    self.get_logger().warn(
                        "MPC planner: opening found while stopped, overtaking "
                        f"{'LEFT' if side > 0.0 else 'RIGHT'}."
                    )
                    return

        if path_clear and held:
            self.mode = LANE_KEEP_RIGHT
            self.blocked_since = None
            self.cooldown_until = now + Duration(seconds=self.cooldown_sec)
            self.get_logger().info("MPC planner: path clear, resuming lane keeping.")

    def _update_active_overtake(self, now):
        idx = self.path_index
        if idx is None:
            idx = self.planner.nearest_index(self.state[0], self.state[1])
        s_now = self.planner.s[idx]
        s_obs = self.overtake["s_obs"]
        radius = self.overtake["radius"]
        half_hold = radius + self.planner.hold_pad
        ds = self._signed_arc_delta(s_now, s_obs)

        if ds > half_hold and self._extend_overtake_for_chained_obstacle():
            self.mode = PASS_OBSTACLE
            return

        if ds > half_hold + self.planner.ramp_out + 0.2:
            self.overtake = None
            self.mode = LANE_KEEP_RIGHT
            self.cooldown_until = now + Duration(seconds=self.cooldown_sec)
            self.get_logger().info("MPC planner: return-right complete, lane keeping.")
            return

        if ds < -half_hold:
            self.mode = (
                LANE_CHANGE_LEFT
                if self.overtake["side"] > 0.0
                else LANE_CHANGE_RIGHT
            )
        elif ds <= half_hold:
            self.mode = PASS_OBSTACLE
        else:
            self.mode = RETURN_RIGHT

    def _extend_overtake_for_chained_obstacle(self):
        """Keep the pass lane if a new obstacle appears during return-right."""
        if self.overtake is None:
            return False
        if self.return_obstacle_hold_distance <= 0.0:
            return False
        if not self._obstacle_is_fresh() or self.obstacle is None:
            return False

        obs = self.obstacle
        if obs["front_distance"] > self.return_obstacle_hold_distance:
            return False

        s_new, e_new, _ = self.planner.project_obstacle(obs["x"], obs["y"])
        if abs(e_new) > self.current_lane_half_width:
            return False

        ds_from_active = self._signed_arc_delta(s_new, self.overtake["s_obs"])
        if ds_from_active < self.return_obstacle_min_arc_delta:
            return False

        radius = max(float(obs["radius"]), 0.05)
        if ds_from_active > 0.0:
            self.overtake["s_obs"] = s_new
            self.overtake["e_obs"] = e_new
            self.overtake["radius"] = max(self.overtake["radius"], radius)

        self.get_logger().warn(
            "MPC planner: second obstacle detected during return-right; "
            "holding pass lane before merging back.",
            throttle_duration_sec=1.5,
        )
        return True

    def _select_side(self, e_obs, radius, left_clear, right_clear):
        """Pick an overtake side that has real room, or None if blocked.

        Required clearance d = r + w_car/2 + margin. During the pass the car
        centre rides at e* = e_obs + side*d, so a side is geometrically feasible
        only if |e*| stays inside the drivable corridor (max_lateral_offset).
        A finite LiDAR side return below min_side_clearance also vetoes a side.
        """
        d = self.planner.clearance_for(radius)
        e_left = e_obs + d          # car-centre offset if we pass on the LEFT
        e_right = e_obs - d         # ... on the RIGHT (negative)

        def feasible(e_signed, clear):
            if abs(e_signed) > self.max_lateral_offset:
                return False
            if math.isfinite(clear) and clear < self.min_side_clearance:
                return False
            return True

        left_ok = feasible(e_left, left_clear)
        right_ok = feasible(e_right, right_clear)
        if not left_ok and not right_ok:
            return None

        if self.prefer_side == "left":
            return 1.0 if left_ok else -1.0
        if self.prefer_side == "right":
            return -1.0 if right_ok else 1.0

        # auto: use the only feasible side, else dodge away from the obstacle
        if left_ok and not right_ok:
            return 1.0
        if right_ok and not left_ok:
            return -1.0
        if math.isfinite(left_clear) and math.isfinite(right_clear):
            if left_clear - right_clear > 0.25:
                return 1.0
            if right_clear - left_clear > 0.25:
                return -1.0
        return -1.0 if e_obs > 0.10 else 1.0

    def _obstacle_is_fresh(self):
        if self.obstacle_time is None:
            return False
        age = (self.get_clock().now() - self.obstacle_time).nanoseconds * 1e-9
        return age <= self.obstacle_timeout_sec

    def _signed_arc_delta(self, s_value, s_reference):
        ds = float(s_value - s_reference)
        if self.loop and self.planner.total_len > 0.0:
            ds = (ds + self.planner.total_len / 2.0) % self.planner.total_len
            ds -= self.planner.total_len / 2.0
        return ds

    def _target_speed(self):
        if self.mode == BLOCKED_STOP:
            self.smoothed_speed = 0.0
            return 0.0
        if self.mode == RETURN_RIGHT:
            base_speed = self.return_speed
        elif self.mode in (LANE_CHANGE_LEFT, LANE_CHANGE_RIGHT, PASS_OBSTACLE):
            base_speed = self.lane_change_speed
        else:
            base_speed = self.normal_speed

        if self.speed_profile_enabled and base_speed > 0.0:
            base_speed = self._curvature_limited_speed(base_speed)
        base_speed *= self._uncertainty_speed_factor()

        if self.smoothed_speed is None:
            self.smoothed_speed = base_speed
        else:
            alpha = self.speed_smoothing_alpha
            self.smoothed_speed = (1.0 - alpha) * self.smoothed_speed + alpha * base_speed
        return float(max(0.0, self.smoothed_speed))

    def _uncertainty_speed_factor(self):
        if self.uncertainty_speed_gain <= 0.0:
            return 1.0
        factor = 1.0 - self.uncertainty_speed_gain * max(0.0, self.state_uncertainty_m)
        return float(np.clip(factor, self.uncertainty_speed_min_factor, 1.0))

    def _curvature_limited_speed(self, requested_speed):
        cap = min(float(requested_speed), self.max_straight_speed)
        curvature = self._path_curvature_ahead()
        if curvature <= 1e-4 or self.max_lateral_accel <= 0.0:
            return cap

        curve_speed = math.sqrt(self.max_lateral_accel / curvature)
        curve_speed = max(self.min_curve_speed, min(cap, curve_speed))
        return float(curve_speed)

    def _path_curvature_ahead(self):
        if self.path_index is None or self.planner.M < 2:
            return 0.0
        i0 = int(self.path_index)
        if self.loop:
            i0 %= self.planner.M
        else:
            i0 = int(np.clip(i0, 0, self.planner.M - 1))

        s0 = float(self.planner.s[i0])
        lookahead = max(self.planner.dl, self.curvature_lookahead)
        if self.loop and self.planner.total_len > 0.0:
            s1 = (s0 + lookahead) % self.planner.total_len
            i1 = self.planner._idx_at_s(s1, loop=True)
            ds = (self.planner.s[i1] - s0) % self.planner.total_len
        else:
            s1 = min(s0 + lookahead, self.planner.total_len)
            i1 = self.planner._idx_at_s(s1, loop=False)
            ds = max(self.planner.s[i1] - s0, self.planner.dl)

        dpsi = math.atan2(
            math.sin(self.planner.psip[i1] - self.planner.psip[i0]),
            math.cos(self.planner.psip[i1] - self.planner.psip[i0]),
        )
        return abs(dpsi) / max(abs(ds), self.planner.dl, 1e-6)

    def _update_adaptive_bias(self):
        if not self.adaptive_bias_enabled or self.state is None or self.path_index is None:
            self._decay_adaptive_bias()
            return
        if self.mode != LANE_KEEP_RIGHT or self.overtake is not None:
            self._decay_adaptive_bias()
            return
        if abs(float(self.state[2])) < self.adaptive_bias_min_speed:
            self._decay_adaptive_bias()
            return
        if self.state_uncertainty_m > self.adaptive_bias_max_uncertainty:
            self._decay_adaptive_bias()
            return

        error = self._cross_track_error()
        self.last_cross_track_error = error
        if abs(error) < self.adaptive_bias_deadband:
            target = 0.0
        else:
            target = -self.adaptive_bias_gain * error
        target = float(np.clip(target, -self.adaptive_bias_max, self.adaptive_bias_max))
        alpha = self.adaptive_bias_alpha
        self.adaptive_lane_bias = (1.0 - alpha) * self.adaptive_lane_bias + alpha * target

    def _decay_adaptive_bias(self):
        alpha = self.adaptive_bias_alpha
        self.adaptive_lane_bias = (1.0 - alpha) * self.adaptive_lane_bias

    def _adaptive_bias_correction(self):
        if not self.adaptive_bias_enabled or self.mode != LANE_KEEP_RIGHT:
            return 0.0
        return float(np.clip(
            self.adaptive_lane_bias,
            -self.adaptive_bias_max,
            self.adaptive_bias_max,
        ))

    def _cross_track_error(self):
        idx = int(np.clip(self.path_index, 0, self.planner.M - 1))
        dx = float(self.state[0] - self.planner.xp[idx])
        dy = float(self.state[1] - self.planner.yp[idx])
        psi = float(self.planner.psip[idx])
        return -math.sin(psi) * dx + math.cos(psi) * dy

    def _publish_reference(self, ref, speed):
        stamp = self.get_clock().now().to_msg()
        path_msg = Path()
        path_msg.header.stamp = stamp
        path_msg.header.frame_id = "map"
        for k in range(ref.shape[1]):
            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = float(ref[0, k])
            pose.pose.position.y = float(ref[1, k])
            z, w = quaternion_from_yaw(float(ref[3, k]))
            pose.pose.orientation.z = z
            pose.pose.orientation.w = w
            path_msg.poses.append(pose)
        self.reference_pub.publish(path_msg)
        self.speed_pub.publish(Float32(data=float(speed)))
        self.mode_pub.publish(String(data=self.mode))


def main(args=None):
    rclpy.init(args=args)
    node = MpcReferencePlannerNode()
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
