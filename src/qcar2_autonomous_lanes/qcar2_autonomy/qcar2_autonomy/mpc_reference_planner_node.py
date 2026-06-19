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
        self.declare_parameter("trigger_distance", 2.0)
        self.declare_parameter("obstacle_timeout_sec", 0.8)
        self.declare_parameter("cooldown_sec", 2.0)
        self.declare_parameter("prefer_side", "left")  # left | right | auto
        self.declare_parameter("current_lane_half_width_m", 0.38)
        self.declare_parameter("min_side_clearance_m", 0.6)
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
        self.declare_parameter("lane_fusion_gain", 1.15)
        self.declare_parameter("lane_fusion_alpha", 0.45)
        self.declare_parameter("lane_fusion_max_correction_m", 0.30)
        self.declare_parameter("lane_fusion_timeout_sec", 0.8)
        self.declare_parameter("lane_fusion_min_confidence", 0.45)
        self.declare_parameter("lane_fusion_disable_heading_delta_rad", 3.20)

        self.config_path = pathlib.Path(str(self.get_parameter("config_path").value))
        config = self._load_config(self.config_path)
        prediction = config["controller"]["prediction"]
        self.dt = float(prediction["timestep"])
        self.horizon_steps = int(float(prediction["horizon_time"]) / self.dt)

        self.loop = bool(self.get_parameter("loop").value)
        self.normal_speed = float(self.get_parameter("normal_speed").value)
        self.lane_change_speed = float(self.get_parameter("lane_change_speed").value)
        self.return_speed = float(self.get_parameter("return_speed").value)
        self.trigger_distance = float(self.get_parameter("trigger_distance").value)
        self.obstacle_timeout_sec = float(self.get_parameter("obstacle_timeout_sec").value)
        self.cooldown_sec = float(self.get_parameter("cooldown_sec").value)
        self.prefer_side = str(self.get_parameter("prefer_side").value).strip().lower()
        self.current_lane_half_width = float(
            self.get_parameter("current_lane_half_width_m").value
        )
        self.min_side_clearance = float(self.get_parameter("min_side_clearance_m").value)
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
        self.lane_fusion_timeout_sec = float(
            self.get_parameter("lane_fusion_timeout_sec").value
        )
        self.lane_fusion_min_confidence = float(
            self.get_parameter("lane_fusion_min_confidence").value
        )
        self.lane_fusion_disable_heading_delta = float(
            self.get_parameter("lane_fusion_disable_heading_delta_rad").value
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
        target_correction = 0.0
        if not self.lane_fusion_enabled:
            self.lane_fusion_status = "disabled"
        elif self.mode == LANE_KEEP_RIGHT:
            target_correction = self._lane_fusion_target(ref)
        else:
            self.lane_fusion_status = f"mode_gate:{self.mode}"

        alpha = self.lane_fusion_alpha
        self.lane_fusion_correction = (
            (1.0 - alpha) * self.lane_fusion_correction
            + alpha * target_correction
        )
        self.lane_fusion_pub.publish(Float32(data=float(self.lane_fusion_correction)))
        self.lane_fusion_status_pub.publish(String(data=(
            f"{self.lane_fusion_status} "
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

        self.lane_fusion_status = "no_lane_target"
        now = self.get_clock().now()
        error_px = None
        px_to_m = self.lane_px_to_m
        source = None

        if self.lane_model is not None and self.lane_model_time is not None:
            age = (now - self.lane_model_time).nanoseconds * 1e-9
            lane = self.lane_model
            if (
                age <= self.lane_fusion_timeout_sec
                and lane.confidence >= self.lane_fusion_min_confidence
                and math.isfinite(lane.error_px)
            ):
                error_px = float(lane.error_px)
                source = f"model:conf={lane.confidence:.2f},age={age:.2f}s"
                if (
                    math.isfinite(lane.estimated_lane_width_px)
                    and lane.estimated_lane_width_px > 1.0
                ):
                    px_to_m = self.lane_width_m / float(lane.estimated_lane_width_px)
            elif age > self.lane_fusion_timeout_sec:
                self.lane_fusion_status = f"stale_model:{age:.2f}s"
            elif lane.confidence < self.lane_fusion_min_confidence:
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
            if self.lane_fusion_status in ("disabled", ""):
                self.lane_fusion_status = "no_lane_target"
            return 0.0

        # Positive image error means the lane target is to the camera's right,
        # so shift the reference to the vehicle's right (negative path-left).
        correction = -error_px * px_to_m * self.lane_fusion_gain
        correction = float(np.clip(
            correction,
            -self.lane_fusion_max_correction,
            self.lane_fusion_max_correction,
        ))
        self.lane_fusion_status = (
            f"active:{source},error={error_px:+.1f}px,px_to_m={px_to_m:.5f}"
        )
        return correction

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
            return 0.0
        if self.mode == RETURN_RIGHT:
            return self.return_speed
        if self.mode in (LANE_CHANGE_LEFT, LANE_CHANGE_RIGHT, PASS_OBSTACLE):
            return self.lane_change_speed
        return self.normal_speed

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
