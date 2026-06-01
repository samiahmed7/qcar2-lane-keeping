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
        self.declare_parameter("vehicle_width", 0.20)
        self.declare_parameter("clearance_margin", 0.18)
        self.declare_parameter("ramp_in", 0.9)
        self.declare_parameter("ramp_out", 0.9)
        self.declare_parameter("hold_pad", 0.25)
        self.declare_parameter("path_search_back_m", 0.5)
        self.declare_parameter("path_search_ahead_m", 2.0)

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
        self.path_search_back_m = float(self.get_parameter("path_search_back_m").value)
        self.path_search_ahead_m = float(self.get_parameter("path_search_ahead_m").value)

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
        self.mode = LANE_KEEP_RIGHT
        self.path_index = None

        self.reference_pub = self.create_publisher(
            Path,
            str(self.get_parameter("reference_path_topic").value),
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

        rate = max(1.0, float(self.get_parameter("publish_rate_hz").value))
        self.create_timer(1.0 / rate, self._tick)

        self.get_logger().info(
            f"MPC reference planner ready: {waypoints.shape[1]} waypoints -> "
            f"{self.path.shape[1]} path samples, horizon={self.horizon_steps}, "
            f"prefer_side={self.prefer_side}"
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
        self._publish_reference(ref, speed)

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

        side = self._choose_side(e_obs, obs["left_clear"], obs["right_clear"])
        radius = max(float(obs["radius"]), 0.05)
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

    def _choose_side(self, e_obs, left_clear, right_clear):
        if self.prefer_side == "left":
            return 1.0
        if self.prefer_side == "right":
            return -1.0

        left_ok = left_clear >= self.min_side_clearance
        right_ok = right_clear >= self.min_side_clearance
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
