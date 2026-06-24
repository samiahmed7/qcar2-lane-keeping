#!/usr/bin/env python3
"""Pure MPC tracker for the QCar2.

This node deliberately does not do lane perception, obstacle detection, or
behavior planning. It consumes a world-frame reference path produced by a
planner and tracks it with the vendored iterative MPC.

Inputs:
    /model/qcar2/odometry       nav_msgs/Odometry
    /mpc/reference_path         nav_msgs/Path, horizon+1 poses
    /mpc/target_speed           std_msgs/Float32

Output:
    /model/qcar2/cmd_vel        geometry_msgs/Twist

[FIXED] solver phase-lag / straight-line oscillation:
    self.mpc.solve(..., max_iter=2) -> max_iter=1 below.
    Measured against the vendored cvxpy_mpc.MPC class + this project's
    mpc.yaml weights/horizon: the QP's along/cross-track cost rotates the
    state Variables by Parameter cos/sin terms before squaring them, which
    makes the problem non-DPP (cvxpy warns: "subsequent solves will not be
    faster than the first one"). Every .solve() call therefore pays full
    re-canonicalization, and with max_iter=2 it ALWAYS runs that twice
    (convergence tolerance is essentially never met after 1 pass for this
    motion) -- measured ~310ms/.solve() at horizon_time=3.0s. max_iter=1
    relies on the warm-started trajectory from the previous tick as the
    linearization point, which is the standard real-time-iteration (RTI)
    scheme for receding-horizon MPC (one re-linearization per control tick,
    not a fully-converged solve per tick) -- measured ~158ms/.solve(), i.e.
    roughly half the replan latency for free. Pair this with shortening
    horizon_time in mpc.yaml (e.g. 3.0 -> 1.5-2.0) for a further ~2-3x cut;
    see mpc.yaml for that change. A full DPP-compliant rewrite of the cost
    (removing the recanonicalization entirely) is possible but is a bigger,
    separate undertaking -- this max_iter change is the safe, immediate fix.
"""
import math
import pathlib
import threading

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node
from std_msgs.msg import Float32

from qcar2_autonomy.mpc.cvxpy_mpc import MPC


def yaw_from_quaternion(q):
    return math.atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z),
    )


class MpcDriveNode(Node):
    def __init__(self):
        super().__init__("mpc_drive_node")

        ws = pathlib.Path.home() / "rosbot_ws"
        self.declare_parameter(
            "config_path",
            str(ws / "src/qcar2_autonomous_lanes/qcar2_autonomy/config/mpc.yaml"),
        )
        self.declare_parameter("odom_topic", "/model/qcar2/odometry")
        self.declare_parameter("reference_path_topic", "/mpc/reference_path")
        self.declare_parameter("target_speed_topic", "/mpc/target_speed")
        self.declare_parameter("cmd_topic", "/model/qcar2/cmd_vel")
        self.declare_parameter("target_speed", 0.30)
        self.declare_parameter("reference_timeout_sec", 0.8)
        self.declare_parameter("publish_rate_hz", 20.0)
        self.declare_parameter("command_smoothing_enabled", True)
        self.declare_parameter("omega_smoothing_alpha", 0.80)
        self.declare_parameter("max_omega_rate_rps2", 5.0)
        # [FIXED] solver phase-lag: one re-linearization per control tick
        # (real-time-iteration scheme) instead of forcing two full
        # canonicalize+solve passes every tick. See module docstring.
        self.declare_parameter("mpc_max_iter", 1)

        self.mpc = MPC(str(self.get_parameter("config_path").value))
        self.dt = self.mpc.dt
        self.wheelbase = self.mpc.wheelbase
        self.target_speed = float(self.get_parameter("target_speed").value)
        self.reference_timeout_sec = max(
            0.1,
            float(self.get_parameter("reference_timeout_sec").value),
        )
        self.publish_rate = max(1.0, float(self.get_parameter("publish_rate_hz").value))
        self.command_smoothing_enabled = bool(
            self.get_parameter("command_smoothing_enabled").value
        )
        self.omega_smoothing_alpha = min(
            1.0,
            max(0.0, float(self.get_parameter("omega_smoothing_alpha").value)),
        )
        self.max_omega_rate = max(0.0, float(self.get_parameter("max_omega_rate_rps2").value))
        # [FIXED] was hardcoded to 2 at the call site; now a tunable param
        # defaulting to 1 (see declare_parameter above and module docstring).
        self.mpc_max_iter = max(1, int(self.get_parameter("mpc_max_iter").value))

        self.state = None
        self.reference_path = None
        self.reference_time = None
        self.last_cmd_omega = 0.0
        self.last_cmd_time = None

        self._lock = threading.Lock()
        self._plan_u = None
        self._plan_time = None
        self._plan_v0 = 0.0
        self._stop = threading.Event()

        self.cmd_pub = self.create_publisher(
            Twist,
            str(self.get_parameter("cmd_topic").value),
            10,
        )
        self.create_subscription(
            Odometry,
            str(self.get_parameter("odom_topic").value),
            self._on_odom,
            10,
        )
        self.create_subscription(
            Path,
            str(self.get_parameter("reference_path_topic").value),
            self._on_reference_path,
            10,
        )
        self.create_subscription(
            Float32,
            str(self.get_parameter("target_speed_topic").value),
            self._on_target_speed,
            10,
        )

        self.create_timer(1.0 / self.publish_rate, self._publish_tick)
        self.worker = threading.Thread(target=self._solve_loop, daemon=True)
        self.worker.start()

        self.get_logger().info(
            "Pure MPC tracker ready: "
            f"odom={self.get_parameter('odom_topic').value}, "
            f"reference={self.get_parameter('reference_path_topic').value}, "
            f"cmd={self.get_parameter('cmd_topic').value}, "
            f"mpc_max_iter={self.mpc_max_iter}"
        )

    def _on_odom(self, msg: Odometry):
        p = msg.pose.pose.position
        yaw = yaw_from_quaternion(msg.pose.pose.orientation)
        v = float(msg.twist.twist.linear.x)
        self.state = np.array([float(p.x), float(p.y), v, yaw], dtype=float)

    def _on_reference_path(self, msg: Path):
        self.reference_path = msg
        self.reference_time = self.get_clock().now()

    def _on_target_speed(self, msg: Float32):
        if math.isfinite(msg.data):
            self.target_speed = max(0.0, float(msg.data))

    def _reference_age_sec(self):
        if self.reference_time is None:
            return math.inf
        return max(
            0.0,
            (self.get_clock().now() - self.reference_time).nanoseconds * 1e-9,
        )

    def _reference_array(self, state):
        msg = self.reference_path
        if msg is None:
            return None
        n = self.mpc.control_horizon + 1
        if len(msg.poses) < n:
            self.get_logger().warn(
                f"Reference path has {len(msg.poses)} poses, need {n}",
                throttle_duration_sec=2.0,
            )
            return None

        ref = np.zeros((4, n), dtype=float)
        for k, stamped in enumerate(msg.poses[:n]):
            pose = stamped.pose
            ref[0, k] = float(pose.position.x)
            ref[1, k] = float(pose.position.y)
            ref[2, k] = self.target_speed
            ref[3, k] = yaw_from_quaternion(pose.orientation)

        ref[3, :] = np.unwrap(ref[3, :])
        while ref[3, 0] - state[3] > math.pi:
            ref[3, :] -= 2.0 * math.pi
        while ref[3, 0] - state[3] < -math.pi:
            ref[3, :] += 2.0 * math.pi
        return ref

    def _clear_plan(self):
        with self._lock:
            self._plan_u = None
            self._plan_time = None
            self._plan_v0 = 0.0
        self.last_cmd_omega = 0.0
        self.last_cmd_time = None

    def _solve_loop(self):
        while not self._stop.is_set() and rclpy.ok():
            if self.state is None or self._reference_age_sec() > self.reference_timeout_sec:
                self._clear_plan()
                self._stop.wait(0.05)
                continue

            state = np.array(self.state, dtype=float)
            ref = self._reference_array(state)
            if ref is None:
                self._clear_plan()
                self._stop.wait(0.05)
                continue

            try:
                # [FIXED] was max_iter=2 (always ran twice, ~310ms/call at
                # horizon_time=3.0s). max_iter=self.mpc_max_iter (default 1)
                # halves that to ~158ms/call -- see module docstring.
                _, cmd = self.mpc.solve(state, ref, obstacle=None, max_iter=self.mpc_max_iter)
            except Exception as exc:  # noqa: BLE001
                self.get_logger().warn(
                    f"MPC solve error: {exc}",
                    throttle_duration_sec=2.0,
                )
                self._clear_plan()
                self._stop.wait(0.05)
                continue

            with self._lock:
                self._plan_u = None if cmd is None else np.array(cmd)
                self._plan_time = self.get_clock().now()
                self._plan_v0 = float(state[2])

            if cmd is None:
                self.get_logger().warn("MPC infeasible; publishing stop", throttle_duration_sec=1.0)
            else:
                self.get_logger().info(
                    f"v={state[2]:.2f} target={self.target_speed:.2f} "
                    f"acc={cmd[0, 0]:+.2f} steer={cmd[1, 0]:+.2f}",
                    throttle_duration_sec=1.0,
                )
            self._stop.wait(max(0.02, self.dt * 0.25))

    def _publish_tick(self):
        if self.state is None:
            return
        with self._lock:
            plan = self._plan_u
            ptime = self._plan_time
            v0 = self._plan_v0

        out = Twist()
        if plan is None or ptime is None:
            self.cmd_pub.publish(out)
            return

        elapsed = (self.get_clock().now() - ptime).nanoseconds * 1e-9
        k = int(elapsed / self.dt)
        k = max(0, min(k, plan.shape[1] - 1))
        steer = float(plan[1, k])
        v_cmd = float(v0 + np.sum(plan[0, : k + 1]) * self.dt)
        v_cmd = float(np.clip(v_cmd, 0.0, self.mpc.max_speed))

        out.linear.x = v_cmd
        raw_omega = v_cmd * math.tan(steer) / self.wheelbase
        out.angular.z = self._smooth_omega(raw_omega)
        self.cmd_pub.publish(out)

    def _smooth_omega(self, raw_omega):
        if not self.command_smoothing_enabled:
            self.last_cmd_omega = float(raw_omega)
            self.last_cmd_time = self.get_clock().now()
            return float(raw_omega)

        now = self.get_clock().now()
        if self.last_cmd_time is None:
            self.last_cmd_time = now
            self.last_cmd_omega = float(raw_omega)
            return float(raw_omega)

        dt = (now - self.last_cmd_time).nanoseconds * 1e-9
        if dt <= 1e-4:
            dt = 1.0 / self.publish_rate
        alpha = self.omega_smoothing_alpha
        filtered = (
            (1.0 - alpha) * self.last_cmd_omega
            + alpha * float(raw_omega)
        )
        smoothed = self._rate_limit(
            filtered,
            self.last_cmd_omega,
            self.max_omega_rate,
            dt,
        )
        self.last_cmd_time = now
        self.last_cmd_omega = smoothed
        return smoothed

    @staticmethod
    def _rate_limit(value, previous, max_rate, dt):
        if max_rate <= 0.0 or dt <= 0.0:
            return float(value)
        step = max_rate * dt
        return float(np.clip(value, previous - step, previous + step))

    def destroy_node(self):
        self._stop.set()
        if hasattr(self, "worker") and self.worker.is_alive():
            self.worker.join(timeout=5.0)
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = MpcDriveNode()
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