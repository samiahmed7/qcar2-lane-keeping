#!/usr/bin/env python3
"""Record the MPC run to an .npz for offline plotting.

Subscribes to everything needed to draw "recorded path vs where the car went vs
the predicted horizon" and dumps it on shutdown (Ctrl-C) or after duration_sec.

    /model/qcar2/odometry   -> actual driven trajectory (x, y, v, yaw)
    /mpc/reference_path     -> the horizon the planner asked the MPC to track
    /mpc/obstacle           -> obstacle (x, y, r)
    /mpc/mode               -> behaviour state string

Output: <out_path> (.npz) with arrays:
    actual         (M, 4)   x, y, v, yaw   (the path the car actually drove)
    actual_t       (M,)     wall seconds since start
    ref_snaps      list saved as object array: each = (K, 2) horizon xy snapshot
    ref_snap_t     (S,)     time of each snapshot
    obstacles      (P, 3)   x, y, r  (unique-ish obstacle samples)
    modes          (Q, 2)   t, mode_code
"""
import math
import pathlib
import time

import numpy as np
import rclpy
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, String


def yaw_from_quaternion(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class MpcLoggerNode(Node):
    def __init__(self):
        super().__init__("mpc_logger_node")
        ws = pathlib.Path.home() / "rosbot_ws"
        self.declare_parameter("out_path", str(ws / "mpc_run_log.npz"))
        self.declare_parameter("waypoints_path", str(ws / "track_waypoints.npy"))
        self.declare_parameter("odom_topic", "/model/qcar2/odometry")
        self.declare_parameter("reference_path_topic", "/mpc/reference_path")
        self.declare_parameter("obstacle_topic", "/mpc/obstacle")
        self.declare_parameter("mode_topic", "/mpc/mode")
        self.declare_parameter("cmd_topic", "/model/qcar2/cmd_vel")
        self.declare_parameter("target_speed_topic", "/mpc/target_speed")
        self.declare_parameter("ref_snapshot_every_sec", 0.5)
        self.declare_parameter("duration_sec", 0.0)  # 0 = until Ctrl-C

        self.out_path = str(self.get_parameter("out_path").value)
        self.t0 = time.monotonic()
        self.ref_every = float(self.get_parameter("ref_snapshot_every_sec").value)
        self.duration = float(self.get_parameter("duration_sec").value)

        self.actual = []          # (t, x, y, v, yaw)
        self.ref_snaps = []       # (K, 2)
        self.ref_snap_t = []
        self.obstacles = []       # (t, x, y, r)
        self.modes = []           # (t, str)
        self.cmd = []             # (t, cmd_v, cmd_omega)
        self.tgt_speed = []       # (t, target_speed)
        self._last_ref_t = -1e9
        self._last_mode = None

        self.create_subscription(Odometry, str(self.get_parameter("odom_topic").value),
                                 self._on_odom, 20)
        self.create_subscription(Path, str(self.get_parameter("reference_path_topic").value),
                                 self._on_ref, 10)
        self.create_subscription(Float32MultiArray, str(self.get_parameter("obstacle_topic").value),
                                 self._on_obs, 10)
        self.create_subscription(String, str(self.get_parameter("mode_topic").value),
                                 self._on_mode, 10)
        from geometry_msgs.msg import Twist as _Twist
        self.create_subscription(_Twist, str(self.get_parameter("cmd_topic").value),
                                 self._on_cmd, 20)
        from std_msgs.msg import Float32 as _F32
        self.create_subscription(_F32, str(self.get_parameter("target_speed_topic").value),
                                 self._on_tgt, 10)
        self.create_timer(0.2, self._check_duration)
        self.get_logger().info(f"MPC logger recording -> {self.out_path}. Ctrl-C to save.")

    def _t(self):
        return time.monotonic() - self.t0

    def _on_odom(self, msg: Odometry):
        p = msg.pose.pose.position
        self.actual.append((self._t(), float(p.x), float(p.y),
                            float(msg.twist.twist.linear.x),
                            yaw_from_quaternion(msg.pose.pose.orientation)))

    def _on_ref(self, msg: Path):
        t = self._t()
        if t - self._last_ref_t < self.ref_every:
            return
        self._last_ref_t = t
        xy = np.array([[ps.pose.position.x, ps.pose.position.y] for ps in msg.poses],
                      dtype=float)
        if xy.size:
            self.ref_snaps.append(xy)
            self.ref_snap_t.append(t)

    def _on_obs(self, msg: Float32MultiArray):
        if len(msg.data) >= 3 and math.isfinite(msg.data[0]) and msg.data[2] > 0:
            self.obstacles.append((self._t(), float(msg.data[0]),
                                   float(msg.data[1]), float(msg.data[2])))

    def _on_mode(self, msg: String):
        if msg.data != self._last_mode:
            self.modes.append((self._t(), msg.data))
            self._last_mode = msg.data

    def _on_cmd(self, msg):
        self.cmd.append((self._t(), float(msg.linear.x), float(msg.angular.z)))

    def _on_tgt(self, msg):
        self.tgt_speed.append((self._t(), float(msg.data)))

    def _check_duration(self):
        if self.duration > 0 and self._t() >= self.duration:
            self.save()
            rclpy.shutdown()

    def save(self):
        actual = np.array(self.actual, dtype=float) if self.actual else np.zeros((0, 5))
        obstacles = np.array(self.obstacles, dtype=float) if self.obstacles else np.zeros((0, 4))
        cmd = np.array(self.cmd, dtype=float) if self.cmd else np.zeros((0, 3))
        tgt = np.array(self.tgt_speed, dtype=float) if self.tgt_speed else np.zeros((0, 2))
        modes = self.modes if self.modes else []
        np.savez(
            self.out_path,
            actual=actual,
            ref_snaps=np.array(self.ref_snaps, dtype=object),
            ref_snap_t=np.array(self.ref_snap_t, dtype=float),
            obstacles=obstacles,      # (t, x, y, r)
            cmd=cmd,                  # (t, cmd_v, cmd_omega)
            tgt_speed=tgt,            # (t, target_speed)
            modes=np.array(modes, dtype=object),
        )
        self.get_logger().info(
            f"Saved {actual.shape[0]} odom, {len(self.ref_snaps)} horizons, "
            f"{obstacles.shape[0]} obstacle, {cmd.shape[0]} cmd samples -> {self.out_path}"
        )


def main(args=None):
    rclpy.init(args=args)
    node = MpcLoggerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.save()
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
