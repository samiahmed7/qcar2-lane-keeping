#!/usr/bin/env python3
"""Dynamic Window Approach local planner for LANE_CHANGE obstacle avoidance.

Active while the state machine reports LANE_CHANGE or LANE_RETURN; in LANE_KEEP
this node publishes nothing so the lane-following PID stays in control.

Algorithm (each tick):
  1. Project the latest LiDAR scan to Cartesian (x, y) points in the robot frame.
  2. Sample a dynamic window: a grid of (v, omega) candidates inside the
     vehicle's allowed velocity envelope.
  3. Forward-simulate each candidate using the unicycle model:
        x_{k+1} = x_k + v cos(theta_k) dt
        y_{k+1} = y_k + v sin(theta_k) dt
        theta_{k+1} = theta_k + omega dt
  4. For every step of every candidate trajectory, measure the distance to the
     nearest LiDAR point. The minimum across the trajectory is its clearance.
  5. Reject candidates whose clearance falls below ``collision_clearance`` --
     they would brush or hit the obstacle.
  6. Score the survivors:
        score = w_clear * clearance + w_heading * heading + w_velocity * v
     where ``heading`` rewards directions that still progress forward, and the
     velocity term breaks ties in favor of making forward progress instead of
     stalling.
  7. Publish the highest-scoring (v, omega) as a Twist.

This is a deliberately compact DWA: no full obstacle-footprint inflation, no
admissible-velocity acceleration constraints (no /odom feedback loop). It
exists to provide visible trajectory-planned avoidance in the sim demo without
standing up the entire Nav2 stack and its lifecycle/costmap config.
"""
import math

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String


LANE_KEEP = 'LANE_KEEP'
LANE_CHANGE = 'LANE_CHANGE'
LANE_RETURN = 'LANE_RETURN'
DWA_ACTIVE_STATES = (LANE_CHANGE, LANE_RETURN)


class DwaLocalPlannerNode(Node):
    """Sample-based local planner; emits Twist when state == LANE_CHANGE."""

    def __init__(self):
        super().__init__('dwa_local_planner_node')

        # Sampling grid
        self.declare_parameter('v_min', 0.10)
        self.declare_parameter('v_max', 0.30)
        self.declare_parameter('v_samples', 5)
        self.declare_parameter('w_max', 1.0)
        self.declare_parameter('w_samples', 11)

        # Trajectory rollout
        self.declare_parameter('horizon_sec', 2.5)
        self.declare_parameter('dt', 0.10)

        # Safety
        self.declare_parameter('collision_clearance', 0.38)  # metres
        self.declare_parameter('lidar_max_consider', 5.0)    # ignore beams beyond this

        # Cost weights. Tuning notes:
        #   w_clearance  - keep high; collision avoidance must dominate.
        #   w_lateral    - drives the lane-change *destination*. Increase if
        #                  the car settles short of target_lane_y, decrease
        #                  if it overshoots before aligning.
        #   w_heading    - rewards straightening back to the yaw captured when
        #                  LANE_CHANGE started. This is more robust than
        #                  assuming the road is always aligned with odom yaw 0.
        #                  Only paid out when |y_error| < heading_gate, so the
        #                  car finishes the turn before this kicks in.
        #   w_velocity   - small forward-progress nudge; breaks ties between
        #                  otherwise-equivalent trajectories.
        self.declare_parameter('w_clearance', 1.0)
        self.declare_parameter('w_lateral', 5.0)
        self.declare_parameter('w_heading', 2.0)
        self.declare_parameter('w_velocity', 0.2)

        # Lane-change goal: target lateral offset from the y position the car
        # had at the moment LANE_CHANGE was triggered, in metres. Positive =
        # left lane. During LANE_RETURN the active target becomes 0.0 m so the
        # car returns to its original lane before the PID takes control again.
        self.declare_parameter('target_lane_y', 0.60)
        # Heading reward only pays out when the trajectory's final lateral
        # position is within this many metres of target_lane_y. This is what
        # turns "swerve left" into "swerve left, then straighten" -- the
        # heading term has zero pull while the car is still mid-lane-change.
        self.declare_parameter('heading_gate', 0.10)

        # Topics
        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('state_topic', '/system/current_state')
        self.declare_parameter('odom_topic', '/odom')

        # Cache parameters
        self.v_min = float(self.get_parameter('v_min').value)
        self.v_max = float(self.get_parameter('v_max').value)
        self.v_samples = int(self.get_parameter('v_samples').value)
        self.w_max = float(self.get_parameter('w_max').value)
        self.w_samples = int(self.get_parameter('w_samples').value)
        self.horizon_sec = float(self.get_parameter('horizon_sec').value)
        self.dt = float(self.get_parameter('dt').value)
        self.collision_clearance = float(self.get_parameter('collision_clearance').value)
        self.lidar_max_consider = float(self.get_parameter('lidar_max_consider').value)
        self.w_clearance = float(self.get_parameter('w_clearance').value)
        self.w_lateral = float(self.get_parameter('w_lateral').value)
        self.w_heading = float(self.get_parameter('w_heading').value)
        self.w_velocity = float(self.get_parameter('w_velocity').value)
        self.target_lane_y = float(self.get_parameter('target_lane_y').value)
        self.heading_gate = float(self.get_parameter('heading_gate').value)

        # Runtime state
        self.current_state = LANE_KEEP
        self.obstacle_points = None  # (N, 2) numpy array of (x, y) in robot frame
        # Odometry-derived pose, in the odom frame (whatever frame /odom uses).
        # We do not care about the absolute frame, only relative changes from
        # the moment LANE_CHANGE was triggered.
        self.odom_x = None
        self.odom_y = None
        self.odom_yaw = None
        # Snapshot of the y coordinate at the instant LANE_CHANGE began.
        # All target-y arithmetic is referenced to this snapshot so the
        # "0.45 m left" goal is relative to wherever the lane change started.
        self.lane_change_start_y = None
        self.lane_change_start_yaw = None
        self.steps = max(1, int(round(self.horizon_sec / self.dt)))
        self.v_grid = np.linspace(self.v_min, self.v_max, self.v_samples)
        self.w_grid = np.linspace(-self.w_max, self.w_max, self.w_samples)

        self.cmd_pub = self.create_publisher(
            Twist,
            self.get_parameter('cmd_vel_topic').value,
            10,
        )
        self.create_subscription(
            LaserScan,
            self.get_parameter('scan_topic').value,
            self._on_scan,
            10,
        )
        self.create_subscription(
            String,
            self.get_parameter('state_topic').value,
            self._on_state,
            10,
        )
        self.create_subscription(
            Odometry,
            self.get_parameter('odom_topic').value,
            self._on_odom,
            10,
        )

        # Plan at 10 Hz. Tighter than the LiDAR scan rate (~10 Hz) so we always
        # publish even if scans arrive irregularly.
        self.create_timer(0.1, self._tick)

        self.get_logger().info(
            f'DWA planner ready: v=[{self.v_min},{self.v_max}]x{self.v_samples}, '
            f'w=+/-{self.w_max}x{self.w_samples}, horizon={self.horizon_sec}s, '
            f'min_clearance={self.collision_clearance}m, '
            f'target_lane_y={self.target_lane_y}m, heading_gate={self.heading_gate}m'
        )

    def _on_state(self, msg: String):
        new_state = msg.data.strip().upper() or LANE_KEEP
        if new_state != self.current_state:
            self.get_logger().info(
                f'state {self.current_state} -> {new_state}'
            )
            if new_state == LANE_CHANGE:
                # Snapshot the current odometry y as the reference for the
                # lateral goal. If no odom has arrived yet, leave it None and
                # let _tick() snapshot on the first scoring pass instead.
                if self.odom_y is not None:
                    self.lane_change_start_y = self.odom_y
                if self.odom_yaw is not None:
                    self.lane_change_start_yaw = self.odom_yaw
                if self.lane_change_start_y is not None:
                    self.get_logger().info(
                        f'LANE_CHANGE snapshot: start_y={self.lane_change_start_y:.3f}, '
                        f'start_yaw={self._format_optional(self.lane_change_start_yaw)}, '
                        f'target=+{self.target_lane_y:.2f} m'
                    )
            elif new_state == LANE_KEEP:
                # Returning to LANE_KEEP -> forget the snapshot so the next
                # trigger establishes a fresh reference.
                self.lane_change_start_y = None
                self.lane_change_start_yaw = None
        self.current_state = new_state

    def _on_odom(self, msg: Odometry):
        """Cache latest pose. We only use position.y and yaw."""
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        # Quaternion -> yaw (rotation about z). The full conversion is:
        #     yaw = atan2( 2*(w*z + x*y),  1 - 2*(y*y + z*z) )
        # For a planar vehicle, x = y = 0 in the quaternion, so this reduces
        # to atan2(2*w*z, 1 - 2*z*z), but the general form below is safe and
        # only costs a few flops.
        self.odom_x = float(p.x)
        self.odom_y = float(p.y)
        self.odom_yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z),
        )

    def _on_scan(self, msg: LaserScan):
        """Convert LaserScan to (N, 2) Cartesian points in the robot frame."""
        ranges = np.asarray(msg.ranges, dtype=np.float32)
        n = ranges.size
        if n == 0:
            self.obstacle_points = None
            return

        # Keep finite, positive, in-range readings, and clip beams beyond what
        # we care to plan around (saves work and keeps the cost field finite).
        valid_mask = (
            np.isfinite(ranges)
            & (ranges > 0.0)
            & (ranges >= msg.range_min)
            & (ranges <= min(msg.range_max, self.lidar_max_consider))
        )
        if not np.any(valid_mask):
            self.obstacle_points = np.zeros((0, 2), dtype=np.float32)
            return

        angles = msg.angle_min + np.arange(n, dtype=np.float32) * msg.angle_increment
        r = ranges[valid_mask]
        a = angles[valid_mask]
        # (x, y) in the LiDAR/robot frame; +x = forward, +y = left.
        xs = r * np.cos(a)
        ys = r * np.sin(a)
        self.obstacle_points = np.column_stack([xs, ys]).astype(np.float32)

    def _tick(self):
        if self.current_state not in DWA_ACTIVE_STATES:
            # In LANE_KEEP we yield to the lane-keeping PID. Publishing nothing
            # here means there's only one author on /cmd_vel at a time.
            return
        if self.obstacle_points is None or self.obstacle_points.size == 0:
            # No usable scan yet: emit a gentle forward crawl so we don't sit
            # still in front of an obstacle waiting for the planner to wake up.
            out = Twist()
            out.linear.x = self.v_min
            self.cmd_pub.publish(out)
            return
        # Late-bind the start_y snapshot if _on_state ran before odom was up.
        if self.lane_change_start_y is None and self.odom_y is not None:
            self.lane_change_start_y = self.odom_y
        if self.lane_change_start_yaw is None and self.odom_yaw is not None:
            self.lane_change_start_yaw = self.odom_yaw

        v, w = self._best_candidate()
        out = Twist()
        out.linear.x = float(v)
        out.angular.z = float(w)
        self.cmd_pub.publish(out)

    def _best_candidate(self):
        """Score the dynamic window and return the winning (v, omega).

        Scoring (per surviving candidate):

            score = w_clearance * clearance
                  - w_lateral   * lateral_error
                  + w_heading   * heading_reward     # only if near target lane
                  + w_velocity  * v

        where, for the final pose of each forward-simulated trajectory:

            clearance       = min distance from the trajectory's predicted
                              poses to any LiDAR return, in metres.
            lateral_error   = |active_target_y - lateral_progress|, where
                              lateral_progress is how far the car will have
                              moved sideways from its lane-change start point,
                              expressed in the odom frame.
            heading_reward  = cos(yaw_error_at_trajectory_end), but ONLY when
                              |lateral_error| < heading_gate. yaw_error is
                              measured against the yaw snapshot from the start
                              of the lane change, not absolute odom yaw 0.
                              This is the "swerve first, straighten second"
                              mechanism -- the term has no pull during the
                              swerve and dominates once the car is in lane.

        The old "lane_change_left_bias" term has been removed: there is no
        blind directional pull anymore. Direction emerges naturally from
        lateral_error pulling the car toward target_lane_y.
        """
        # If we still have no pose reference, fall back to "creep forward"
        # so we never freeze in front of an obstacle waiting for odom.
        if self.lane_change_start_y is None or self.odom_y is None:
            return self.v_min, 0.0
        if self.lane_change_start_yaw is None:
            self.lane_change_start_yaw = self.odom_yaw if self.odom_yaw is not None else 0.0

        active_target_y = self._active_target_lane_y()

        # Lateral progress already accomplished, in odom y.
        # Positive values mean the car has moved toward the target lane
        # (assuming target_lane_y is also positive, i.e. left lane).
        current_lateral = self.odom_y - self.lane_change_start_y
        yaw_now = self.odom_yaw if self.odom_yaw is not None else 0.0

        best_score = -math.inf
        best_v = self.v_min
        best_w = 0.0

        for v in self.v_grid:
            for w in self.w_grid:
                # Forward-simulate this (v, w) using the unicycle model. Output
                # is in the ROBOT frame (start of trajectory = origin).
                xs, ys, thetas = self._rollout(float(v), float(w))

                clearance = self._trajectory_clearance(xs, ys)
                if clearance < self.collision_clearance:
                    continue  # would collide; drop candidate

                # Project the trajectory's endpoint into the odom frame so we
                # can compare it to the lane-change goal. A displacement
                # (dx, dy) in the robot frame, when the robot's yaw in odom is
                # yaw_now, becomes:
                #     odom_dx = dx * cos(yaw) - dy * sin(yaw)
                #     odom_dy = dx * sin(yaw) + dy * cos(yaw)
                # We only need the y component to compute lateral_error.
                dx = float(xs[-1])
                dy = float(ys[-1])
                odom_dy_pred = dx * math.sin(yaw_now) + dy * math.cos(yaw_now)
                predicted_lateral_progress = current_lateral + odom_dy_pred

                lateral_error = abs(active_target_y - predicted_lateral_progress)

                # Heading reward, gated. Final yaw in odom = current + delta.
                # The target is the yaw captured when LANE_CHANGE started,
                # because the "straight road direction" in Gazebo/physical
                # odometry may not be exactly 0 rad. cos(0) == 1 means the car
                # is parallel to the original lane again; cos(pi) == -1 would
                # be pointing backward and is heavily penalized.
                heading_reward = 0.0
                if lateral_error < self.heading_gate:
                    yaw_end_odom = yaw_now + float(thetas[-1])
                    yaw_error = self._angle_error(
                        yaw_end_odom,
                        self.lane_change_start_yaw,
                    )
                    heading_reward = math.cos(yaw_error)

                score = (
                    self.w_clearance * clearance
                    - self.w_lateral * lateral_error
                    + self.w_heading * heading_reward
                    + self.w_velocity * v
                )

                if score > best_score:
                    best_score = score
                    best_v = float(v)
                    best_w = float(w)

        if best_score == -math.inf:
            # Every candidate failed the clearance gate -> hard brake.
            return 0.0, 0.0
        return best_v, best_w

    def _active_target_lane_y(self):
        """Return the lateral target for the current DWA-managed state.

        LANE_CHANGE drives to +target_lane_y to pass the obstacle. LANE_RETURN
        drives back to 0.0, meaning the lateral position captured at the moment
        the lane change began. This is the missing physical recovery step: the
        state machine should not hand control back to vision until the DWA has
        steered the car back near this zero-offset line.
        """
        if self.current_state == LANE_RETURN:
            return 0.0
        return self.target_lane_y

    def _rollout(self, v: float, w: float):
        """Forward-simulate the unicycle model from the robot's origin."""
        xs = np.empty(self.steps, dtype=np.float32)
        ys = np.empty(self.steps, dtype=np.float32)
        thetas = np.empty(self.steps, dtype=np.float32)
        x = y = theta = 0.0
        for k in range(self.steps):
            x += v * math.cos(theta) * self.dt
            y += v * math.sin(theta) * self.dt
            theta += w * self.dt
            xs[k] = x
            ys[k] = y
            thetas[k] = theta
        return xs, ys, thetas

    def _trajectory_clearance(self, xs, ys):
        """Min Euclidean distance from any predicted pose to any LiDAR point."""
        pts = self.obstacle_points
        # Broadcast: (steps, 1, 2) - (1, N, 2) -> (steps, N, 2)
        traj = np.column_stack([xs, ys])[:, None, :]
        obs = pts[None, :, :]
        d2 = np.sum((traj - obs) ** 2, axis=2)
        return float(math.sqrt(d2.min()))

    @staticmethod
    def _angle_error(angle, reference):
        """Smallest signed difference from reference to angle, in radians."""
        raw = angle - reference
        return math.atan2(math.sin(raw), math.cos(raw))

    @staticmethod
    def _format_optional(value):
        if value is None:
            return 'none'
        return f'{value:.3f}'


def main(args=None):
    rclpy.init(args=args)
    node = DwaLocalPlannerNode()
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
