#!/usr/bin/env python3
"""High-level decision/state-machine node for the Quanser QCar 2.

Watches a forward LiDAR cone and moves the vehicle through three states:

    LANE_KEEP    -> the vision pipeline drives via /planning/validated_target_x
    LANE_CHANGE  -> control yields to the Nav2 DWA local planner so it can
                    swerve around the obstacle; vision pauses its EMA width
                    update so the temporary geometry seen during the swerve
                    does not corrupt the trusted lane-width memory.
    LANE_RETURN  -> DWA targets the original lateral offset so the car returns
                    to the starting lane before vision takes back control.

State changes are exposed on /system/current_state at a fixed 10 Hz so the
validation node always sees a fresh value even when the LiDAR scan rate
varies. The 4-second clear-window on the recovery side prevents flapping
when the car is physically alongside the obstacle (cone clear but body still
in the avoidance lane).
"""
import math

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.duration import Duration
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, Float32MultiArray, String


LANE_KEEP = 'LANE_KEEP'
LANE_CHANGE = 'LANE_CHANGE'
LANE_RETURN = 'LANE_RETURN'


class StateMachineNode(Node):
    """LiDAR-driven state machine with explicit return-to-lane recovery."""

    def __init__(self):
        super().__init__('state_machine_node')

        # Required spec parameters.
        self.declare_parameter('safety_distance', 1.5)      # metres
        self.declare_parameter('lidar_fov_cone', 0.4)       # radians (total cone width)

        # Optional knobs. forward_angle_rad lets you cope with a LiDAR that
        # is bolted on rotated (e.g. ribbon cable on the front so the LiDAR's
        # 0-rad mark faces *backwards* -> set forward_angle_rad = math.pi).
        # Default 0.0 assumes the LiDAR's local x-axis already points forward.
        self.declare_parameter('forward_angle_rad', 0.0)
        self.declare_parameter('recovery_delay_sec', 4.0)
        self.declare_parameter('min_lane_change_y', 0.30)
        # require_odom_lane_reach: keep waiting in LANE_CHANGE until odom shows the
        #   car has shifted >= min_lane_change_y before allowing LANE_RETURN. Set
        #   False for the in-controller lateral-offset lane change, whose lateral
        #   travel is driven by vision and where world-frame odom-y is unreliable
        #   on curved track.
        # return_duration_sec: >0 enables curve-safe time-based LANE_RETURN ->
        #   LANE_KEEP (used when no open-loop planner publishes maneuver_complete).
        self.declare_parameter('require_odom_lane_reach', True)
        self.declare_parameter('return_duration_sec', 0.0)
        self.declare_parameter('return_tolerance_y', 0.08)
        self.declare_parameter('return_yaw_tolerance_rad', 0.12)
        self.declare_parameter('return_angular_tolerance', 0.10)
        self.declare_parameter('return_settle_sec', 0.75)
        # One-way lane change: avoid into the adjacent lane and CONTINUE there
        # (no LANE_RETURN). LANE_CHANGE runs open-loop for lane_change_duration_sec
        # (long enough for the S-curve + clearing the obstacle), then hands back
        # to vision lane-keeping. A cooldown stops the just-passed obstacle (or
        # track walls) from immediately re-triggering another change.
        self.declare_parameter('oneway_lane_change', False)
        self.declare_parameter('lane_change_duration_sec', 7.0)
        self.declare_parameter('lane_change_cooldown_sec', 3.0)
        # LiDAR-tracked avoidance: detect + size the obstacle in the front cone,
        # hold the shift while it is in the front OR side sectors (still ahead or
        # alongside), and switch back only once it is neither (i.e. behind/passed).
        # Decisions are LiDAR-driven (front -> beside -> behind -> return); the
        # motion is still the open-loop S-curve planner. track_range_m gates out
        # the far track walls so they are not mistaken for "still beside me".
        self.declare_parameter('lidar_tracked_avoid', False)
        self.declare_parameter('side_sector_min_deg', 30.0)
        self.declare_parameter('side_sector_max_deg', 150.0)
        self.declare_parameter('track_range_m', 1.5)
        self.declare_parameter('pass_confirm_sec', 1.0)
        self.declare_parameter('lane_change_min_sec', 3.5)
        # Obstacle sizing + side selection. The FSM measures the obstacle's
        # lateral centre and width in the front cone, computes how far it must
        # shift to clear it (half the obstacle width + half the car + margin),
        # picks the clear side to dodge toward, and tells the planner the
        # direction (+1 left / -1 right) and the straight-phase duration that
        # produces that shift. lateral_per_straight_sec maps metres-of-shift to
        # seconds-of-straight-phase (calibrated from sim: ~0.16 m/s of shift).
        self.declare_parameter('vehicle_half_width_m', 0.18)
        self.declare_parameter('clearance_margin_m', 0.20)
        self.declare_parameter('min_obstacle_width_m', 0.25)
        self.declare_parameter('lateral_per_straight_sec', 0.16)
        self.declare_parameter('min_straight_dur', 1.2)
        self.declare_parameter('max_straight_dur', 4.5)
        self.declare_parameter('obstacle_depth_tol_m', 0.6)
        self.declare_parameter('prefer_side', 'auto')   # auto | left | right
        # If the obstacle's measured lateral centre is within this band it counts
        # as "centred" and we dodge to default_side, instead of letting a cm of
        # measurement noise flip the chosen side.
        self.declare_parameter('center_deadband_m', 0.20)
        self.declare_parameter('default_side', 'left')  # side for a centred obstacle
        # How far (m) the car must travel PAST the obstacle's position before it
        # switches back. Judged from odometry, not LiDAR sectors.
        self.declare_parameter('pass_clearance_m', 0.8)
        self.declare_parameter('lane_change_params_topic', '/qcar2/control/lane_change_params')
        self.declare_parameter('publish_rate_hz', 10.0)
        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('odom_topic', '/odom')
        self.declare_parameter('state_topic', '/system/current_state')
        self.declare_parameter('maneuver_complete_topic', '/qcar2/control/maneuver_complete')

        self.safety_distance = float(self.get_parameter('safety_distance').value)
        self.lidar_fov_cone = float(self.get_parameter('lidar_fov_cone').value)
        self.forward_angle_rad = float(self.get_parameter('forward_angle_rad').value)
        self.recovery_delay_sec = float(self.get_parameter('recovery_delay_sec').value)
        self.min_lane_change_y = float(self.get_parameter('min_lane_change_y').value)
        self.require_odom_lane_reach = bool(self.get_parameter('require_odom_lane_reach').value)
        self.return_duration_sec = float(self.get_parameter('return_duration_sec').value)
        self.return_tolerance_y = float(self.get_parameter('return_tolerance_y').value)
        self.return_yaw_tolerance_rad = float(self.get_parameter('return_yaw_tolerance_rad').value)
        self.return_angular_tolerance = float(self.get_parameter('return_angular_tolerance').value)
        self.return_settle_sec = float(self.get_parameter('return_settle_sec').value)
        self.oneway_lane_change = bool(self.get_parameter('oneway_lane_change').value)
        self.lane_change_duration_sec = float(self.get_parameter('lane_change_duration_sec').value)
        self.lane_change_cooldown_sec = float(self.get_parameter('lane_change_cooldown_sec').value)
        self.lidar_tracked_avoid = bool(self.get_parameter('lidar_tracked_avoid').value)
        self.side_sector_min_deg = float(self.get_parameter('side_sector_min_deg').value)
        self.side_sector_max_deg = float(self.get_parameter('side_sector_max_deg').value)
        self.track_range_m = float(self.get_parameter('track_range_m').value)
        self.pass_confirm_sec = float(self.get_parameter('pass_confirm_sec').value)
        self.lane_change_min_sec = float(self.get_parameter('lane_change_min_sec').value)
        self.vehicle_half_width_m = float(self.get_parameter('vehicle_half_width_m').value)
        self.clearance_margin_m = float(self.get_parameter('clearance_margin_m').value)
        self.min_obstacle_width_m = max(
            0.0,
            float(self.get_parameter('min_obstacle_width_m').value),
        )
        self.lateral_per_straight_sec = max(
            1e-3,
            float(self.get_parameter('lateral_per_straight_sec').value),
        )
        self.min_straight_dur = float(self.get_parameter('min_straight_dur').value)
        self.max_straight_dur = float(self.get_parameter('max_straight_dur').value)
        self.obstacle_depth_tol_m = float(self.get_parameter('obstacle_depth_tol_m').value)
        self.prefer_side = str(self.get_parameter('prefer_side').value).strip().lower()
        self.center_deadband_m = float(self.get_parameter('center_deadband_m').value)
        self.default_side = str(self.get_parameter('default_side').value).strip().lower()
        self.pass_clearance_m = float(self.get_parameter('pass_clearance_m').value)

        if self.safety_distance <= 0.0:
            raise ValueError(f'safety_distance must be > 0, got {self.safety_distance}')
        if self.lidar_fov_cone <= 0.0:
            raise ValueError(f'lidar_fov_cone must be > 0, got {self.lidar_fov_cone}')

        # Spec-required class variables.
        self.current_state = LANE_KEEP
        self.clear_time = None  # rclpy.Time | None
        self.odom_y = None
        self.odom_yaw = None
        self.odom_wz = None
        self.lane_change_start_y = None
        self.lane_change_start_yaw = None
        self.return_settle_time = None
        # Maneuver planner can directly signal it has landed the LANE_RETURN
        # quintic at the target lateral offset, so the FSM does not have to
        # infer recovery from odom drift on a curving track.
        self.maneuver_complete_signal = False
        # One-way / LiDAR-tracked lane-change bookkeeping.
        self.change_start_time = None
        self.cooldown_until = None
        self.pass_clear_time = None
        self.return_start_time = None
        self.odom_x = None
        # Obstacle world position + travel heading captured at detection, so
        # "passed" can be judged from odometry progress (robust) instead of the
        # LiDAR sectors going clear (which happens far too early as the car yaws).
        self.obs_world_x = None
        self.obs_world_y = None
        self.change_yaw = 0.0

        self.state_pub = self.create_publisher(
            String,
            self.get_parameter('state_topic').value,
            10,
        )
        # [direction_sign(+1 left / -1 right), straight_phase_duration_sec]
        self.lc_params_pub = self.create_publisher(
            Float32MultiArray,
            self.get_parameter('lane_change_params_topic').value,
            10,
        )
        self._meas = None  # latest front-obstacle measurement
        self.create_subscription(
            LaserScan,
            self.get_parameter('scan_topic').value,
            self._on_scan,
            10,
        )
        self.create_subscription(
            Odometry,
            self.get_parameter('odom_topic').value,
            self._on_odom,
            10,
        )
        self.create_subscription(
            Bool,
            self.get_parameter('maneuver_complete_topic').value,
            self._on_maneuver_complete,
            10,
        )

        # Continuous broadcasting timer. Runs even when no scans arrive so the
        # validation node sees a heartbeat and never drifts on a stale state.
        rate = float(self.get_parameter('publish_rate_hz').value)
        self.create_timer(1.0 / rate, self._broadcast_state)

        self.get_logger().info(
            f'State machine ready: safety_distance={self.safety_distance} m, '
            f'lidar_fov_cone={self.lidar_fov_cone} rad, '
            f'forward_angle_rad={self.forward_angle_rad} rad, '
            f'recovery_delay={self.recovery_delay_sec} s, '
            f'min_lane_change_y={self.min_lane_change_y} m, '
            f'return_tolerance_y={self.return_tolerance_y} m, '
            f'return_yaw_tolerance={self.return_yaw_tolerance_rad} rad, '
            f'return_angular_tolerance={self.return_angular_tolerance} rad/s, '
            f'return_settle={self.return_settle_sec} s'
        )

    # ---- publishers ------------------------------------------------------

    def _broadcast_state(self):
        """Heartbeat publish at the configured rate."""
        msg = String()
        msg.data = self.current_state
        self.state_pub.publish(msg)

    # ---- scan handling ---------------------------------------------------

    def _on_scan(self, msg: LaserScan):
        """Detect a forward obstacle and feed the state machine."""
        if self.lidar_tracked_avoid:
            self._on_scan_tracked(msg)
            return
        cone_indices = self._front_cone_indices(msg)
        if cone_indices.size == 0:
            # No valid indices fall inside the configured cone. Treat as "no
            # observation" rather than "clear" to avoid spurious
            # LANE_CHANGE -> LANE_KEEP transitions when the sensor itself is
            # misconfigured.
            return

        ranges = np.asarray(msg.ranges, dtype=np.float32)
        cone = ranges[cone_indices]

        # A reading is *valid* only if it's a finite, positive number inside
        # the sensor's declared range. inf/nan signal "no return", 0.0 is a
        # common driver placeholder for "too close / invalid".
        valid = cone[
            np.isfinite(cone)
            & (cone > 0.0)
            & (cone >= float(msg.range_min))
            & (cone <= float(msg.range_max))
        ]
        obstacle_present = bool(valid.size and valid.min() < self.safety_distance)

        self._update_state(obstacle_present)

    def _on_maneuver_complete(self, msg: Bool):
        """Maneuver planner says it has finished its current quintic curve.

        Only treated as a recovery trigger while we are in LANE_RETURN; any
        completion signal that arrives in another state is ignored, so a stale
        Bool from a previous lane change cannot prematurely end the next one.
        """
        if msg.data:
            self.maneuver_complete_signal = True

    def _on_odom(self, msg: Odometry):
        """Cache lateral position, yaw, and yaw rate for recovery gating."""
        q = msg.pose.pose.orientation
        self.odom_x = float(msg.pose.pose.position.x)
        self.odom_y = float(msg.pose.pose.position.y)
        self.odom_yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z),
        )
        self.odom_wz = float(msg.twist.twist.angular.z)

    def _front_cone_indices(self, scan: LaserScan):
        """Return scan.ranges indices that lie inside the forward cone.

        Math:
            A LaserScan stores measurements in scan.ranges[i] taken at angle
                theta_i = scan.angle_min + i * scan.angle_increment.

            For a normally-mounted QCar LiDAR, forward_angle_rad = 0.0 and the
            desired cone is:

                -lidar_fov_cone/2 < theta_i < +lidar_fov_cone/2

            If the LiDAR is mounted backwards, set forward_angle_rad = pi.
            A naive angle-to-index calculation breaks at +/-pi because the
            interval [pi - fov/2, pi + fov/2] wraps across the scan boundary.

            To make both cases work, compute each beam's wrapped angular error
            relative to the physical forward direction:

                raw_error     = theta_i - forward_angle_rad
                wrapped_error = atan2(sin(raw_error), cos(raw_error))

            atan2(sin, cos) maps any angle to [-pi, +pi], so beams just below
            -pi and just above +pi both become near zero when forward is pi.
            The selected indices are those strictly inside the cone:

                abs(wrapped_error) < lidar_fov_cone / 2

        This function returns explicit indices instead of one contiguous slice
        because a backwards-mounted LiDAR can produce two valid index blocks at
        the beginning and end of a [-pi, +pi] scan.
        """
        if not math.isfinite(scan.angle_increment) or scan.angle_increment == 0.0:
            return np.array([], dtype=np.int64)

        n = len(scan.ranges)
        if n == 0:
            return np.array([], dtype=np.int64)

        half = 0.5 * self.lidar_fov_cone
        indices = np.arange(n, dtype=np.int64)
        angles = scan.angle_min + indices.astype(np.float64) * scan.angle_increment
        raw_error = angles - self.forward_angle_rad
        wrapped_error = np.arctan2(np.sin(raw_error), np.cos(raw_error))
        return indices[np.abs(wrapped_error) < half]

    # ---- state machine ---------------------------------------------------

    def _update_state(self, obstacle_present: bool):
        """Apply LANE_KEEP -> LANE_CHANGE -> LANE_RETURN -> LANE_KEEP."""
        if self.oneway_lane_change:
            self._update_oneway(obstacle_present)
            return
        if obstacle_present:
            if self.current_state in (LANE_KEEP, LANE_RETURN):
                self._snapshot_lane_change_start()
                self.current_state = LANE_CHANGE
                self.return_settle_time = None
                self.maneuver_complete_signal = False
                self.get_logger().warn(
                    'Obstacle Detected! Yielding to DWA Local Planner.'
                )
            # Any in-cone return restarts the recovery countdown, even if we
            # were already in LANE_CHANGE -- otherwise a flickering return
            # could let the timer expire while the obstacle is still present.
            self.clear_time = None
            return

        # No obstacle this scan. Only the LANE_CHANGE state needs to count;
        # LANE_KEEP has nothing to recover from.
        if self.current_state == LANE_KEEP:
            return

        now = self.get_clock().now()

        if self.current_state == LANE_CHANGE:
            if self.clear_time is None:
                self.clear_time = now
                return

            elapsed_sec = (now - self.clear_time).nanoseconds * 1e-9
            if elapsed_sec >= self.recovery_delay_sec:
                if (self.require_odom_lane_reach and self.odom_y is not None
                        and not self._has_reached_avoidance_lane()):
                    # The cone is clear, but the vehicle has not actually
                    # reached the adjacent lane yet. Keep DWA in LANE_CHANGE so
                    # it continues targeting +target_lane_y instead of
                    # declaring recovery too early.
                    return

                self.current_state = LANE_RETURN
                self.clear_time = now
                self.return_settle_time = None
                self.maneuver_complete_signal = False
                self.get_logger().info(
                    'Obstacle Cleared: Returning to Original Lane.'
                )
            return

        if self.current_state == LANE_RETURN:
            # Curve-safe completion: when no open-loop planner is present (the
            # lateral-offset lane change runs entirely in the controller),
            # recover purely on elapsed time in LANE_RETURN. clear_time is set
            # to "now" on entry to LANE_RETURN, so it measures time-in-state.
            if self.return_duration_sec > 0.0 and self.clear_time is not None:
                elapsed_sec = (now - self.clear_time).nanoseconds * 1e-9
                if elapsed_sec >= self.return_duration_sec:
                    self.current_state = LANE_KEEP
                    self.clear_time = None
                    self.lane_change_start_y = None
                    self.lane_change_start_yaw = None
                    self.return_settle_time = None
                    self.maneuver_complete_signal = False
                    self.get_logger().info(
                        'Recovery Complete (timed): Resuming Vision Tracking.'
                    )
                    return
            # Fast path: the maneuver planner has explicitly told us it has
            # finished its quintic curve. Trust that signal -- the planner
            # knows when the curve has expired better than we can guess from
            # odom drift on a curving track.
            if self.maneuver_complete_signal:
                self.current_state = LANE_KEEP
                self.clear_time = None
                self.lane_change_start_y = None
                self.lane_change_start_yaw = None
                self.return_settle_time = None
                self.maneuver_complete_signal = False
                self.get_logger().info(
                    'Recovery Complete (planner signal): Resuming Vision Tracking.'
                )
                return
            if self._is_recovered_at_start_lane(now):
                self.current_state = LANE_KEEP
                self.clear_time = None
                self.lane_change_start_y = None
                self.lane_change_start_yaw = None
                self.return_settle_time = None
                self.get_logger().info(
                    'Recovery Complete: Resuming Vision Tracking.'
                )
                return

            # If odometry disappears, avoid getting stuck forever in
            # LANE_RETURN. The DWA also depends on odometry for precise return
            # targeting, so after one clear window we hand control back to the
            # vision safety timeout path.
            if self.odom_y is None and self.clear_time is not None:
                elapsed_sec = (now - self.clear_time).nanoseconds * 1e-9
                if elapsed_sec >= self.recovery_delay_sec:
                    self.current_state = LANE_KEEP
                    self.clear_time = None
                    self.lane_change_start_y = None
                    self.lane_change_start_yaw = None
                    self.return_settle_time = None
                    self.get_logger().warn(
                        'Recovery Complete without odometry: Resuming Vision Tracking.'
                    )
            return

    def _update_oneway(self, obstacle_present: bool):
        """One-way avoidance: change into the adjacent lane and stay there.

        LANE_KEEP --(obstacle)--> LANE_CHANGE --(timer)--> LANE_KEEP.
        There is no LANE_RETURN: the open-loop S-curve shifts the car over and,
        after lane_change_duration_sec (S-curve + enough creep to clear the
        obstacle), control hands back to vision lane-keeping in the new lane.
        A cooldown after each change stops the just-passed obstacle or the track
        walls from immediately re-triggering another shift.
        """
        now = self.get_clock().now()

        if self.current_state == LANE_CHANGE:
            if self.change_start_time is None:
                self.change_start_time = now
                return
            elapsed = (now - self.change_start_time).nanoseconds * 1e-9
            if elapsed >= self.lane_change_duration_sec:
                self.current_state = LANE_KEEP
                self.change_start_time = None
                self.cooldown_until = now + Duration(seconds=self.lane_change_cooldown_sec)
                self.get_logger().info(
                    'Lane change complete: continuing in the new lane.'
                )
            return

        # LANE_KEEP: trigger a single change when an obstacle is ahead and we are
        # not in the post-change cooldown.
        if obstacle_present:
            if self.cooldown_until is not None and now < self.cooldown_until:
                return
            self.current_state = LANE_CHANGE
            self.change_start_time = now
            self.get_logger().warn(
                'Obstacle Detected! Changing lane (one-way, no return).'
            )

    def _band_min_range(self, scan, abs_lo_rad, abs_hi_rad):
        """Min valid range among beams whose |angle-to-forward| is in [lo, hi)."""
        if not math.isfinite(scan.angle_increment) or scan.angle_increment == 0.0:
            return math.inf
        n = len(scan.ranges)
        if n == 0:
            return math.inf
        idx = np.arange(n, dtype=np.int64)
        angles = scan.angle_min + idx.astype(np.float64) * scan.angle_increment
        err = np.arctan2(np.sin(angles - self.forward_angle_rad),
                         np.cos(angles - self.forward_angle_rad))
        ae = np.abs(err)
        sel = (ae >= abs_lo_rad) & (ae < abs_hi_rad)
        r = np.asarray(scan.ranges, dtype=np.float32)[sel]
        valid = r[np.isfinite(r) & (r > 0.0)
                  & (r >= float(scan.range_min)) & (r <= float(scan.range_max))]
        return float(valid.min()) if valid.size else math.inf

    def _signed_band_min_range(self, scan, lo_rad, hi_rad):
        """Min valid range among beams whose signed angle-to-forward is in [lo,hi)."""
        if not math.isfinite(scan.angle_increment) or scan.angle_increment == 0.0:
            return math.inf
        n = len(scan.ranges)
        if n == 0:
            return math.inf
        idx = np.arange(n, dtype=np.int64)
        angles = scan.angle_min + idx.astype(np.float64) * scan.angle_increment
        err = np.arctan2(np.sin(angles - self.forward_angle_rad),
                         np.cos(angles - self.forward_angle_rad))
        sel = (err >= lo_rad) & (err < hi_rad)
        r = np.asarray(scan.ranges, dtype=np.float32)[sel]
        valid = r[np.isfinite(r) & (r > 0.0)
                  & (r >= float(scan.range_min)) & (r <= float(scan.range_max))]
        return float(valid.min()) if valid.size else math.inf

    def _measure_front_obstacle(self, scan, front_min):
        """Lateral centre + width (metres) of the front obstacle's near face.

        +lateral is left. Uses front-cone beams that land on the near face of
        the obstacle (range within obstacle_depth_tol of the closest return).
        """
        half = 0.5 * self.lidar_fov_cone
        n = len(scan.ranges)
        idx = np.arange(n, dtype=np.int64)
        angles = scan.angle_min + idx.astype(np.float64) * scan.angle_increment
        err = np.arctan2(np.sin(angles - self.forward_angle_rad),
                         np.cos(angles - self.forward_angle_rad))
        r = np.asarray(scan.ranges, dtype=np.float32)
        sel = ((np.abs(err) < half) & np.isfinite(r) & (r > 0.0)
               & (r >= float(scan.range_min)) & (r <= float(scan.range_max))
               & (r <= front_min + self.obstacle_depth_tol_m))
        if np.count_nonzero(sel) < 2:
            return 0.0, 0.0
        lat = r[sel] * np.sin(err[sel])           # +left
        return float(np.mean(lat)), float(lat.max() - lat.min())

    def _on_scan_tracked(self, msg: LaserScan):
        """LiDAR sector tracking: front -> beside -> behind -> switch back."""
        half = 0.5 * self.lidar_fov_cone
        front_min = self._band_min_range(msg, 0.0, half)
        side_lo = math.radians(self.side_sector_min_deg)
        side_hi = math.radians(self.side_sector_max_deg)
        side_min = self._band_min_range(msg, side_lo, side_hi)
        left_min = self._signed_band_min_range(msg, side_lo, side_hi)
        right_min = self._signed_band_min_range(msg, -side_hi, -side_lo)
        front_blocked = front_min < self.safety_distance
        beside = side_min < self.track_range_m
        if front_blocked:
            center, width = self._measure_front_obstacle(msg, front_min)
            self._meas = (center, width, left_min, right_min)
        self._update_lidar_tracked(front_blocked, beside, front_min)

    def _decide_dodge(self):
        """Pick direction (+1 left / -1 right) and straight-phase duration."""
        center, width, left_min, right_min = self._meas or (0.0, 0.25, math.inf, math.inf)
        width = max(float(width), self.min_obstacle_width_m)
        base = 0.5 * width + self.vehicle_half_width_m + self.clearance_margin_m
        need_left = max(0.0, base + center)    # +center: obstacle toward left
        need_right = max(0.0, base - center)
        left_wall = left_min < self.track_range_m
        right_wall = right_min < self.track_range_m

        default_dir = -1.0 if self.default_side == 'right' else +1.0

        if self.prefer_side == 'left':
            direction = +1.0
        elif self.prefer_side == 'right':
            direction = -1.0
        elif left_wall and not right_wall:
            direction = -1.0                      # wall on left -> go right
        elif right_wall and not left_wall:
            direction = +1.0                      # wall on right -> go left
        elif center > self.center_deadband_m:
            direction = -1.0                      # obstacle clearly LEFT -> go right
        elif center < -self.center_deadband_m:
            direction = +1.0                      # obstacle clearly RIGHT -> go left
        else:
            direction = default_dir               # centred -> default side (left)

        shift = need_left if direction > 0 else need_right

        straight = shift / self.lateral_per_straight_sec
        straight = max(self.min_straight_dur, min(self.max_straight_dur, straight))
        return direction, straight, shift

    def _update_lidar_tracked(self, front_blocked, beside, front_min):
        """Avoid + return driven by which LiDAR sector the obstacle is in."""
        now = self.get_clock().now()

        if self.current_state == LANE_KEEP:
            if front_blocked:
                if self.cooldown_until is not None and now < self.cooldown_until:
                    return
                direction, straight, shift = self._decide_dodge()
                self.lc_params_pub.publish(
                    Float32MultiArray(data=[float(direction), float(straight)])
                )
                self.current_state = LANE_CHANGE
                self.change_start_time = now
                self.pass_clear_time = None
                self.maneuver_complete_signal = False
                # Capture the obstacle's world position + our heading now, so we
                # can tell when we have physically driven past it.
                if self.odom_x is not None:
                    self.obs_world_x = self.odom_x + front_min * math.cos(self.odom_yaw)
                    self.obs_world_y = self.odom_y + front_min * math.sin(self.odom_yaw)
                    self.change_yaw = self.odom_yaw
                else:
                    self.obs_world_x = None
                center, width = (self._meas[0], self._meas[1]) if self._meas else (0.0, 0.0)
                side = 'LEFT' if direction > 0 else 'RIGHT'
                self.get_logger().warn(
                    f'Obstacle @ {front_min:.2f} m (width~{width:.2f} m, '
                    f'centre~{center:+.2f} m): dodging {side}, shift~{shift:.2f} m '
                    f'(straight {straight:.1f} s).'
                )
            return

        if self.current_state == LANE_CHANGE:
            # Safety: if something is dead ahead again, keep avoiding.
            if front_blocked:
                self.pass_clear_time = None
                return
            # Let the S-curve develop before judging "passed".
            elapsed = ((now - self.change_start_time).nanoseconds * 1e-9
                       if self.change_start_time is not None else 0.0)
            if elapsed < self.lane_change_min_sec:
                return
            if self.oneway_lane_change and self.maneuver_complete_signal:
                self._finish_oneway_lane_change(
                    now,
                    'Lane-change maneuver complete: staying in the new lane.',
                )
                return
            # "Passed" = we have physically driven past the obstacle (odometry).
            # Falls back to the LiDAR beside-clear test only if odom is missing.
            if self.obs_world_x is not None and self.odom_x is not None:
                progress = (
                    (self.odom_x - self.obs_world_x) * math.cos(self.change_yaw)
                    + (self.odom_y - self.obs_world_y) * math.sin(self.change_yaw)
                )
                passed = progress > self.pass_clearance_m
            else:
                passed = not beside
            if not passed:
                self.pass_clear_time = None
                return
            if self.pass_clear_time is None:
                self.pass_clear_time = now
                return
            if (now - self.pass_clear_time).nanoseconds * 1e-9 >= self.pass_confirm_sec:
                if self.oneway_lane_change:
                    self._finish_oneway_lane_change(
                        now,
                        'Obstacle is behind (odom): staying in the new lane.',
                    )
                    return
                self.current_state = LANE_RETURN
                self.return_start_time = now
                self.pass_clear_time = None
                self.maneuver_complete_signal = False
                self.get_logger().info(
                    'Obstacle is behind (odom): switching back to the original lane.'
                )
            return

        if self.current_state == LANE_RETURN:
            elapsed = ((now - self.return_start_time).nanoseconds * 1e-9
                       if self.return_start_time is not None else 0.0)
            if self.maneuver_complete_signal or (
                    self.return_duration_sec > 0.0 and elapsed >= self.return_duration_sec):
                self.current_state = LANE_KEEP
                self.cooldown_until = now + Duration(seconds=self.lane_change_cooldown_sec)
                self.return_start_time = None
                self.maneuver_complete_signal = False
                self.get_logger().info(
                    'Back in the original lane: resuming vision lane-keeping.'
                )
            return

    def _finish_oneway_lane_change(self, now, log_message):
        self.current_state = LANE_KEEP
        self.cooldown_until = now + Duration(seconds=self.lane_change_cooldown_sec)
        self.change_start_time = None
        self.pass_clear_time = None
        self.return_start_time = None
        self.maneuver_complete_signal = False
        self.get_logger().info(log_message)

    def _snapshot_lane_change_start(self):
        if self.odom_y is not None and self.lane_change_start_y is None:
            self.lane_change_start_y = self.odom_y
        if self.odom_yaw is not None and self.lane_change_start_yaw is None:
            self.lane_change_start_yaw = self.odom_yaw

    def _is_back_at_start_lane(self):
        if self.odom_y is None or self.lane_change_start_y is None:
            return False
        return abs(self.odom_y - self.lane_change_start_y) <= self.return_tolerance_y

    def _is_recovered_at_start_lane(self, now):
        """Return True only when lateral position and heading are both settled.

        The previous recovery gate checked only lateral ``y``. That let the
        state machine hand control back to the vision/PID lane follower while
        the car was still yawed and physically turning across the lane lines.
        In that pose, the camera sees the middle/right markings at a strange
        angle and the histogram can split the wrong line pair again.

        We therefore require three things continuously for return_settle_sec:

            1. Lateral position is back near the lane-change start y.
            2. Yaw is back near the lane-change start yaw, not hardcoded 0 rad.
            3. Yaw rate is small, meaning the car is no longer actively rotating.
        """
        if not self._is_back_at_start_lane():
            self.return_settle_time = None
            return False
        if self.odom_yaw is None or self.lane_change_start_yaw is None:
            self.return_settle_time = None
            return False
        if self.odom_wz is None:
            self.return_settle_time = None
            return False

        yaw_error = abs(self._angle_error(self.odom_yaw, self.lane_change_start_yaw))
        yaw_rate = abs(self.odom_wz)
        is_settled = (
            yaw_error <= self.return_yaw_tolerance_rad
            and yaw_rate <= self.return_angular_tolerance
        )
        if not is_settled:
            self.return_settle_time = None
            return False

        if self.return_settle_time is None:
            self.return_settle_time = now
            return self.return_settle_sec <= 0.0

        elapsed_sec = (now - self.return_settle_time).nanoseconds * 1e-9
        return elapsed_sec >= self.return_settle_sec

    def _has_reached_avoidance_lane(self):
        if self.odom_y is None or self.lane_change_start_y is None:
            return False
        return abs(self.odom_y - self.lane_change_start_y) >= self.min_lane_change_y

    @staticmethod
    def _angle_error(angle, reference):
        """Smallest signed difference from reference to angle, in radians."""
        raw = angle - reference
        return math.atan2(math.sin(raw), math.cos(raw))


def main(args=None):
    rclpy.init(args=args)
    node = StateMachineNode()
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
