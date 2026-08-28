#!/usr/bin/env python3

import math

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32, Bool, String, Int32

from qcar_science_night_pkg.lidar_sector_analyzer import LidarSectorAnalyzer
from qcar_science_night_pkg.overtake_state_machine import OvertakeStateMachine
from qcar_science_night_pkg.overtake_types import OvertakeDecision
from qcar_science_night_pkg.lidar_debug_file_logger import LidarDebugFileLogger


class LidarOvertakeNode(Node):
    def __init__(self):
        super().__init__("lidar_overtake_node")

        self.last_drive_state = None
        self.allow_overtake = False
        self.last_scan_time = None
        self.safety_armed = False
        self.sensor_timeout_sec = 0.7

        self.current_path_idx = 0
        self.yaw_stable = False

        # After this idx, normal overtaking is disabled.
        # Front stop/emergency safety still remains active.
        # Was 1600 -- stale from a longer, earlier recording. The current
        # trajectory (track_run_cartographer_final.npy) is only 540 points,
        # so that check never actually triggered -- no real protection near
        # the trajectory end (Issue 15's idx 525-539 zone). Fixed relative
        # to the actual length, with margin.
        self.disable_obstacle_after_idx = 520

        # Distance behavior
        # On straights, look farther ahead.
        # Needs real runway above overtake_start_min_distance (0.85) -- at
        # 0.90 there was only ~0.05m between "detected" and "too close to
        # overtake", which the debounce + closing speed ate before a
        # decision could land above the overtake floor. Result: it kept
        # collapsing into WAIT_FOR_CLEAR instead of ever overtaking, even on
        # straights. Raised to give the confirm counters + decision time to
        # actually complete while still above overtake_start_min_distance.
        # NOT scaled with v_max -- was scaled to 1.72/1.09 for the v_max=0.75
        # pass, but confirmed live that a real static object (likely the
        # desk/cabinet near the loop-seam section of the track) sits at
        # front=1.30-1.69m, which the original 1.10m threshold safely
        # ignored but the scaled 1.72m did not, causing spurious
        # obstacle_ahead detections and repeated OVERTAKE_LEFT triggers
        # right at the already-fragile loop-seam transition. Physically
        # unnecessary anyway: real stopping distance from v_max=0.75 is
        # only ~0.35m, so the original value already had huge margin.
        # Reverted, same reasoning as the curve-context values below.
        #
        # Lowered 1.10 -> 1.00 (2026-08-28, explicit request: "front gap
        # too big"). NOT lowered to 0.90 -- that's the exact value the
        # comment above documents as already having failed (only 0.05m
        # runway above overtake_start_min_distance=0.85, collapsed into
        # permanent WAIT_FOR_CLEAR). 1.00 keeps 0.15m runway -- 3x that
        # failed margin -- while still being a real reduction. The 2026-
        # 08-28 V2V-preferred commit-distance fix (see enough_distance_to_
        # overtake below) may make tighter values more viable than when
        # 0.90 was originally tried, but that wasn't re-tested -- if 1.00
        # still isn't tight enough, retest 0.90 deliberately rather than
        # assume it's now safe.
        self.front_stop_straight_m = 1.00
        self.emergency_stop_straight_m = 0.70

        # On curves, look less far ahead because the car is turning.
        # NOT scaled with v_max like the straight-line distances above --
        # confirmed live that this track has a corner with a real wall at
        # front_min=0.48-0.60m, and actual curve speed is capped by
        # v_curve_min/curvature (~0.4-0.44 m/s measured) regardless of
        # v_max, so it never needed v_max-scale margin here. Scaling these
        # off v_max was the bug: physically-required stopping margin at the
        # real curve speed is ~0.12m (v^2/2*max_decel), so the original
        # values already had ample headroom -- reverted to them.
        self.front_stop_curve_m = 0.45
        self.emergency_stop_curve_m = 0.65

        # Overtake only starts if obstacle is far enough.
        # If obstacle is closer than hard_stop_front_distance, stop.
        # Must stay comfortably above emergency_stop_straight_m (0.70) --
        # the state machine checks status.emergency before it ever considers
        # overtaking, so if this value is at/below the emergency threshold,
        # a closing obstacle reaches EMERGENCY_STOP before an overtake can
        # ever be committed to. This was overlapping (0.60 < 0.70), which is
        # exactly why overtakes were losing the race to emergency stops.
        # Reverted with front_stop_straight_m above -- must stay below it
        # with real margin (Issue 14) or overtaking deadlocks again.
        self.overtake_start_min_distance = 0.85

        # Commit margin + debounce (added 2026-08-28): follow_target_gap_m
        # on the V2V side was just lowered to 0.8, only 0.05m below this
        # 0.85 floor -- comfortably inside known localization noise (this
        # session repeatedly found AMCL/position jitter on both robots).
        # enough_distance_to_overtake below used to be a raw instantaneous
        # threshold crossing with NO debounce, unlike obstacle_ahead/
        # left_clear which both go through confirm-counters -- so a single
        # noisy sample crossing 0.85 could commit to an overtake on noise.
        # Deliberately NOT touching overtake_start_min_distance itself
        # (Issue 14's hard-won tuning, also reused as v2v_overtake_detect_min
        # above) -- this adds a separate, additive margin + debounce layer
        # around ITS USE for the commit decision specifically, so it can
        # only make triggering an overtake harder, never change what
        # emergency/hard_stop/V2V-detection already rely on.
        self.overtake_commit_margin_m = 0.10   # commit floor = 0.85 + 0.10 = 0.95
        self.overtake_distance_confirm_required = 3
        self.overtake_distance_ok_counter = 0

        # Proactive following-distance speed cap (added 2026-08-27, in
        # response to: QCar2 closed on ROSbot3 at full speed and hard-
        # braked right at front_stop/hard_stop, over and over, rather than
        # settling into a steady gap it could actually overtake from. This
        # is purely additive — a smooth ramp published on its own topic,
        # consumed by path_mpc as `min(target_v, cap)`. It never touches
        # the state machine's own stop/overtake thresholds above, so the
        # existing hard-won tuning there (Issues 14-16) is untouched; this
        # only smooths the approach BEFORE those thresholds are reached.
        # Ceiling is a fixed cap value, not the real v_max (this node
        # doesn't know it) — set comfortably above any speed QCar2 is
        # actually tuned to, so at/beyond follow_start_distance the cap is
        # a functional no-op rather than a second, possibly-wrong "full
        # speed" guess that could clip a legitimately faster curve target.
        self.follow_start_distance = 2.0    # ramp begins here
        self.follow_min_speed = 0.20        # crawl speed right at overtake_start_min_distance
        self.follow_full_speed = 1.0        # ramp ceiling; safely above any real v_max in use
        # NOT scaled -- applied context-blind (both curve and straight), so
        # it inherits the same wall-proximity problem as the curve values
        # above. Reverted for the same reason.
        self.hard_stop_front_distance = 0.40

        # V2V-assisted obstacle detection (added 2026-08-27): on curves,
        # front_stop_curve_m (0.45) means obstacle_ahead goes False beyond
        # 0.45m, so the state machine never even considers overtaking --
        # QCar2 just creeps behind ROSbot3 at the follow-cap's low speed
        # forever, never committing to OVERTAKE_LEFT, because LiDAR alone
        # can't see far enough through the turn to confirm "obstacle ahead."
        # /v2v/gap is a real along-path distance and doesn't suffer that
        # curve-blindness, so it fills exactly this gap.
        #
        # Gated to [overtake_start_min_distance, follow_start_distance) =
        # [0.85, 2.0)m. The floor is a safety property, not a tuning choice:
        # it guarantees the front_min V2V ever supplies is >= 0.85m, safely
        # above hard_stop_front_distance (0.40), so V2V data can NEVER by
        # itself trigger the hard-stop path below. status.emergency and
        # left/right_clear stay exclusively LiDAR's real-time sensing --
        # V2V only ever ADDS an obstacle_ahead detection LiDAR's curve-
        # limited box would otherwise miss, never removes or overrides one,
        # and never touches the acute close-range safety layers.
        self.v2v_overtake_detect_min = self.overtake_start_min_distance
        self.v2v_overtake_detect_max = self.follow_start_distance

        self.v2v_gap = float("nan")
        self.v2v_on_path = False
        self.v2v_alive = False

        self.analyzer = LidarSectorAnalyzer(
            front_offset_deg=180.0,
            max_range=2.0,
            min_range=0.05,
            lane_width=0.43,

            # Keep analyzer boxes large enough.
            # We reduce the effective distance later based on curve/straight.
            front_x_min=0.10,
            front_x_max=1.35,

            # LidarSectorAnalyzer's own obstacle_ahead gate (defaults to
            # 0.75 if left unset) -- front_stop_straight_m/front_stop_curve_m
            # below only ever *shrink* this after the fact, they can't raise
            # it. Left at the 0.75 default, it silently capped obstacle_ahead
            # below overtake_start_min_distance (0.85), making
            # "confirmed obstacle" and "far enough to overtake" mutually
            # exclusive -- overtaking could never trigger, at any distance.
            overtake_start_distance=1.10,

            side_x_min=0.10,
            side_x_max=0.70,

            emergency_x_min=0.03,
            emergency_x_max=0.70,

            # Keep low so legs/feet/thin objects are not missed.
            min_front_points=1,
            min_side_points=1,
            min_emergency_points=1,
        )

        self.state_machine = OvertakeStateMachine(
            # Widened 0.55 -> 0.70 (2026-08-28, explicit report: QCar2
            # repeatedly stalled mid-overtake against a static ROSbot3,
            # frozen idx + continuous depth_emergency stops, consistent
            # with insufficient lateral clearance during the swerve. NOT
            # verified against actual track/lane width -- confirm live
            # that QCar2 stays on the drivable surface at this offset.
            overtake_offset=0.70,
            obstacle_confirm_required=2,
            left_clear_confirm_required=2,
            no_obstacle_confirm_required=5,
            right_clear_confirm_required=3,
            return_confirm_required=10,
            min_overtake_steps=70,
        )

        self.debug_logger = LidarDebugFileLogger(
            "/home/nvidia/ros2_ws_sami/lidar_overtake_debug_log.csv"
        )

        self.offset_pub = self.create_publisher(Float32, "/avoidance_offset", 10)
        self.speed_cap_pub = self.create_publisher(Float32, "/lidar_speed_cap", 10)
        self.motion_pub = self.create_publisher(Bool, "/lidar_motion_safe", 10)
        self.state_pub = self.create_publisher(String, "/drive_state", 10)

        self.sound_pub = self.create_publisher(
            String,
            "/qcar2/sound_event",
            10,
        )

        self.scan_sub = self.create_subscription(
            LaserScan,
            "/scan",
            self.scan_callback,
            10,
        )

        self.allow_overtake_sub = self.create_subscription(
            Bool,
            "/allow_overtake",
            self.allow_overtake_callback,
            10,
        )

        self.idx_sub = self.create_subscription(
            Int32,
            "/current_path_idx",
            self.idx_callback,
            10,
        )

        self.yaw_sub = self.create_subscription(
            Bool,
            "/overtake_yaw_stable",
            self.yaw_callback,
            10,
        )

        # v2v_receiver only publishes gap/on_path while its link is fresh
        # (see v2v_receiver_node's _process()); /v2v/alive is published
        # every tick regardless and is the correct trust gate here -- same
        # pattern the follow-speed-cap fix already relies on.
        self.v2v_gap_sub = self.create_subscription(
            Float32, "/v2v/gap", self.v2v_gap_callback, 10
        )
        self.v2v_on_path_sub = self.create_subscription(
            Bool, "/v2v/on_path", self.v2v_on_path_callback, 10
        )
        self.v2v_alive_sub = self.create_subscription(
            Bool, "/v2v/alive", self.v2v_alive_callback, 10
        )

        self.watchdog_timer = self.create_timer(0.1, self.watchdog)

        self.publish_decision(
            OvertakeDecision("STARTUP_WAIT", 999.0, False)
        )

        self.get_logger().info("LiDAR-only overtake node ready")

    def idx_callback(self, msg):
        self.current_path_idx = int(msg.data)

    def yaw_callback(self, msg):
        self.yaw_stable = bool(msg.data)

    def allow_overtake_callback(self, msg):
        self.allow_overtake = bool(msg.data)

    def v2v_gap_callback(self, msg):
        self.v2v_gap = float(msg.data)

    def v2v_on_path_callback(self, msg):
        self.v2v_on_path = bool(msg.data)

    def v2v_alive_callback(self, msg):
        self.v2v_alive = bool(msg.data)

    def is_fresh(self, stamp):
        if stamp is None:
            return False

        age = (self.get_clock().now() - stamp).nanoseconds * 1e-9
        return age < self.sensor_timeout_sec

    def watchdog(self):
        if not self.safety_armed:
            self.publish_decision(
                OvertakeDecision("STARTUP_WAIT", 999.0, False)
            )
            return

        if not self.is_fresh(self.last_scan_time):
            self.publish_decision(
                OvertakeDecision("LIDAR_TIMEOUT", 999.0, False)
            )

    def publish_sound_if_needed(self, decision):
        current_state = str(decision.state)

        # Only announce when we have to fully stop and can't overtake at
        # all -- not for a normal OVERTAKE_LEFT maneuver, which is the car
        # successfully avoiding the obstacle, not stopping. Both of these
        # states make path_mpc fully stop (should_stop=True): EMERGENCY_STOP
        # is the acute close-range case, WAIT_FOR_CLEAR is everything else
        # that blocks overtaking (both sides blocked, or on a corner where
        # /allow_overtake is False due to curvature).
        obstacle_states = [
            "EMERGENCY_STOP",
            "WAIT_FOR_CLEAR",
        ]

        if (
            current_state in obstacle_states
            and self.last_drive_state != current_state
        ):
            self.sound_pub.publish(String(data="obstacle"))

        self.last_drive_state = current_state

    def limit_status_by_path_context(self, status):
        """
        Use shorter effective front/emergency distance in curves.

        self.allow_overtake comes from MPC. When it is False, the MPC thinks
        the path section is too curved for overtaking, so we reduce lookahead.
        """

        if self.allow_overtake:
            effective_front_stop_m = self.front_stop_straight_m
            effective_emergency_stop_m = self.emergency_stop_straight_m
            path_context = "STRAIGHT"
        else:
            effective_front_stop_m = self.front_stop_curve_m
            effective_emergency_stop_m = self.emergency_stop_curve_m
            path_context = "CURVE"

        front_too_far = status.front_min > effective_front_stop_m
        emergency_too_far = status.front_min > effective_emergency_stop_m

        if front_too_far or emergency_too_far:
            status = type(status)(
                obstacle_ahead=(
                    status.obstacle_ahead and not front_too_far
                ),
                emergency=(
                    status.emergency and not emergency_too_far
                ),
                left_clear=status.left_clear,
                right_clear=status.right_clear,
                front_min=status.front_min,
                left_min=status.left_min,
                right_min=status.right_min,
                front_count=status.front_count,
                left_count=status.left_count,
                right_count=status.right_count,
            )

        return status, path_context, effective_front_stop_m, effective_emergency_stop_m

    def compute_speed_cap(self, front_min):
        """Smooth speed cap from raw front distance, independent of the
        curve/straight-limited obstacle_ahead flag so it starts ramping
        down well before that binary threshold ever flips.

        front_min < 0 is the analyzer's "nothing detected" sentinel, not a
        distance -- must not be read as "very close."
        """
        if front_min < 0.0 or front_min >= self.follow_start_distance:
            return -1.0
        if front_min <= self.overtake_start_min_distance:
            return self.follow_min_speed

        span = self.follow_start_distance - self.overtake_start_min_distance
        frac = (front_min - self.overtake_start_min_distance) / span
        return self.follow_min_speed + frac * (
            self.follow_full_speed - self.follow_min_speed
        )

    def force_no_overtake_zone(self, status):
        """
        After disable_obstacle_after_idx, do not overtake.

        But do NOT disable obstacle_ahead or emergency.
        This keeps people/front safety active.
        """

        if self.current_path_idx >= self.disable_obstacle_after_idx:
            status = type(status)(
                obstacle_ahead=status.obstacle_ahead,
                emergency=status.emergency,
                left_clear=True,
                right_clear=True,
                front_min=status.front_min,
                left_min=status.left_min,
                right_min=status.right_min,
                front_count=status.front_count,
                left_count=status.left_count,
                right_count=status.right_count,
            )

        return status

    def compute_sufficient_lead_to_return(self, status):
        """True if QCar2 has enough real lead over ROSbot3 to cut back
        into its lane without entering ROSbot3's braking range.

        Switched from V2V gap to LiDAR (2026-08-28, explicit request).
        IMPORTANT LIMITATION: QCar2's LiDAR is FORWARD-facing only -- there
        is no rear sector, so this can only measure LATERAL clearance in
        the side box (side_x_max=0.70), not true longitudinal lead. It is
        a proxy: "ROSbot3 no longer registers in my side sensor" is not
        the same guarantee as "I am far enough ahead that it won't need to
        brake," but it's the best signal actually buildable with the
        sensors on this car. This is now largely the SAME check the state
        machine's own right_side_confirmed_empty already performs for the
        RETURN transition -- deliberately redundant, not a bug; keeping it
        as its own named signal for visibility/logging.
        """
        return status.right_clear and status.right_count == 0

    def apply_v2v_obstacle_detection(self, status, path_context):
        """OR a V2V-derived obstacle_ahead into status, for the curve
        case where LiDAR's own front_stop_curve_m box (0.45m) can't see
        ROSbot3 but /v2v/gap can. See __init__ for the full safety gate
        rationale -- summary: only fires within [0.85, 2.0)m and only
        when the link is alive + on_path, so it can never assert close-
        range presence and can never trigger the hard-stop/emergency path.
        emergency, left_clear, right_clear stay untouched -- LiDAR-only.

        Range is CONTEXT-AWARE (added 2026-08-28, explicit report:
        "QCar2 starts overtaking from quite back"). Confirmed live: on a
        STRAIGHT, this fired obstacle_ahead=True at v2v_gap=1.89m even
        though LiDAR's own front_stop_straight_m (1.00m) means raw LiDAR
        would never have detected anything there -- V2V's wide detection
        range, built specifically for the curve-blindness case, was ALSO
        extending detection (and therefore triggering the whole overtake
        sequence) on straights where LiDAR already sees fine on its own.
        On a straight the ceiling now matches front_stop_straight_m, so
        V2V no longer detects any farther out than LiDAR itself would;
        the wide ceiling (v2v_overtake_detect_max, ~2.0m) is reserved for
        CURVE, its original intended use case.
        """
        if status.obstacle_ahead:
            return status  # LiDAR already sees it; nothing to add.

        if not (self.v2v_alive and self.v2v_on_path):
            return status

        gap = self.v2v_gap
        if not math.isfinite(gap):
            return status

        detect_max = (
            self.v2v_overtake_detect_max
            if path_context == "CURVE"
            else self.front_stop_straight_m
        )
        if not (self.v2v_overtake_detect_min <= gap < detect_max):
            return status

        combined_front_min = (
            gap if status.front_min < 0.0 else min(status.front_min, gap)
        )

        return type(status)(
            obstacle_ahead=True,
            emergency=status.emergency,
            left_clear=status.left_clear,
            right_clear=status.right_clear,
            front_min=combined_front_min,
            left_min=status.left_min,
            right_min=status.right_min,
            front_count=status.front_count,
            left_count=status.left_count,
            right_count=status.right_count,
        )

    def scan_callback(self, msg):
        self.last_scan_time = self.get_clock().now()

        if not self.safety_armed:
            self.safety_armed = True
            self.get_logger().warn("Safety armed: LiDAR-only mode")

        raw_status = self.analyzer.analyze(msg)

        # Published every tick, before any branch/early-return below, so
        # path_mpc always has a fresh cap regardless of state.
        self.speed_cap_pub.publish(
            Float32(data=float(self.compute_speed_cap(raw_status.front_min)))
        )

        status, path_context, effective_front_stop_m, effective_emergency_stop_m = (
            self.limit_status_by_path_context(raw_status)
        )

        status = self.force_no_overtake_zone(status)
        status = self.apply_v2v_obstacle_detection(status, path_context)

        # If obstacle is too close, stop immediately.
        # Do not start an overtake when there is not enough room.
        if status.obstacle_ahead and status.front_min < self.hard_stop_front_distance:
            decision = OvertakeDecision(
                "EMERGENCY_STOP",
                999.0,
                False,
            )

            self.publish_sound_if_needed(decision)
            self.publish_decision(decision)
            self.log_debug(
                raw_status, status, decision,
                stop_reason="hard_stop_front_distance",
            )

            self.get_logger().warn(
                f"Hard stop: obstacle too close for overtake | "
                f"idx={self.current_path_idx} | "
                f"front_min={status.front_min:.2f} m | "
                f"hard_stop={self.hard_stop_front_distance:.2f} m | "
                f"context={path_context}"
            )

            return

        # Only allow overtake if obstacle is far enough, sustained --
        # see overtake_commit_margin_m/overtake_distance_confirm_required
        # in __init__ for why this is debounced with a margin rather than
        # a raw instantaneous threshold crossing.
        #
        # Prefer V2V gap over status.front_min here when trustworthy
        # (found 2026-08-28): LiDAR's front_min is distance to ROSbot3's
        # NEAREST SURFACE POINT, which is structurally always closer than
        # the V2V pose-to-pose along-path gap by roughly the vehicle's own
        # size -- confirmed live, front_min=0.77 vs v2v_gap=1.05 for the
        # SAME real-world moment. Once tight following holds QCar2 close,
        # front_min can permanently sit under 0.85/0.95 even though the
        # true gap already has real room, deadlocking the car in
        # WAIT_FOR_CLEAR forever with dist_ok_ctr stuck at 0. V2V's own
        # obstacle-detection floor (v2v_overtake_detect_min, still 0.85)
        # keeps this safe -- gap has to be a real, trustworthy, in-range
        # reading, same bar as apply_v2v_obstacle_detection above.
        if status.obstacle_ahead:
            v2v_gap_usable = (
                self.v2v_alive
                and self.v2v_on_path
                and math.isfinite(self.v2v_gap)
                and self.v2v_gap >= self.v2v_overtake_detect_min
            )
            commit_check_distance = (
                self.v2v_gap if v2v_gap_usable else status.front_min
            )
            distance_ok_now = (
                commit_check_distance
                >= self.overtake_start_min_distance + self.overtake_commit_margin_m
            )
            self.overtake_distance_ok_counter = (
                self.overtake_distance_ok_counter + 1 if distance_ok_now else 0
            )
            enough_distance_to_overtake = (
                self.overtake_distance_ok_counter
                >= self.overtake_distance_confirm_required
            )
        else:
            self.overtake_distance_ok_counter = 0
            enough_distance_to_overtake = True
            commit_check_distance = status.front_min

        # Overtake requires:
        # 1. MPC says path is straight enough
        # 2. obstacle is far enough
        # 3. still before disable index
        overtake_allowed = (
            self.allow_overtake
            and enough_distance_to_overtake
            and self.current_path_idx < self.disable_obstacle_after_idx
        )

        sufficient_lead_to_return = self.compute_sufficient_lead_to_return(status)

        decision = self.state_machine.update(
            status=status,
            overtake_allowed=overtake_allowed,
            yaw_stable=self.yaw_stable,
            sufficient_lead_to_return=sufficient_lead_to_return,
        )

        self.publish_sound_if_needed(decision)
        self.publish_decision(decision)
        self.log_debug(
            raw_status, status, decision,
            stop_reason="" if decision.motion_enabled else decision.state,
        )

        self.get_logger().info(
            f"idx={self.current_path_idx} | "
            f"context={path_context} | "
            f"state={decision.state} | "
            f"obs={status.obstacle_ahead} | "
            f"emg={status.emergency} | "
            f"L={status.left_clear} | "
            f"R={status.right_clear} | "
            f"allow_raw={self.allow_overtake} | "
            f"allow_final={overtake_allowed} | "
            f"yaw_stable={self.yaw_stable} | "
            f"enough_dist={enough_distance_to_overtake} | "
            f"commit_check_dist={commit_check_distance:.2f} | "
            f"front={status.front_min:.2f} | "
            f"left={status.left_min:.2f} | "
            f"right={status.right_min:.2f} | "
            f"front_limit={effective_front_stop_m:.2f} | "
            f"emg_limit={effective_emergency_stop_m:.2f} | "
            f"hard_stop={self.hard_stop_front_distance:.2f} | "
            f"overtake_min={self.overtake_start_min_distance:.2f} | "
            f"fc={status.front_count} | "
            f"lc={status.left_count} | "
            f"rc={status.right_count} | "
            f"offset={decision.offset:.2f} | "
            f"speed_cap={self.compute_speed_cap(raw_status.front_min):.2f} | "
            f"v2v_alive={self.v2v_alive} | "
            f"v2v_on_path={self.v2v_on_path} | "
            f"v2v_gap={self.v2v_gap:.2f} | "
            f"sufficient_lead={sufficient_lead_to_return} | "
            f"abort_return={self.state_machine.last_return_was_abort} | "
            f"dist_ok_ctr={self.overtake_distance_ok_counter} | "
            f"motion={decision.motion_enabled}",
            throttle_duration_sec=0.5,
        )

    def log_debug(self, raw_status, status, decision, stop_reason=""):
        self.debug_logger.write(
            drive_state=decision.state,
            lane_offset=decision.offset,
            safety_stop=not decision.motion_enabled,
            stop_reason=stop_reason,
            obstacle_ahead=status.obstacle_ahead,
            emergency=status.emergency,
            left_clear=status.left_clear,
            right_clear=status.right_clear,
            obstacle_ahead_raw=raw_status.obstacle_ahead,
            emergency_raw=raw_status.emergency,
            left_clear_raw=raw_status.left_clear,
            right_clear_raw=raw_status.right_clear,
            front_min=status.front_min,
            left_min=status.left_min,
            right_min=status.right_min,
            front_count=status.front_count,
            left_count=status.left_count,
            right_count=status.right_count,
        )

    def publish_decision(self, decision):
        offset_msg = Float32()
        offset_msg.data = float(decision.offset)

        motion_msg = Bool()
        motion_msg.data = bool(decision.motion_enabled)

        state_msg = String()
        state_msg.data = str(decision.state)

        self.offset_pub.publish(offset_msg)
        self.motion_pub.publish(motion_msg)
        self.state_pub.publish(state_msg)


def main():
    rclpy.init()

    node = LidarOvertakeNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.publish_decision(
        OvertakeDecision("SHUTDOWN_STOP", 999.0, False)
    )
    node.debug_logger.close()

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()