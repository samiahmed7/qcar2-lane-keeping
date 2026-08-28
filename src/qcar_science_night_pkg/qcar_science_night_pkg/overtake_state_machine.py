from qcar_science_night_pkg.overtake_types import OvertakeDecision


class OvertakeStateMachine:
    DRIVE = "DRIVE"
    WAIT = "WAIT_FOR_CLEAR"
    OVERTAKE = "OVERTAKE_LEFT"
    RETURN = "RETURN_RIGHT"
    ESTOP = "EMERGENCY_STOP"

    def __init__(
        self,
        overtake_offset=0.65,
        obstacle_confirm_required=3,
        left_clear_confirm_required=2,
        no_obstacle_confirm_required=15,
        right_clear_confirm_required=10,
        return_confirm_required=30,
        min_overtake_steps=30,
    ):
        self.state = self.DRIVE
        self.overtake_offset = float(overtake_offset)

        self.obstacle_confirm_required = obstacle_confirm_required
        self.left_clear_confirm_required = left_clear_confirm_required
        self.no_obstacle_confirm_required = no_obstacle_confirm_required
        self.right_clear_confirm_required = right_clear_confirm_required
        self.return_confirm_required = return_confirm_required
        self.min_overtake_steps = min_overtake_steps

        self.obstacle_counter = 0
        self.left_clear_counter = 0
        self.right_clear_counter = 0
        self.no_obstacle_counter = 0
        self.return_counter = 0
        self.overtake_counter = 0

        # True when RETURN was entered by aborting a blocked overtake rather
        # than by completing one. Purely for observability -- a normal
        # completion and an abort otherwise look identical in the logs, and
        # telling them apart matters when diagnosing overtake behavior.
        self.last_return_was_abort = False

    def reset_counters(self):
        self.obstacle_counter = 0
        self.left_clear_counter = 0
        self.right_clear_counter = 0
        self.no_obstacle_counter = 0
        self.return_counter = 0
        self.overtake_counter = 0

    def update_counters(self, status):
        if status.obstacle_ahead:
            self.obstacle_counter += 1
            self.no_obstacle_counter = 0
        else:
            self.no_obstacle_counter += 1
            self.obstacle_counter = 0

        if status.left_clear:
            self.left_clear_counter += 1
        else:
            self.left_clear_counter = 0

        if status.right_clear:
            self.right_clear_counter += 1
        else:
            self.right_clear_counter = 0

    def start_overtake(self):
        self.state = self.OVERTAKE
        self.overtake_counter = 0
        self.return_counter = 0
        return OvertakeDecision(self.state, self.overtake_offset, True)

    def update(self, status, overtake_allowed, yaw_stable=True, sufficient_lead_to_return=True):
        self.update_counters(status)

        confirmed_obstacle = (
            self.obstacle_counter >= self.obstacle_confirm_required
        )
        confirmed_left_clear = (
            self.left_clear_counter >= self.left_clear_confirm_required
        )
        confirmed_no_obstacle = (
            self.no_obstacle_counter >= self.no_obstacle_confirm_required
        )
        confirmed_right_clear = (
            self.right_clear_counter >= self.right_clear_confirm_required
        )

        if status.emergency:
            self.state = self.ESTOP
            self.return_counter = 0
            self.overtake_counter = 0
            return OvertakeDecision(self.state, 999.0, False)

        if self.state == self.DRIVE:
            if not confirmed_obstacle:
                return OvertakeDecision(self.state, 0.0, True)

            both_blocked = (
                not status.left_clear
                and not status.right_clear
            )

            if both_blocked:
                self.state = self.WAIT
                return OvertakeDecision(self.state, 999.0, False)

            can_avoid = (
                confirmed_obstacle
                and overtake_allowed
                and status.left_clear
                and confirmed_left_clear
            )

            if can_avoid:
                return self.start_overtake()

            self.state = self.WAIT
            return OvertakeDecision(self.state, 999.0, False)

        if self.state == self.WAIT:
            if not confirmed_obstacle:
                self.state = self.DRIVE
                self.reset_counters()
                return OvertakeDecision(self.state, 0.0, True)

            both_blocked = (
                not status.left_clear
                and not status.right_clear
            )

            if both_blocked:
                return OvertakeDecision(self.state, 999.0, False)

            can_avoid = (
                confirmed_obstacle
                and overtake_allowed
                and status.left_clear
                and confirmed_left_clear
            )

            if can_avoid:
                return self.start_overtake()

            return OvertakeDecision(self.state, 999.0, False)

        if self.state == self.OVERTAKE:
            self.overtake_counter += 1

            # Overtake lane blocked mid-maneuver. This used to go straight to
            # WAIT (a full stop) with no regard for the lane we came from --
            # so a blocked overtake lane stopped the car dead even when the
            # original lane was completely clear. Demonstrated on hardware by
            # the supervisor 2026-08-28: he stood in the overtake lane during
            # a pass and the car stopped instead of pulling back in. The
            # paper this work reconstructs argues against exactly this: the
            # ego should keep exploring options rather than "merely adopting
            # a passive safety mode in response to the environment"
            # (IDEAM Sec. V-A-2). Now we only stop when there is genuinely
            # nowhere to go.
            if not status.left_clear and status.left_count > 0:
                original_lane_empty = (
                    status.right_clear
                    and status.right_count == 0
                )

                if original_lane_empty:
                    # Abort the pass and pull back into the lane we came from.
                    #
                    # Deliberately NOT gated on min_overtake_steps or
                    # sufficient_lead_to_return, unlike the normal completion
                    # below. This is an escape from a blocked lane, not a
                    # finished overtake -- requiring "am I far enough ahead of
                    # ROSbot3" would be backwards, since we are aborting
                    # precisely because we are NOT completing the pass.
                    #
                    # Also deliberately NOT gated on confirmed_right_clear
                    # (the 3-tick counter used by the normal return): if this
                    # check fails we fall through to WAIT, and WAIT cannot
                    # re-enter OVERTAKE while the left lane is still blocked
                    # (`can_avoid` requires left_clear), so a counter that
                    # needs several ticks to build would trap the car stopped
                    # -- the exact failure being fixed. right_count == 0 is
                    # already strict on its own: literally zero LiDAR returns
                    # in the side box, not a marginal distance reading.
                    self.state = self.RETURN
                    self.return_counter = 0
                    self.last_return_was_abort = True
                    return OvertakeDecision(self.state, 0.0, True)

                # Both lanes blocked -- nowhere to go, stopping is correct.
                self.state = self.WAIT
                return OvertakeDecision(self.state, 999.0, False)

            right_side_confirmed_empty = (
                confirmed_right_clear
                and status.right_clear
                and status.right_count == 0
            )

            # confirmed_no_obstacle only means ROSbot3 fell out of the FRONT
            # sensor -- true the instant it's behind QCar2, regardless of
            # how much actual lead QCar2 has. Without sufficient_lead_to_
            # return, that let the car snap straight to offset=0.0 (full
            # cut-in) the moment it lost sight of ROSbot3, sometimes
            # merging back inside ROSbot3's own braking range. Found
            # 2026-08-28, fixed using V2V's real longitudinal gap (computed
            # by the caller -- this class has no V2V knowledge of its own).
            if (
                confirmed_no_obstacle
                and right_side_confirmed_empty
                and yaw_stable
                and sufficient_lead_to_return
                and self.overtake_counter >= self.min_overtake_steps
            ):
                self.state = self.RETURN
                self.return_counter = 0
                self.last_return_was_abort = False
                return OvertakeDecision(self.state, 0.0, True)

            return OvertakeDecision(self.state, self.overtake_offset, True)

        if self.state == self.RETURN:
            # Do not jump back left from one noisy right-side point.
            # Only abort return if right lane is clearly blocked.
            if not status.right_clear and status.right_count >= 2:
                
                return OvertakeDecision(
                    self.state,
                    0.0,
                    True,
                )

            self.return_counter += 1

            if self.return_counter >= self.return_confirm_required:
                self.state = self.DRIVE
                self.return_counter = 0
                self.overtake_counter = 0
                self.reset_counters()

            return OvertakeDecision(self.state, 0.0, True)

        if self.state == self.ESTOP:
            if status.emergency:
                return OvertakeDecision(self.state, 999.0, False)

            self.state = self.WAIT
            self.reset_counters()
            return OvertakeDecision(self.state, 999.0, False)

        self.state = self.DRIVE
        self.reset_counters()
        return OvertakeDecision(self.state, 0.0, True)