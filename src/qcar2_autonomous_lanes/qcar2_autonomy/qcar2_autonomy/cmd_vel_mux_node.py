#!/usr/bin/env python3
"""State-aware two-input Twist mux for the QCar 2 lane-keeping pipeline.

Architecture:

    lane-keep author        -> /qcar2/control/raw_cmd_vel       (e.g. PID,
                                                                  lane_controller,
                                                                  rfdetr->PID)
    maneuver author         -> /qcar2/control/maneuver_cmd_vel  (e.g. quintic
                                                                  lane-change
                                                                  planner)
    state_machine_node      -> /system/current_state            (String)

    THIS NODE:
        sub  /qcar2/control/raw_cmd_vel
        sub  /qcar2/control/maneuver_cmd_vel
        sub  /system/current_state
        pub  /model/qcar2/cmd_vel

Rule:
    LANE_KEEP                -> forward lane-keep input
    LANE_CHANGE / LANE_RETURN -> forward maneuver input
    unknown / startup        -> default to LANE_KEEP

Why a separate node and not a topic remap: two separate authors keep publishing
on their own topics even when they're "off-shift" (PID still emits zero Twists
during a swerve, planner still settles after maneuver completion). Routing both
into one output topic from outside would race them. The mux drops the off-shift
author entirely so the simulator only ever sees one author per moment.
"""
import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from std_msgs.msg import Bool, String


LANE_KEEP = 'LANE_KEEP'
LANE_CHANGE = 'LANE_CHANGE'
LANE_RETURN = 'LANE_RETURN'
MANEUVER_STATES = (LANE_CHANGE, LANE_RETURN)


class CmdVelMuxNode(Node):
    """Route lane-keep or maneuver Twist using planner-active + FSM-state.

    Earlier the mux switched on FSM state alone. Problem: LANE_CHANGE includes
    both the active quintic swerve and a multi-second "hold past the obstacle"
    phase. Routing the planner during the hold pins the car at a fixed
    path-frame offset instead of letting the lane-keeper track the centre of
    whichever lane the car is actually in.

    Decision rule (in priority order):
        1. maneuver_active == True  -> forward maneuver_cmd (quintic owns)
        2. FSM in (LANE_CHANGE, LANE_RETURN) and no planner active
              -> drop lane_keep (legacy DWA case publishes direct to output)
        3. otherwise -> forward lane_keep_cmd

    This keeps the rfdetr branch (planner present, FSM-state-only legacy
    avoidance absent) and the bev/DWA branch (planner absent) both correct
    from one mux implementation.
    """

    def __init__(self):
        super().__init__('cmd_vel_mux_node')

        self.declare_parameter('lane_keep_topic', '/qcar2/control/raw_cmd_vel')
        self.declare_parameter('maneuver_topic', '/qcar2/control/maneuver_cmd_vel')
        self.declare_parameter('output_cmd_topic', '/model/qcar2/cmd_vel')
        self.declare_parameter('maneuver_active_topic', '/qcar2/control/maneuver_active')
        self.declare_parameter('state_topic', '/system/current_state')
        # route_by_state: forward the maneuver author for the WHOLE
        # LANE_CHANGE/LANE_RETURN window (planner owns the open-loop S-curve and
        # the straight "pass" creep), and the lane-keep author only in LANE_KEEP.
        # This is the robust mode for vision detectors that lose the lane while
        # the car is yawed. Default False keeps the legacy maneuver_active rule.
        self.declare_parameter('route_by_state', False)
        self.route_by_state = bool(self.get_parameter('route_by_state').value)

        self.maneuver_active = False
        self.current_state = LANE_KEEP
        # Once the planner has ever spoken to us, we trust its active flag and
        # ignore the FSM-state fallback. The fallback exists only for legacy
        # branches (bev/DWA) where no planner publishes maneuver_active.
        self.planner_seen = False

        self.pub = self.create_publisher(
            Twist,
            self.get_parameter('output_cmd_topic').value,
            10,
        )
        self.create_subscription(
            Twist,
            self.get_parameter('lane_keep_topic').value,
            self._on_lane_keep_cmd,
            10,
        )
        self.create_subscription(
            Twist,
            self.get_parameter('maneuver_topic').value,
            self._on_maneuver_cmd,
            10,
        )
        self.create_subscription(
            Bool,
            self.get_parameter('maneuver_active_topic').value,
            self._on_active,
            10,
        )
        self.create_subscription(
            String,
            self.get_parameter('state_topic').value,
            self._on_state,
            10,
        )

        self.get_logger().info(
            f'cmd_vel mux ready: '
            f'lane_keep={self.get_parameter("lane_keep_topic").value}, '
            f'maneuver={self.get_parameter("maneuver_topic").value} '
            f'-> {self.get_parameter("output_cmd_topic").value}'
        )

    def _on_active(self, msg: Bool):
        self.maneuver_active = bool(msg.data)
        self.planner_seen = True

    def _on_state(self, msg: String):
        self.current_state = msg.data.strip().upper() or LANE_KEEP

    def _on_lane_keep_cmd(self, msg: Twist):
        if self.route_by_state:
            # Planner owns CHANGE/RETURN entirely; lane-keep drives only KEEP.
            if self.current_state == LANE_KEEP:
                self.pub.publish(msg)
            return
        if self.maneuver_active:
            return
        if not self.planner_seen and self.current_state in MANEUVER_STATES:
            # Legacy DWA branch only: a separate planner is publishing directly
            # to the output topic, so drop lane_keep to avoid two authors.
            return
        self.pub.publish(msg)

    def _on_maneuver_cmd(self, msg: Twist):
        if self.route_by_state:
            if self.current_state in MANEUVER_STATES:
                self.pub.publish(msg)
            return
        if self.maneuver_active:
            self.pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = CmdVelMuxNode()
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
