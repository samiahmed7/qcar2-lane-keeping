#!/usr/bin/env python3
"""
Safety filter — final veto layer between any cmd_vel source and the drive.

Subscribes:
  /qcar2/cmd_vel_desired    geometry_msgs/Twist
    The intended drive command from the lane follower (or any other source,
    e.g. a future V2V coordinator).
  /qcar2/lidar/scan         sensor_msgs/LaserScan
    Front-facing 360° lidar; we only look at a configurable front arc.

Publishes:
  /model/qcar2/cmd_vel      geometry_msgs/Twist
    The actual drive command sent to Gazebo. Passes the desired command
    through unchanged unless an obstacle is inside the front arc closer
    than `stop_distance`, in which case linear.x is forced to 0 and a
    warning is logged.

The point of this node is that **safety is independent of intent**. The lane
follower can request any cmd_vel it wants; if a real obstacle is in front,
the car stops regardless. When V2V comes online and says "change lane",
the safety filter still vetoes a move into something solid.
"""
import math
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist


class SafetyFilter(Node):
    def __init__(self):
        super().__init__('safety_filter')

        self.declare_parameter('stop_distance', 0.40)        # m
        self.declare_parameter('slow_distance', 0.80)        # m — start braking here
        self.declare_parameter('front_arc_deg', 30.0)        # half-angle, total arc = 2x
        self.declare_parameter('min_valid_range', 0.05)      # m — filter near-noise
        self.declare_parameter('cmd_vel_timeout', 0.5)       # s — stop if desired stale

        self.stop_distance = self.get_parameter('stop_distance').value
        self.slow_distance = self.get_parameter('slow_distance').value
        self.front_arc_rad = math.radians(self.get_parameter('front_arc_deg').value)
        self.min_valid_range = self.get_parameter('min_valid_range').value
        self.cmd_vel_timeout = self.get_parameter('cmd_vel_timeout').value

        self.last_desired = Twist()
        self.last_desired_stamp = None
        self.last_min_range = float('inf')
        self.blocked = False

        self.create_subscription(
            Twist, '/qcar2/cmd_vel_desired',
            self.on_desired_cmd, 10,
        )
        self.create_subscription(
            LaserScan, '/qcar2/lidar/scan',
            self.on_scan, 10,
        )
        self.cmd_pub = self.create_publisher(
            Twist, '/model/qcar2/cmd_vel', 10
        )

        # 20 Hz heartbeat — publish current safe command even if nothing
        # changes upstream. This guarantees the actuator gets fresh data.
        self.create_timer(0.05, self.tick)

        self.get_logger().info(
            f'Safety filter up. stop_dist={self.stop_distance}m '
            f'slow_dist={self.slow_distance}m arc=±{math.degrees(self.front_arc_rad):.0f}°'
        )

    def on_desired_cmd(self, msg: Twist):
        self.last_desired = msg
        self.last_desired_stamp = self.get_clock().now()

    def on_scan(self, msg: LaserScan):
        """Find the minimum range in the front arc."""
        # Index range covering ±front_arc_rad around angle 0 (forward).
        # LiDAR is mounted with the heading along +x; in our URDF the lidar
        # frame has 0 at -π and increases to +π, so angle 0 is straight ahead.
        n = len(msg.ranges)
        angles = [msg.angle_min + i * msg.angle_increment for i in range(n)]
        front_ranges = []
        for r, a in zip(msg.ranges, angles):
            if not math.isfinite(r):
                continue
            if r < self.min_valid_range:
                continue
            if abs(a) <= self.front_arc_rad:
                front_ranges.append(r)
        self.last_min_range = min(front_ranges) if front_ranges else float('inf')

    def tick(self):
        """Compute and publish the safe cmd_vel at fixed rate."""
        twist = Twist()

        # Stale-desired guard: if upstream went quiet, stop.
        now = self.get_clock().now()
        if (self.last_desired_stamp is None
                or (now - self.last_desired_stamp).nanoseconds * 1e-9
                    > self.cmd_vel_timeout):
            self.cmd_pub.publish(twist)   # zeros
            return

        # Copy desired
        twist.linear.x = self.last_desired.linear.x
        twist.angular.z = self.last_desired.angular.z

        # Apply safety
        r = self.last_min_range
        if r < self.stop_distance:
            # Hard stop
            if not self.blocked:
                self.get_logger().warn(
                    f'OBSTACLE at {r:.2f}m — EMERGENCY STOP'
                )
                self.blocked = True
            twist.linear.x = 0.0
            twist.angular.z = 0.0
        elif r < self.slow_distance:
            # Linear brake between slow and stop distance
            scale = (r - self.stop_distance) / (self.slow_distance - self.stop_distance)
            scale = max(0.0, min(1.0, scale))
            twist.linear.x *= scale
            if self.blocked:
                self.get_logger().info(f'Obstacle in slow zone at {r:.2f}m')
        else:
            if self.blocked:
                self.get_logger().info(f'Path clear ({r:.2f}m) — resuming')
                self.blocked = False

        self.cmd_pub.publish(twist)


def main(args=None):
    rclpy.init(args=args)
    node = SafetyFilter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
