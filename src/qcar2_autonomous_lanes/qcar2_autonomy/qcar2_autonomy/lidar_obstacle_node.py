#!/usr/bin/env python3
"""LiDAR curved-corridor obstacle detector for QCar2."""
import math

import rclpy
from qcar2_msgs.msg import BehaviorState, LaneModel, ObstacleState
from rclpy.node import Node
from sensor_msgs.msg import LaserScan


LEFT = 'LEFT'
RIGHT = 'RIGHT'
KEEP_RIGHT = 'KEEP_RIGHT'
PREPARE_LEFT_CHANGE = 'PREPARE_LEFT_CHANGE'
CHANGE_LEFT = 'CHANGE_LEFT'
LEFT_LANE_KEEP = 'LEFT_LANE_KEEP'
RETURN_RIGHT = 'RETURN_RIGHT'
EMERGENCY_STOP = 'EMERGENCY_STOP'


class LidarObstacleNode(Node):
    def __init__(self):
        super().__init__('lidar_obstacle_node')

        self.declare_parameter('scan_topic', '/qcar2/lidar/scan')
        self.declare_parameter('lane_model_topic', '/qcar2/lane/model')
        self.declare_parameter('behavior_state_topic', '/qcar2/behavior/state')
        self.declare_parameter('obstacle_state_topic', '/qcar2/obstacle/state')
        self.declare_parameter('front_angle_deg', 25.0)
        self.declare_parameter('side_angle_min_deg', 35.0)
        self.declare_parameter('side_angle_max_deg', 110.0)
        self.declare_parameter('front_stop_distance', 2.0)
        self.declare_parameter('side_clear_distance', 0.60)
        self.declare_parameter('publish_rate', 10.0)
        self.declare_parameter('lane_width_m', 0.75)
        self.declare_parameter('corridor_width_m', 0.75)
        self.declare_parameter('corridor_start_m', 0.10)
        self.declare_parameter('corridor_length_m', 3.2)
        self.declare_parameter('corridor_samples', 18)
        self.declare_parameter('lane_clear_distance', 3.0)
        self.declare_parameter('min_points_to_block', 2)
        self.declare_parameter('scan_timeout_sec', 0.8)
        self.declare_parameter('max_abs_curvature', 1.2)

        self.front_angle = math.radians(float(self.get_parameter('front_angle_deg').value))
        self.side_angle_min = math.radians(
            float(self.get_parameter('side_angle_min_deg').value)
        )
        self.side_angle_max = math.radians(
            float(self.get_parameter('side_angle_max_deg').value)
        )
        self.front_stop_distance = max(
            0.1,
            float(self.get_parameter('front_stop_distance').value),
        )
        self.side_clear_distance = max(
            0.1,
            float(self.get_parameter('side_clear_distance').value),
        )
        self.lane_width_m = max(0.1, float(self.get_parameter('lane_width_m').value))
        self.corridor_width_m = max(
            0.2,
            float(self.get_parameter('corridor_width_m').value),
        )
        self.corridor_start_m = max(
            0.0,
            float(self.get_parameter('corridor_start_m').value),
        )
        self.corridor_length_m = max(
            self.corridor_start_m + 0.5,
            float(self.get_parameter('corridor_length_m').value),
        )
        self.corridor_samples = max(4, int(self.get_parameter('corridor_samples').value))
        self.lane_clear_distance = max(
            self.side_clear_distance,
            float(self.get_parameter('lane_clear_distance').value),
        )
        self.min_points_to_block = max(
            1,
            int(self.get_parameter('min_points_to_block').value),
        )
        self.scan_timeout_sec = max(
            0.1,
            float(self.get_parameter('scan_timeout_sec').value),
        )
        self.max_abs_curvature = max(
            0.0,
            float(self.get_parameter('max_abs_curvature').value),
        )

        self.last_scan = None
        self.last_scan_time = None
        self.last_lane = None
        self.target_lane = RIGHT
        self.behavior_state = KEEP_RIGHT

        self.pub = self.create_publisher(
            ObstacleState,
            self.get_parameter('obstacle_state_topic').value,
            10,
        )
        self.create_subscription(
            LaserScan,
            self.get_parameter('scan_topic').value,
            self._on_scan,
            10,
        )
        self.create_subscription(
            LaneModel,
            self.get_parameter('lane_model_topic').value,
            self._on_lane,
            10,
        )
        self.create_subscription(
            BehaviorState,
            self.get_parameter('behavior_state_topic').value,
            self._on_behavior,
            10,
        )

        rate = max(float(self.get_parameter('publish_rate').value), 1.0)
        self.create_timer(1.0 / rate, self._tick)

        self.get_logger().info(
            'Curved-corridor LiDAR obstacle detector ready: '
            f'lane_width={self.lane_width_m}, corridor_width={self.corridor_width_m}'
        )

    def _on_scan(self, msg: LaserScan):
        self.last_scan = msg
        self.last_scan_time = self.get_clock().now()

    def _on_lane(self, msg: LaneModel):
        self.last_lane = msg

    def _on_behavior(self, msg: BehaviorState):
        requested = msg.target_lane.strip().upper()
        self.target_lane = LEFT if requested == LEFT else RIGHT
        self.behavior_state = msg.state.strip().upper() or KEEP_RIGHT

    def _tick(self):
        if self.last_scan is None or self._scan_age_sec() > self.scan_timeout_sec:
            self.pub.publish(self._unknown_state())
            return

        self.pub.publish(self._state_from_scan(self.last_scan))

    def _scan_age_sec(self):
        if self.last_scan_time is None:
            return math.inf
        return max(0.0, (self.get_clock().now() - self.last_scan_time).nanoseconds * 1e-9)

    def _state_from_scan(self, scan: LaserScan):
        points = self._scan_points(scan)
        curvature = self._path_curvature()
        left_offset, right_offset, active_offset = self._corridor_offsets()

        right = self._corridor_stats(
            points,
            lane_offset_y=right_offset,
            curvature=curvature,
        )
        left = self._corridor_stats(
            points,
            lane_offset_y=left_offset,
            curvature=curvature,
        )
        active = self._corridor_stats(
            points,
            lane_offset_y=active_offset,
            curvature=curvature,
        )

        msg = ObstacleState()
        msg.header = scan.header
        msg.front_min_distance = float(active['min_distance'])
        msg.left_min_distance = float(left['min_distance'])
        msg.right_min_distance = float(right['min_distance'])
        msg.front_blocked = self._blocked(active, self.front_stop_distance)
        msg.left_clear = self._clear(left, self.lane_clear_distance)
        msg.right_clear = self._clear(right, self.lane_clear_distance)
        return msg

    def _corridor_offsets(self):
        """Return left-adjacent, right-adjacent, and active path offsets.

        Offsets are in the vehicle/LiDAR frame: +y is to the car's left. When
        the car has completed the left lane change, the right lane is no longer
        straight ahead; it is one lane width to the car's right. The FSM uses
        right_clear to decide when to return, so this frame switch matters.
        """
        if self.behavior_state == LEFT_LANE_KEEP:
            return self.lane_width_m, -self.lane_width_m, 0.0

        if self.behavior_state == RETURN_RIGHT:
            return self.lane_width_m, -self.lane_width_m, -self.lane_width_m

        if self.behavior_state == CHANGE_LEFT:
            return self.lane_width_m, 0.0, self.lane_width_m

        # KEEP_RIGHT, PREPARE_LEFT_CHANGE, startup, and emergency recovery.
        active = self.lane_width_m if self.target_lane == LEFT else 0.0
        return self.lane_width_m, 0.0, active

    def _path_curvature(self):
        if self.last_lane is None or not math.isfinite(self.last_lane.curvature):
            return 0.0
        return max(
            -self.max_abs_curvature,
            min(self.max_abs_curvature, float(self.last_lane.curvature)),
        )

    def _scan_points(self, scan: LaserScan):
        points = []
        angle = scan.angle_min
        for distance in scan.ranges:
            if self._valid_range(scan, distance):
                x = float(distance) * math.cos(angle)
                y = float(distance) * math.sin(angle)
                if self.corridor_start_m <= x <= self.corridor_length_m:
                    points.append((x, y, float(distance)))
            angle += scan.angle_increment
        return points

    def _corridor_stats(self, points, lane_offset_y, curvature):
        polygon = self._corridor_polygon(lane_offset_y, curvature)
        min_distance = math.inf
        min_x = math.inf
        hit_count = 0
        for x, y, distance in points:
            if self._point_in_polygon(x, y, polygon):
                hit_count += 1
                if distance < min_distance:
                    min_distance = distance
                    min_x = x
        return {
            'min_distance': min_distance,
            'min_x': min_x,
            'hit_count': hit_count,
        }

    def _corridor_polygon(self, lane_offset_y, curvature):
        half_width = 0.5 * self.corridor_width_m
        xs = [
            self.corridor_start_m
            + i * (self.corridor_length_m - self.corridor_start_m)
            / float(self.corridor_samples - 1)
            for i in range(self.corridor_samples)
        ]
        left_edge = []
        right_edge = []
        for x in xs:
            center_y = lane_offset_y + 0.5 * curvature * x * x
            left_edge.append((x, center_y + half_width))
            right_edge.append((x, center_y - half_width))
        return left_edge + list(reversed(right_edge))

    @staticmethod
    def _point_in_polygon(x, y, polygon):
        inside = False
        j = len(polygon) - 1
        for i, (xi, yi) in enumerate(polygon):
            xj, yj = polygon[j]
            crosses_y = (yi > y) != (yj > y)
            if crosses_y:
                denom = yj - yi
                if abs(denom) < 1e-9:
                    j = i
                    continue
                x_at_y = (xj - xi) * (y - yi) / denom + xi
                if x < x_at_y:
                    inside = not inside
            j = i
        return inside

    def _blocked(self, stats, distance_limit):
        return (
            stats['hit_count'] >= self.min_points_to_block
            and stats['min_distance'] < distance_limit
        )

    def _clear(self, stats, clear_distance):
        if stats['hit_count'] < self.min_points_to_block:
            return True
        return stats['min_distance'] > clear_distance

    @staticmethod
    def _valid_range(scan: LaserScan, distance):
        return (
            math.isfinite(distance)
            and scan.range_min <= distance <= scan.range_max
        )

    def _unknown_state(self):
        msg = ObstacleState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.front_blocked = True
        msg.left_clear = False
        msg.right_clear = False
        msg.front_min_distance = math.inf
        msg.left_min_distance = math.inf
        msg.right_min_distance = math.inf
        return msg


def main(args=None):
    rclpy.init(args=args)
    node = LidarObstacleNode()
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
