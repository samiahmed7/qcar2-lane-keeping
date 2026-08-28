import math
import numpy as np

from qcar_science_night_pkg.overtake_types import ObstacleStatus


class LidarSectorAnalyzer:
    def __init__(
        self,
        front_offset_deg=180.0,
        max_range=1.2,
        min_range=0.03,
        lane_width=0.43,
        front_x_min=0.03,
        front_x_max=0.75,
        side_x_min=0.05,
        side_x_max=0.70,
        emergency_x_min=0.05,
        emergency_x_max=0.40,
        overtake_start_distance=0.75,
        emergency_distance=0.40,
        lane_clear_distance=0.75,
        min_front_points=1,
        min_side_points=1,
        min_emergency_points=2,
    ):
        self.front_offset = math.radians(front_offset_deg)
        self.max_range = float(max_range)
        self.min_range = float(min_range)

        self.lane_width = float(lane_width)
        self.half_lane_width = self.lane_width / 2.0

        self.front_x_min = float(front_x_min)
        self.front_x_max = float(front_x_max)

        self.side_x_min = float(side_x_min)
        self.side_x_max = float(side_x_max)

        self.emergency_x_min = float(emergency_x_min)
        self.emergency_x_max = float(emergency_x_max)

        self.overtake_start_distance = float(overtake_start_distance)
        self.emergency_distance = float(emergency_distance)
        self.lane_clear_distance = float(lane_clear_distance)

        self.min_front_points = int(min_front_points)
        self.min_side_points = int(min_side_points)
        self.min_emergency_points = int(min_emergency_points)

    def scan_to_xy(self, scan_msg):
        points = []

        for i, r in enumerate(scan_msg.ranges):
            if np.isnan(r) or np.isinf(r):
                continue

            if not (self.min_range < r < self.max_range):
                continue

            angle = scan_msg.angle_min + i * scan_msg.angle_increment

            rel = math.atan2(
                math.sin(angle - self.front_offset),
                math.cos(angle - self.front_offset),
            )

            x = float(r * math.cos(rel))  # forward
            y = float(r * math.sin(rel))  # left positive

            if x <= 0.0:
                continue

            points.append((x, y, float(r)))

        return points

    def box_points(self, points, x_min, x_max, y_min, y_max):
        selected = []

        for x, y, r in points:
            if x_min <= x <= x_max and y_min <= y <= y_max:
                selected.append(r)

        return np.array(selected, dtype=float)

    @staticmethod
    def min_distance(points):
        if len(points) == 0:
            return -1.0
        return float(np.min(points))

    @staticmethod
    def count_points(points):
        return int(len(points))

    def analyze(self, scan_msg):
        points = self.scan_to_xy(scan_msg)

        half = self.half_lane_width
        lane = self.lane_width

        # Ego/front lane: y = -half to +half
        # front_x_min is reduced to 0.05 so low/close objects are not missed.
        front_points = self.box_points(
            points,
            self.front_x_min,
            self.front_x_max,
            -half,
            half,
        )

        # Left lane: y = +half to +(half + lane)
        left_points = self.box_points(
            points,
            self.side_x_min,
            self.side_x_max,
            half,
            half + lane,
        )

        # Right lane: y = -(half + lane) to -half
        right_points = self.box_points(
            points,
            self.side_x_min,
            self.side_x_max,
            -(half + lane),
            -half,
        )

        # Sudden close obstacle in ego lane
        emergency_half_width = 0.05  # 8 cm each side
        emergency_center_y = 0.02
        emergency_points = self.box_points(
            points,
            self.emergency_x_min,
            self.emergency_x_max,
            emergency_center_y - emergency_half_width,
            emergency_center_y + emergency_half_width,
        )

        front_min = self.min_distance(front_points)
        left_min = self.min_distance(left_points)
        right_min = self.min_distance(right_points)
        emergency_min = self.min_distance(emergency_points)

        front_count = self.count_points(front_points)
        left_count = self.count_points(left_points)
        right_count = self.count_points(right_points)
        emergency_count = self.count_points(emergency_points)

        # No wall filtering here.
        # Waypoint/curve logic in lidar_overtake_node should decide where to ignore walls.
        obstacle_ahead = (
            front_count >= self.min_front_points
            and front_min > 0.0
            and front_min <= self.overtake_start_distance
        )

        emergency = (
            emergency_count >= self.min_emergency_points
            and emergency_min > 0.0
            and emergency_min <= self.emergency_distance
        )

        left_clear = (
            left_count < self.min_side_points
            or (left_min > self.lane_clear_distance and left_min > 0)
        )
        right_clear = (
            right_count < self.min_side_points
            or (right_min > self.lane_clear_distance and right_min > 0)
        )

        return ObstacleStatus(
            obstacle_ahead=obstacle_ahead,
            emergency=emergency,
            left_clear=left_clear,
            right_clear=right_clear,
            front_min=front_min,
            left_min=left_min,
            right_min=right_min,
            front_count=front_count,
            left_count=left_count,
            right_count=right_count,
        )