#!/usr/bin/env python3
"""Launch Phase 1 lane perception."""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory('qcar2_autonomy')
    config = os.path.join(pkg_share, 'config', 'lane_detection.yaml')

    return LaunchDescription([
        Node(
            package='qcar2_autonomy',
            executable='bev_lane_detector',
            name='bev_lane_detector_node',
            parameters=[config],
            output='screen',
        ),
    ])
