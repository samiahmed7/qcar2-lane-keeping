#!/usr/bin/env python3
"""Launch lane perception + lane-keeping controller."""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory('qcar2_autonomy')
    lane_config = os.path.join(pkg_share, 'config', 'lane_detection.yaml')
    controller_config = os.path.join(pkg_share, 'config', 'controller.yaml')
    safety_config = os.path.join(pkg_share, 'config', 'safety.yaml')

    return LaunchDescription([
        Node(
            package='qcar2_autonomy',
            executable='bev_lane_detector',
            name='bev_lane_detector_node',
            parameters=[lane_config],
            output='screen',
        ),
        Node(
            package='qcar2_autonomy',
            executable='lane_controller',
            name='lane_controller_node',
            parameters=[controller_config],
            output='screen',
        ),
        Node(
            package='qcar2_autonomy',
            executable='safety_supervisor',
            name='safety_supervisor_node',
            parameters=[safety_config],
            output='screen',
        ),
    ])
