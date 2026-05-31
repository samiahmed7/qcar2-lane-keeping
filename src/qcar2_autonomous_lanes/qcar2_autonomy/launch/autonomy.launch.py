#!/usr/bin/env python3
"""Launch the QCar2 autonomy decision stack."""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory('qcar2_autonomy')
    config_dir = os.path.join(pkg_share, 'config')

    return LaunchDescription([
        Node(
            package='qcar2_autonomy',
            executable='bev_lane_detector',
            name='bev_lane_detector_node',
            parameters=[os.path.join(config_dir, 'lane_detection.yaml')],
            output='screen',
        ),
        Node(
            package='qcar2_autonomy',
            executable='lane_controller',
            name='lane_controller_node',
            parameters=[os.path.join(config_dir, 'controller.yaml')],
            output='screen',
        ),
        Node(
            package='qcar2_autonomy',
            executable='safety_supervisor',
            name='safety_supervisor_node',
            parameters=[os.path.join(config_dir, 'safety.yaml')],
            output='screen',
        ),
        Node(
            package='qcar2_autonomy',
            executable='lidar_obstacle',
            name='lidar_obstacle_node',
            parameters=[os.path.join(config_dir, 'obstacle.yaml')],
            output='screen',
        ),
        Node(
            package='qcar2_autonomy',
            executable='behavior_fsm',
            name='behavior_fsm_node',
            parameters=[os.path.join(config_dir, 'behavior.yaml')],
            output='screen',
        ),
    ])
