#!/usr/bin/env python3
"""Launch the metrics-recording placeholder."""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory('qcar2_autonomy')
    config = os.path.join(pkg_share, 'config', 'metrics.yaml')

    return LaunchDescription([
        Node(
            package='qcar2_autonomy',
            executable='placeholder_node',
            name='metrics_placeholder',
            parameters=[config, {'placeholder_role': 'metrics'}],
            output='screen',
        ),
    ])
