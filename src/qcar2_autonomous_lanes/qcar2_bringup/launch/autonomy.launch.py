#!/usr/bin/env python3
"""Launch simulation bringup plus the autonomous lane stack."""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    pkg_bringup = get_package_share_directory('qcar2_bringup')
    pkg_autonomy = get_package_share_directory('qcar2_autonomy')

    return LaunchDescription([
        DeclareLaunchArgument('world', default_value='lab_track'),
        DeclareLaunchArgument('x', default_value='0.0'),
        DeclareLaunchArgument('y', default_value='-6.20'),
        DeclareLaunchArgument('z', default_value='0.07'),
        DeclareLaunchArgument('yaw', default_value='0.0'),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(pkg_bringup, 'launch', 'sim_bringup.launch.py')
            ),
            launch_arguments={
                'world': LaunchConfiguration('world'),
                'x': LaunchConfiguration('x'),
                'y': LaunchConfiguration('y'),
                'z': LaunchConfiguration('z'),
                'yaw': LaunchConfiguration('yaw'),
            }.items(),
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(pkg_autonomy, 'launch', 'autonomy.launch.py')
            )
        ),
    ])
