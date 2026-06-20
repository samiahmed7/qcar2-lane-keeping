#!/usr/bin/env python3
"""Full simulation lane-keeping: Gazebo + bridge + perception + controller."""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():
    pkg_bringup = get_package_share_directory('qcar2_bringup')
    pkg_autonomy = get_package_share_directory('qcar2_autonomy')

    sim_bringup = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_bringup, 'launch', 'sim_bringup.launch.py')
        ),
    )
    lane_keep = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_autonomy, 'launch', 'lane_keep.launch.py')
        ),
    )

    return LaunchDescription([sim_bringup, lane_keep])
