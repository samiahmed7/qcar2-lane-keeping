#!/usr/bin/env python3
"""Launch the QCar2 lab track world in Gazebo."""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def launch_setup(context, *args, **kwargs):
    world = LaunchConfiguration('world').perform(context)
    extra_resource_paths = LaunchConfiguration('extra_resource_paths').perform(context)
    headless = LaunchConfiguration('headless').perform(context).lower() in (
        '1', 'true', 'yes', 'on'
    )

    pkg_worlds = get_package_share_directory('qcar2_worlds')
    pkg_ros_gz = get_package_share_directory('ros_gz_sim')
    world_file = os.path.join(pkg_worlds, 'worlds', f'{world}.sdf')
    if not os.path.exists(world_file):
        raise FileNotFoundError(f'Gazebo world not found: {world_file}')

    resource_paths = [
        os.path.dirname(pkg_worlds),
        pkg_worlds,
        os.path.join(pkg_worlds, 'models'),
        os.path.join(pkg_worlds, 'maps'),
        *[path for path in extra_resource_paths.split(':') if path],
        os.environ.get('GZ_SIM_RESOURCE_PATH', ''),
    ]

    gz_args = f'-r -v 4 {world_file}'
    if headless:
        gz_args = f'-r -s -v 4 {world_file}'

    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_ros_gz, 'launch', 'gz_sim.launch.py')
        ),
        launch_arguments={'gz_args': gz_args}.items(),
    )

    return [
        SetEnvironmentVariable(
            name='GZ_SIM_RESOURCE_PATH',
            value=':'.join(path for path in resource_paths if path),
        ),
        gz_sim,
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('world', default_value='lab_track'),
        DeclareLaunchArgument('extra_resource_paths', default_value=''),
        DeclareLaunchArgument('headless', default_value='false'),
        OpaqueFunction(function=launch_setup),
    ])
