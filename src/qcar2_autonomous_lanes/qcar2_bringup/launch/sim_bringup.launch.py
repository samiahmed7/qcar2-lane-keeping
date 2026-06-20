#!/usr/bin/env python3
"""Launch Gazebo, spawn QCar2, and bridge simulation topics."""
import os

from ament_index_python.packages import get_package_prefix, get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def launch_setup(context, *args, **kwargs):
    world = LaunchConfiguration('world').perform(context)
    x = LaunchConfiguration('x').perform(context)
    y = LaunchConfiguration('y').perform(context)
    z = LaunchConfiguration('z').perform(context)
    yaw = LaunchConfiguration('yaw').perform(context)
    headless = LaunchConfiguration('headless')

    pkg_bringup = get_package_share_directory('qcar2_bringup')
    pkg_description = get_package_share_directory('qcar2_description')
    pkg_worlds = get_package_share_directory('qcar2_worlds')
    desc_share_parent = os.path.join(get_package_prefix('qcar2_description'), 'share')
    worlds_share_parent = os.path.join(get_package_prefix('qcar2_worlds'), 'share')

    urdf_file = os.path.join(pkg_description, 'urdf', 'QCar2.urdf')
    bridge_config = os.path.join(pkg_bringup, 'config', 'bridge.yaml')
    if not os.path.exists(urdf_file):
        raise FileNotFoundError(f'QCar2 URDF not found: {urdf_file}')
    if not os.path.exists(bridge_config):
        raise FileNotFoundError(f'Bridge config not found: {bridge_config}')

    with open(urdf_file, 'r', encoding='utf-8') as f:
        robot_description = f.read()

    resource_paths = [
        desc_share_parent,
        worlds_share_parent,
        pkg_description,
        os.path.join(pkg_description, 'meshes'),
        pkg_worlds,
        os.path.join(pkg_worlds, 'models'),
        os.path.join(pkg_worlds, 'maps'),
        os.environ.get('GZ_SIM_RESOURCE_PATH', ''),
    ]

    actions = [
        SetEnvironmentVariable(
            name='GZ_SIM_RESOURCE_PATH',
            value=':'.join(path for path in resource_paths if path),
        ),
    ]

    gz_vendor_lib = '/opt/ros/jazzy/opt/gz_sim_vendor/lib'
    if os.path.isdir(gz_vendor_lib):
        actions.append(SetEnvironmentVariable(
            name='GZ_SIM_SYSTEM_PLUGIN_PATH',
            value=':'.join(
                path for path in [
                    gz_vendor_lib,
                    os.environ.get('GZ_SIM_SYSTEM_PLUGIN_PATH', ''),
                ] if path
            ),
        ))

    actions.extend([
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(pkg_worlds, 'launch', 'sim_world.launch.py')
            ),
            launch_arguments={
                'world': world,
                'extra_resource_paths': ':'.join(path for path in resource_paths if path),
                'headless': headless,
            }.items(),
        ),
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            parameters=[{
                'robot_description': robot_description,
                'use_sim_time': True,
            }],
            output='screen',
        ),
        Node(
            package='ros_gz_sim',
            executable='create',
            name='spawn_qcar2',
            arguments=[
                '-world', world,
                '-name', 'qcar2',
                '-allow_renaming=false',
                '-param', 'robot_description',
                '-x', x,
                '-y', y,
                '-z', z,
                '-Y', yaw,
            ],
            parameters=[{
                'robot_description': robot_description,
            }],
            output='screen',
        ),
        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            name='qcar2_bridge',
            parameters=[{'config_file': bridge_config}],
            output='screen',
        ),
    ])

    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('world', default_value='lab_track'),
        DeclareLaunchArgument('headless', default_value='false'),
        DeclareLaunchArgument('x', default_value='0.0'),
        DeclareLaunchArgument('y', default_value='-6.20'),
        DeclareLaunchArgument('z', default_value='0.07'),
        DeclareLaunchArgument('yaw', default_value='0.0'),
        OpaqueFunction(function=launch_setup),
    ])
