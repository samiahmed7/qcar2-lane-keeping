#!/usr/bin/env python3
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def launch_setup(context, *args, **kwargs):
    world = LaunchConfiguration('world').perform(context)
    x     = LaunchConfiguration('x').perform(context)
    y     = LaunchConfiguration('y').perform(context)
    z     = LaunchConfiguration('z').perform(context)
    yaw   = LaunchConfiguration('yaw').perform(context)

    pkg_qcar2        = get_package_share_directory('qcar2')
    pkg_qcar2_worlds = get_package_share_directory('qcar2_worlds')
    pkg_ros_gz       = get_package_share_directory('ros_gz_sim')

    urdf_file  = os.path.join(pkg_qcar2, 'urdf', 'QCar2.urdf')
    world_file = os.path.join(pkg_qcar2_worlds, 'worlds', f'{world}.sdf')

    with open(urdf_file, 'r') as f:
        robot_description = f.read()

    # Gazebo
    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_ros_gz, 'launch', 'gz_sim.launch.py')
        ),
        launch_arguments={'gz_args': f'-r {world_file}'}.items(),
    )

    # Robot state publisher
    rsp = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{'robot_description': robot_description, 'use_sim_time': True}],
        output='screen',
    )

    # Spawn robot
    spawn = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-name', 'qcar2',
            '-string', robot_description,
            '-x', x, '-y', y, '-z', z, '-Y', yaw,
        ],
        output='screen',
    )

    # Bridge: ROS cmd_vel → Gazebo, Gazebo odometry/TF/lidar → ROS
    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='gz_bridge',
        arguments=[
            '/model/qcar2/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist',
            '/model/qcar2/odometry@nav_msgs/msg/Odometry[gz.msgs.Odometry',
            '/model/qcar2/tf@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V',
            '/qcar2/lidar/scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan',
            '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
        ],
        output='screen',
    )

    return [gz_sim, rsp, spawn, bridge]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('world', default_value='qcar_oval'),
        DeclareLaunchArgument('x',   default_value='0.0'),
        DeclareLaunchArgument('y',   default_value='-6.20'),
        DeclareLaunchArgument('z',   default_value='0.07'),
        DeclareLaunchArgument('yaw', default_value='0.0'),
        OpaqueFunction(function=launch_setup),
    ])
