# This is the launch file that starts up the basic QCar2 nodes

import subprocess
import os
from launch import LaunchDescription
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node
from launch_ros.actions import Node
from launch.substitutions import Command,FindExecutable
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
        
    lidar_node = Node(
            package='qcar2_nodes',
            executable='lidar',
            name='Lidar'
        )
    
    qcar2_nav2_converter = Node(
    package='qcar2_nodes',
    executable='nav2_qcar2_converter',
    name='nav2_qcar2_converter',
    )
    
    realsense_camera_node = Node(
            package='qcar2_nodes',
            executable='rgbd',
            name='RealsenseCamera'
        )
    
    csi_camera_node = Node(
            package='qcar2_nodes',
            executable='csi',
            name='csi_camera'
        )
    
    qcar2_hardware = Node(
            package='qcar2_nodes',
            executable='qcar2_hardware',
            name='qcar2_hardware',
        )

    qcar2_sensor_tf_node = Node(
        package='qcar2_nodes',
        executable='fixed_lidar_frame',
        name='fixed_lidar_frame')
    
    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{
        'robot_description': ParameterValue(
            Command([
                FindExecutable(name='xacro'),
                ' ',
                os.path.join(get_package_share_directory('qcar_lane_pkg'), 'urdf', 'qcar_ros2_original.urdf.xacro')
            ]),
            
            value_type=str
        )
    }],
)
     
    return LaunchDescription([
        lidar_node,
        qcar2_nav2_converter,
        qcar2_sensor_tf_node,
        realsense_camera_node,
        # csi_camera_node,
        qcar2_hardware,
        robot_state_publisher_node
    ])
