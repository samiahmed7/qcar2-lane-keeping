#!/usr/bin/env python3
"""Launch the camera bridge + IPM calibration tool together."""
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    camera_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='camera_bridge',
        arguments=[
            '/qcar2/front_camera/image@sensor_msgs/msg/Image[gz.msgs.Image',
        ],
        output='screen',
    )

    calibrator = Node(
        package='qcar2_line_tracker',
        executable='ipm_calibrate',
        name='ipm_calibrate',
        output='screen',
        parameters=[{
            'real_width_m': 0.80,
            'real_length_m': 1.50,
            'image_topic': '/qcar2/front_camera/image',
        }],
    )

    return LaunchDescription([camera_bridge, calibrator])
