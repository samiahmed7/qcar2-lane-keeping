#!/usr/bin/env python3
"""Launch the camera bridge + lane follower node together."""
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

    camera_info_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='camera_info_bridge',
        arguments=[
            '/qcar2/front_camera/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo',
        ],
        output='screen',
    )

    lane_follower = Node(
        package='qcar2_line_tracker',
        executable='lane_follower',
        name='lane_follower',
        parameters=[{
            'speed': 0.15,
            'crawl_speed': 0.05,
            'kp': 0.0022,
            'kd': 0.0030,
            'steering_limit': 0.5,
            'roi_start': 0.45,
            'look_ahead_top': 0.0,
            'look_ahead_bottom': 0.35,
            'white_v_low': 180,
            'min_pixels_per_side': 50,
            'default_lane_width_px': 320,
            'midpoint_smoothing': 0.70,
        }],
        output='screen',
    )

    return LaunchDescription([camera_bridge, camera_info_bridge, lane_follower])
