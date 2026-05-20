#!/usr/bin/env python3
"""
Launch the lane_filter node (IPM + sliding-window + Clothoid EKF).

This runs in parallel with the existing lane_follower. It subscribes to
the camera image and the QCar2 odometry, but does NOT publish cmd_vel.

Make sure simulation.launch.py is running (provides odometry bridge) and
the camera bridge is up. The camera bridge is also launched here for
convenience — if you already have camera_bridge.launch.py running, you
can remove the camera_bridge node below to avoid a duplicate.
"""
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    camera_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='camera_bridge_for_filter',
        arguments=[
            '/qcar2/front_camera/image@sensor_msgs/msg/Image[gz.msgs.Image',
        ],
        output='screen',
    )

    lane_filter = Node(
        package='qcar2_line_tracker',
        executable='lane_filter',
        name='lane_filter',
        output='screen',
        parameters=[{
            'image_topic': '/qcar2/front_camera/image',
            'odom_topic': '/model/qcar2/odometry',
            'white_v_low': 180,
            'n_windows': 9,
            'window_margin': 50,
            'min_pixels_per_side': 100,
            'confident_pixels_per_side': 600,
            'min_windows_confident': 3,
            'default_lane_full_width_m': 0.80,
            'lane_width_alpha': 0.10,
            'q_c0': 0.001,
            'q_c1': 0.001,
            'q_c2': 0.0001,
            'q_c3': 0.00001,
            'r_meas': 0.01,
        }],
    )

    return LaunchDescription([camera_bridge, lane_filter])
