#!/usr/bin/env python3
"""
Drive the QCar2 using the Clothoid EKF + Stanley controller.

This launches:
  - the camera bridge (Gz→ROS for /qcar2/front_camera/image)
  - the lane_filter node with drive_enable=true so it publishes cmd_vel

** Before running this **
  - Make sure simulation.launch.py is up.
  - STOP any other driver (lane_follower) so it doesn't fight this one
    for /model/qcar2/cmd_vel.

Tunable parameters
------------------
  drive_speed         forward velocity (m/s)         default 0.15
  drive_k_lat         Stanley cross-track gain       default 1.5
  drive_k_head        heading-error gain             default 1.0
  drive_k_ff_curv     curvature feed-forward gain    default 0.5
  drive_lookahead_m   distance for curvature FF      default 0.7
  drive_steering_limit  max |steering| (rad)         default 0.5
  drive_rate_limit    max change per frame (rad)     default 0.06
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
            # Detection
            'image_topic': '/qcar2/front_camera/image',
            'odom_topic':  '/model/qcar2/odometry',
            'white_v_low': 180,
            'n_windows': 9,
            'window_margin': 50,
            'min_pixels_per_side': 100,
            'confident_pixels_per_side': 600,
            'min_windows_confident': 3,
            'default_lane_full_width_m': 0.80,
            'lane_width_alpha': 0.10,

            # EKF tuning
            'q_c0': 0.001,
            'q_c1': 0.001,
            'q_c2': 0.0001,
            'q_c3': 0.00001,
            'r_meas': 0.01,

            # Steering — actively driving the car (tuned for smoothness)
            'drive_enable': True,
            'drive_speed': 0.15,
            'drive_crawl_speed': 0.05,
            'drive_k_lat': 0.65,           # cross-track gain
            'drive_k_head': 0.80,          # heading-error gain
            'drive_k_ff_curv': 0.20,       # curvature feed-forward (low → less over-shoot)
            'drive_lookahead_m': 0.70,
            'drive_steering_limit': 0.5,
            'drive_rate_limit': 0.04,      # max rad change per frame
        }],
    )

    return LaunchDescription([camera_bridge, lane_filter])
