#!/usr/bin/env python3
"""Bridge Gazebo camera topics to ROS so rqt_image_view can display them."""
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    camera_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='camera_bridge',
        arguments=[
            '/qcar2/front_camera/image@sensor_msgs/msg/Image[gz.msgs.Image',
            '/qcar2/front_camera/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo',
        ],
        output='screen',
    )

    return LaunchDescription([camera_bridge])
