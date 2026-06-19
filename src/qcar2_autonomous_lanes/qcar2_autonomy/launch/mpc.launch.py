#!/usr/bin/env python3
"""Launch the separated MPC architecture."""
import os
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    pkg_share = get_package_share_directory("qcar2_autonomy")
    solver_config = os.path.join(pkg_share, "config", "mpc.yaml")
    node_config = os.path.join(pkg_share, "config", "mpc_nodes.yaml")
    ws_root = Path(os.environ.get("QCAR2_WS", Path.home() / "rosbot_ws"))

    return LaunchDescription([
        DeclareLaunchArgument("config_path", default_value=solver_config),
        DeclareLaunchArgument(
            "waypoints_path",
            default_value=str(ws_root / "track_waypoints.npy"),
        ),
        DeclareLaunchArgument("target_speed", default_value="0.30"),
        DeclareLaunchArgument("prefer_side", default_value="left"),
        DeclareLaunchArgument("loop", default_value="true"),
        DeclareLaunchArgument("record_log", default_value="false"),
        DeclareLaunchArgument("lane_fusion_enabled", default_value="false"),
        DeclareLaunchArgument("lane_model_topic", default_value="/qcar2/lane/model"),
        DeclareLaunchArgument("lane_target_topic", default_value="/planning/validated_target_x"),
        DeclareLaunchArgument("lane_fusion_max_correction_m", default_value="0.30"),
        DeclareLaunchArgument("lane_px_to_m", default_value="0.0015"),
        DeclareLaunchArgument("lane_fusion_gain", default_value="1.15"),
        DeclareLaunchArgument("lane_fusion_alpha", default_value="0.45"),
        DeclareLaunchArgument("lane_fusion_disable_heading_delta_rad", default_value="3.20"),
        Node(
            package="qcar2_autonomy",
            executable="mpc_lidar_obstacle_node",
            name="mpc_lidar_obstacle_node",
            parameters=[node_config],
            output="screen",
        ),
        Node(
            package="qcar2_autonomy",
            executable="mpc_reference_planner_node",
            name="mpc_reference_planner_node",
            parameters=[
                node_config,
                {
                    "config_path": LaunchConfiguration("config_path"),
                    "waypoints_path": LaunchConfiguration("waypoints_path"),
                    "normal_speed": ParameterValue(
                        LaunchConfiguration("target_speed"),
                        value_type=float,
                    ),
                    "prefer_side": LaunchConfiguration("prefer_side"),
                    "loop": ParameterValue(
                        LaunchConfiguration("loop"),
                        value_type=bool,
                    ),
                    "lane_fusion_enabled": ParameterValue(
                        LaunchConfiguration("lane_fusion_enabled"),
                        value_type=bool,
                    ),
                    "lane_model_topic": LaunchConfiguration("lane_model_topic"),
                    "lane_target_topic": LaunchConfiguration("lane_target_topic"),
                    "lane_fusion_max_correction_m": ParameterValue(
                        LaunchConfiguration("lane_fusion_max_correction_m"),
                        value_type=float,
                    ),
                    "lane_px_to_m": ParameterValue(
                        LaunchConfiguration("lane_px_to_m"),
                        value_type=float,
                    ),
                    "lane_fusion_gain": ParameterValue(
                        LaunchConfiguration("lane_fusion_gain"),
                        value_type=float,
                    ),
                    "lane_fusion_alpha": ParameterValue(
                        LaunchConfiguration("lane_fusion_alpha"),
                        value_type=float,
                    ),
                    "lane_fusion_disable_heading_delta_rad": ParameterValue(
                        LaunchConfiguration("lane_fusion_disable_heading_delta_rad"),
                        value_type=float,
                    ),
                },
            ],
            output="screen",
        ),
        Node(
            package="qcar2_autonomy",
            executable="mpc_drive_node",
            name="mpc_drive_node",
            parameters=[
                node_config,
                {
                    "config_path": LaunchConfiguration("config_path"),
                    "target_speed": ParameterValue(
                        LaunchConfiguration("target_speed"),
                        value_type=float,
                    ),
                },
            ],
            output="screen",
        ),
        Node(
            package="qcar2_autonomy",
            executable="mpc_logger_node",
            name="mpc_logger_node",
            parameters=[{"use_sim_time": True}],
            output="screen",
            condition=IfCondition(LaunchConfiguration("record_log")),
        ),
    ])
