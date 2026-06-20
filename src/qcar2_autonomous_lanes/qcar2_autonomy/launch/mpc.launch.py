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
        DeclareLaunchArgument("target_speed", default_value="0.55"),
        DeclareLaunchArgument("odom_topic", default_value="/model/qcar2/odometry"),
        DeclareLaunchArgument("scan_topic", default_value="/qcar2/lidar/scan"),
        DeclareLaunchArgument("prefer_side", default_value="left"),
        DeclareLaunchArgument("loop", default_value="true"),
        DeclareLaunchArgument("record_log", default_value="false"),
        DeclareLaunchArgument("sensor_noise_enabled", default_value="false"),
        DeclareLaunchArgument("raw_odom_topic", default_value="/model/qcar2/odometry"),
        DeclareLaunchArgument("raw_scan_topic", default_value="/qcar2/lidar/scan"),
        DeclareLaunchArgument("noisy_odom_topic", default_value="/mpc/noisy_odometry"),
        DeclareLaunchArgument("noisy_scan_topic", default_value="/mpc/noisy_scan"),
        DeclareLaunchArgument("odom_position_stddev_m", default_value="0.015"),
        DeclareLaunchArgument("odom_yaw_stddev_rad", default_value="0.010"),
        DeclareLaunchArgument("scan_range_stddev_m", default_value="0.015"),
        DeclareLaunchArgument("scan_dropout_probability", default_value="0.01"),
        DeclareLaunchArgument("speed_profile_enabled", default_value="true"),
        DeclareLaunchArgument("max_straight_speed", default_value="0.80"),
        DeclareLaunchArgument("min_curve_speed", default_value="0.18"),
        DeclareLaunchArgument("max_lateral_accel_mps2", default_value="0.45"),
        DeclareLaunchArgument("curvature_lookahead_m", default_value="1.20"),
        DeclareLaunchArgument("lane_fusion_enabled", default_value="false"),
        DeclareLaunchArgument("lane_model_topic", default_value="/qcar2/lane/model"),
        DeclareLaunchArgument("lane_target_topic", default_value="/planning/validated_target_x"),
        DeclareLaunchArgument("lane_fusion_max_correction_m", default_value="0.06"),
        DeclareLaunchArgument("lane_fusion_max_step_m", default_value="0.006"),
        DeclareLaunchArgument("lane_px_to_m", default_value="0.0015"),
        DeclareLaunchArgument("lane_fusion_gain", default_value="0.35"),
        DeclareLaunchArgument("lane_fusion_alpha", default_value="0.25"),
        DeclareLaunchArgument("lane_fusion_min_confidence", default_value="0.30"),
        DeclareLaunchArgument("lane_fusion_hold_low_confidence", default_value="true"),
        DeclareLaunchArgument("lane_fusion_disable_heading_delta_rad", default_value="3.20"),
        DeclareLaunchArgument("command_smoothing_enabled", default_value="true"),
        DeclareLaunchArgument("omega_smoothing_alpha", default_value="0.80"),
        DeclareLaunchArgument("max_omega_rate_rps2", default_value="5.0"),
        Node(
            package="qcar2_autonomy",
            executable="mpc_sensor_noise_node",
            name="mpc_sensor_noise_node",
            parameters=[
                {
                    "use_sim_time": True,
                    "raw_odom_topic": LaunchConfiguration("raw_odom_topic"),
                    "raw_scan_topic": LaunchConfiguration("raw_scan_topic"),
                    "noisy_odom_topic": LaunchConfiguration("noisy_odom_topic"),
                    "noisy_scan_topic": LaunchConfiguration("noisy_scan_topic"),
                    "odom_position_stddev_m": ParameterValue(
                        LaunchConfiguration("odom_position_stddev_m"),
                        value_type=float,
                    ),
                    "odom_yaw_stddev_rad": ParameterValue(
                        LaunchConfiguration("odom_yaw_stddev_rad"),
                        value_type=float,
                    ),
                    "scan_range_stddev_m": ParameterValue(
                        LaunchConfiguration("scan_range_stddev_m"),
                        value_type=float,
                    ),
                    "scan_dropout_probability": ParameterValue(
                        LaunchConfiguration("scan_dropout_probability"),
                        value_type=float,
                    ),
                },
            ],
            output="screen",
            condition=IfCondition(LaunchConfiguration("sensor_noise_enabled")),
        ),
        Node(
            package="qcar2_autonomy",
            executable="mpc_lidar_obstacle_node",
            name="mpc_lidar_obstacle_node",
            parameters=[
                node_config,
                {
                    "scan_topic": LaunchConfiguration("scan_topic"),
                    "odom_topic": LaunchConfiguration("odom_topic"),
                },
            ],
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
                    "odom_topic": LaunchConfiguration("odom_topic"),
                    "normal_speed": ParameterValue(
                        LaunchConfiguration("target_speed"),
                        value_type=float,
                    ),
                    "speed_profile_enabled": ParameterValue(
                        LaunchConfiguration("speed_profile_enabled"),
                        value_type=bool,
                    ),
                    "max_straight_speed": ParameterValue(
                        LaunchConfiguration("max_straight_speed"),
                        value_type=float,
                    ),
                    "min_curve_speed": ParameterValue(
                        LaunchConfiguration("min_curve_speed"),
                        value_type=float,
                    ),
                    "max_lateral_accel_mps2": ParameterValue(
                        LaunchConfiguration("max_lateral_accel_mps2"),
                        value_type=float,
                    ),
                    "curvature_lookahead_m": ParameterValue(
                        LaunchConfiguration("curvature_lookahead_m"),
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
                    "lane_fusion_max_step_m": ParameterValue(
                        LaunchConfiguration("lane_fusion_max_step_m"),
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
                    "lane_fusion_min_confidence": ParameterValue(
                        LaunchConfiguration("lane_fusion_min_confidence"),
                        value_type=float,
                    ),
                    "lane_fusion_hold_low_confidence": ParameterValue(
                        LaunchConfiguration("lane_fusion_hold_low_confidence"),
                        value_type=bool,
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
                    "odom_topic": LaunchConfiguration("odom_topic"),
                    "target_speed": ParameterValue(
                        LaunchConfiguration("target_speed"),
                        value_type=float,
                    ),
                    "command_smoothing_enabled": ParameterValue(
                        LaunchConfiguration("command_smoothing_enabled"),
                        value_type=bool,
                    ),
                    "omega_smoothing_alpha": ParameterValue(
                        LaunchConfiguration("omega_smoothing_alpha"),
                        value_type=float,
                    ),
                    "max_omega_rate_rps2": ParameterValue(
                        LaunchConfiguration("max_omega_rate_rps2"),
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
