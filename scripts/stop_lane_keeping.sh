#!/usr/bin/env bash
# Stop the BEV-MPC lane-keeping pipeline. Safe to run repeatedly.

pkill -9 -f 'bev_lane_detector_node'        2>/dev/null
pkill -9 -f 'mpc_drive_node'                 2>/dev/null
pkill -9 -f 'mpc_reference_planner_node'     2>/dev/null
pkill -9 -f 'mpc_lidar_obstacle_node'        2>/dev/null
pkill -9 -f 'mpc_sensor_noise_node'          2>/dev/null
pkill -9 -f 'mpc_logger_node'                2>/dev/null
pkill -9 -f 'ros2 launch'                    2>/dev/null
pkill -9 -f 'parameter_bridge'               2>/dev/null
pkill -9 -f 'robot_state_publisher'          2>/dev/null
pkill -9 -f 'gz sim'                         2>/dev/null
sleep 1
echo "pipeline torn down"
