#!/usr/bin/env bash
# Stop the lane-keeping pipeline started by run_lane_keeping.sh.
# Kills sim, bridges, and all spec nodes. Safe to run repeatedly.

pkill -9 -f 'perception_node'        2>/dev/null
pkill -9 -f 'validation_node'        2>/dev/null
pkill -9 -f 'pid_lane_follower_node' 2>/dev/null
pkill -9 -f 'ai_lane_keeper_node'    2>/dev/null
pkill -9 -f 'bev_lane_detector_node'        2>/dev/null
pkill -9 -f 'lane_controller_node'          2>/dev/null
pkill -9 -f 'state_to_behavior_bridge_node' 2>/dev/null
pkill -9 -f 'cmd_vel_mux_node'              2>/dev/null
pkill -9 -f 'dl_lane_segmentation_node'     2>/dev/null
pkill -9 -f 'yolo_lane_node'                2>/dev/null
pkill -9 -f 'rfdetr_onnx_lane_node'         2>/dev/null
pkill -9 -f 'state_machine_node'       2>/dev/null
pkill -9 -f 'dwa_local_planner_node'   2>/dev/null
pkill -9 -f 'lane_change_planner_node' 2>/dev/null
pkill -9 -f 'kinematics_node'          2>/dev/null
pkill -9 -f 'ros2 launch'            2>/dev/null
pkill -9 -f 'parameter_bridge'       2>/dev/null
pkill -9 -f 'robot_state_publisher'  2>/dev/null
pkill -9 -f 'gz sim'                 2>/dev/null
sleep 1
echo "pipeline torn down"
