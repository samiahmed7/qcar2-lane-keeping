#!/usr/bin/env bash
# QCar2 lane-keeping + ONE-WAY obstacle avoidance demo.
#
# The car lane-keeps with the RF-DETR ONNX detector, detects a box ahead via
# LiDAR, changes lane around it with an open-loop S-curve, and CONTINUES in the
# new lane (no return).
#
# Runs HEADLESS by default: the Gazebo GUI + RF-DETR CUDA load freezes the sim
# on a single laptop GPU. Watch the live overlay instead with:
#     ros2 run rqt_image_view rqt_image_view /rfdetr_lane/debug_image
#
# Usage:
#     scripts/run_avoid.sh                 # box 2.0 m ahead (default)
#     OBSTACLE=3.0,-6.20 scripts/run_avoid.sh   # box at world (x,y)
#     GUI=1 scripts/run_avoid.sh           # try the Gazebo GUI (may freeze)
#
# Stop everything with:  scripts/stop_lane_keeping.sh
set -e
cd "$(dirname "$0")/.."

# Car spawns at world (0, -6.20). Default obstacle is 2.0 m straight ahead.
export SPAWN_OBSTACLE_AT="${OBSTACLE:-2.0,-6.20}"
export LANE_KEEPER=rfdetr
export HEADLESS="$([ "${GUI:-0}" = "1" ] && echo false || echo true)"
export SHOW_3LANE=0          # skip the heavy classical debug detector
export SHOW_DL_SEG=0         # skip the YOLO debug overlay

exec bash scripts/run_lane_keeping.sh
