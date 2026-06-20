#!/usr/bin/env bash
# Record the driven centreline as MPC waypoints.
# Prereq: sim is up (Terminal 1) AND the car is driving via the RF-DETR
# lane-keeper (./scripts/run_autonomy.sh). This just logs odometry to waypoints.
# Drive one full lap, then Ctrl-C (or it auto-saves on loop closure).
set -e
cd "$(dirname "$0")/.."
WS="$(pwd)"
source /opt/ros/jazzy/setup.bash
source "${WS}/install/setup.bash"
ros2 run qcar2_autonomy record_path_node --ros-args \
    -p use_sim_time:=true \
    -p out_path:=${WAYPOINTS:-${WS}/track_waypoints.npy} \
    -p spacing:=${SPACING:-0.20}
