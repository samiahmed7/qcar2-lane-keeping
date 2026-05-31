#!/usr/bin/env bash
# Open rqt windows for the lane-keeping pipeline.
#
# Launches three standalone rqt plugins (each in its own window) so you can
# see everything at once:
#   - image_view: live camera feed from the sim
#   - plot:       target_x and angular.z over time
#   - graph:      node/topic connection diagram
#
# Tip: rqt_plot starts blank; type the topic+field in the box at the top, or
# the URL bar already pre-fills them via --args below.
#
# Stop everything with: scripts/stop_rqt.sh  (or just close the windows).

set -e
cd "$(dirname "$0")/.."
WS="$(pwd)"
source /opt/ros/jazzy/setup.bash
source "${WS}/install/setup.bash"

mkdir -p /tmp/qcar2_pipeline
LOG=/tmp/qcar2_pipeline

# 1. Camera feed (raw sim image). Switch the dropdown to /qcar2/lane/debug_image
# if you want the BEV-detector overlay instead.
echo "[1/3] rqt_image_view -> /qcar2/front_camera/image"
nohup ros2 run rqt_image_view rqt_image_view /qcar2/front_camera/image \
    > "${LOG}/rqt_image.log" 2>&1 &
disown

# 2. Plot the controller's tracking error and steering command.
# rqt_plot can subscribe to multiple scalar fields at once.
echo "[2/3] rqt_plot -> target_x + cmd_vel.angular.z + odom.x"
nohup ros2 run rqt_plot rqt_plot \
    /planning/validated_target_x/data \
    /model/qcar2/cmd_vel/angular/z \
    /model/qcar2/odometry/pose/pose/position/y \
    > "${LOG}/rqt_plot.log" 2>&1 &
disown

# 3. Node graph: confirms perception -> validation -> pid_lane_follower -> sim.
echo "[3/3] rqt_graph"
nohup ros2 run rqt_graph rqt_graph \
    > "${LOG}/rqt_graph.log" 2>&1 &
disown

sleep 2
echo
echo "rqt windows launched. Logs in ${LOG}/rqt_*.log"
echo "Close the windows to stop, or run: scripts/stop_rqt.sh"
