#!/usr/bin/env bash
# Integrated test: MPC navigation + BEV lane-keeping fusion, headless, on
# lab_track. Records the run, samples /mpc/lane_fusion_status, then plots.
cd "$(dirname "$0")/.." || exit 1
WS="$(pwd)"
source /opt/ros/jazzy/setup.bash
source "$WS/install/setup.bash"
LOG=/tmp/bevfusion; mkdir -p "$LOG"
SPEED="${SPEED:-0.30}"
DUR="${DUR:-180}"

cleanup() {
  ./scripts/stop_lane_keeping.sh >/dev/null 2>&1
  pkill -9 -f 'gz sim' >/dev/null 2>&1
  pkill -9 -f bev_lane_detector >/dev/null 2>&1
  pkill -9 -f mpc_drive_node >/dev/null 2>&1
  pkill -9 -f mpc_reference_planner_node >/dev/null 2>&1
  pkill -9 -f mpc_lidar_obstacle_node >/dev/null 2>&1
  pkill -9 -f mpc_logger_node >/dev/null 2>&1
}

echo "[1] teardown"; cleanup; sleep 3
echo "[2] sim lab_track headless"
nohup ros2 launch qcar2_bringup sim_bringup.launch.py world:=lab_track headless:=true \
  >"$LOG/sim.log" 2>&1 &
echo "[3] wait for odometry"
for i in $(seq 1 45); do ros2 topic list 2>/dev/null | grep -q '/model/qcar2/odometry' && break; sleep 2; done
sleep 5
echo "[4] BEV lane detector (calibrated IPM)"
nohup ros2 run qcar2_autonomy bev_lane_detector_node --ros-args \
  -p use_sim_time:=true -p image_topic:=/qcar2/front_camera/image \
  -p lane_model_topic:=/qcar2/lane/model -p debug_topic:=/qcar2/lane/debug_image \
  -p use_ipm:=true >"$LOG/bev.log" 2>&1 &
sleep 5
echo "[5] MPC stack + lane fusion (BEV source), loop, speed=$SPEED"
nohup ros2 launch qcar2_autonomy mpc.launch.py \
  waypoints_path:=$WS/track_waypoints.npy target_speed:=$SPEED loop:=true \
  lane_fusion_enabled:=true lane_model_topic:=/qcar2/lane/model \
  lane_fusion_disable_heading_delta_rad:=0.30 >"$LOG/mpc.log" 2>&1 &
echo "[6] record ${DUR}s + sample fusion status"
( for i in $(seq 1 8); do
    printf "t~%s: " "$i"
    timeout 4 ros2 topic echo /mpc/lane_fusion_status --field data --once 2>/dev/null || echo "(none)"
    sleep $((DUR / 8))
  done ) >"$LOG/fusion.log" 2>&1 &
timeout -s INT $((DUR + 15)) ros2 run qcar2_autonomy mpc_logger_node --ros-args \
  -p use_sim_time:=true -p duration_sec:=${DUR}.0 -p out_path:=$WS/mpc_run_log.npz \
  >"$LOG/logger.log" 2>&1
echo "[7] stop + plot"; cleanup; sleep 2
python3 scripts/plot_mpc_run.py "$WS/mpc_run_log.npz" "$WS/track_waypoints.npy" \
  >"$LOG/plot.log" 2>&1
echo "--- logger ---"; tail -2 "$LOG/logger.log"
echo "--- plot ---"; tail -3 "$LOG/plot.log"
echo "--- fusion status samples ---"; cat "$LOG/fusion.log"
echo "--- mpc problems ---"; grep -iE "infeasible|solve error|BLOCK|exception" "$LOG/mpc.log" | tail -5
echo "BEVFUSION DONE"
