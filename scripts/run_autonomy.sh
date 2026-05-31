#!/usr/bin/env bash
# Autonomy ONLY (no sim, no obstacle): RF-DETR lane-keep + LiDAR-tracked
# obstacle avoidance. Run this AFTER Gazebo is already up (sim_bringup).
#
#   detector -> PID lane-keep -> mux ─┐
#   state machine (LiDAR) ────────────┤-> /model/qcar2/cmd_vel
#   S-curve planner ──────────────────┘
#
# Stop with: scripts/stop_lane_keeping.sh
set -e
cd "$(dirname "$0")/.."
WS="$(pwd)"
source /opt/ros/jazzy/setup.bash
source "${WS}/install/setup.bash"
LOG=/tmp/qcar2_pipeline; mkdir -p "${LOG}"

echo "Waiting for the sim camera (/qcar2/front_camera/image)..."
until timeout 2 ros2 topic list 2>/dev/null | grep -q '/qcar2/front_camera/image'; do
    sleep 2
done
echo "Sim is up. Starting autonomy nodes..."

# RF-DETR ONNX (GPU) -> /planning/validated_target_x
NV_LIBS=$(find /usr/local/lib/python3.12/dist-packages/nvidia -maxdepth 3 -type d -name lib 2>/dev/null | tr '\n' ':')
export LD_LIBRARY_PATH="${NV_LIBS}${LD_LIBRARY_PATH}"
nohup ros2 run qcar2_autonomy rfdetr_onnx_lane_node --ros-args \
    -p use_sim_time:=true \
    -p image_topic:=/qcar2/front_camera/image \
    -p target_topic:=/planning/validated_target_x \
    -p onnx_path:=${WS}/weights/car_track_v3_lane.onnx \
    -p classes_path:=${WS}/weights/car_track_v3_lane.classes.txt \
    -p lane_class_name:=lane2 \
    -p provider:=${RFDETR_PROVIDER:-CUDAExecutionProvider} \
    -p confidence_threshold:=${RFDETR_CONF:-0.5} \
    > "${LOG}/rfdetr.log" 2>&1 & disown

# PID lane-keeper (vision) -> mux input; creeps if it briefly loses the lane
nohup ros2 run qcar2_autonomy pid_lane_follower_node --ros-args \
    -p use_sim_time:=true \
    -p cmd_vel_topic:=/qcar2/control/raw_cmd_vel \
    -p base_speed:=${PID_SPEED:-0.20} -p kp:=${PID_KP:-1.4} -p ki:=0.0 -p kd:=0.0 \
    -p max_angular:=${PID_MAX_W:-1.5} -p target_timeout_sec:=1.0 \
    -p lost_creep_speed:=${LOST_CREEP:-0.10} \
    > "${LOG}/pid.log" 2>&1 & disown

# Open-loop S-curve planner (owns the swerve; vision can't steer mid-yaw)
nohup ros2 run qcar2_autonomy lane_change_planner_node --ros-args \
    -p use_sim_time:=true \
    -p cmd_topic:=/qcar2/control/maneuver_cmd_vel \
    -p cruise_speed:=${MANEUVER_SPEED:-0.30} -p step_angular:=${MANEUVER_OMEGA:-1.0} \
    -p turn_duration:=${MANEUVER_TURN_T:-0.6} -p straight_duration:=${MANEUVER_STRAIGHT_T:-2.2} \
    -p max_angular:=1.5 \
    > "${LOG}/lane_change_planner.log" 2>&1 & disown

# Mux: planner owns CHANGE/RETURN, PID owns KEEP -> single /model/qcar2/cmd_vel
nohup ros2 run qcar2_autonomy cmd_vel_mux_node --ros-args \
    -p use_sim_time:=true \
    -p lane_keep_topic:=/qcar2/control/raw_cmd_vel \
    -p maneuver_topic:=/qcar2/control/maneuver_cmd_vel \
    -p output_cmd_topic:=/model/qcar2/cmd_vel \
    -p route_by_state:=true \
    > "${LOG}/cmd_mux.log" 2>&1 & disown

# State machine: LiDAR-tracked detect -> measure -> pick side -> shift -> return
nohup ros2 run qcar2_autonomy state_machine_node --ros-args \
    -p use_sim_time:=true \
    -p scan_topic:=/qcar2/lidar/scan -p odom_topic:=/model/qcar2/odometry \
    -p lidar_tracked_avoid:=true -p oneway_lane_change:=${ONEWAY_LANE_CHANGE:-true} \
    -p track_range_m:=${TRACK_RANGE:-0.55} -p pass_confirm_sec:=${PASS_CONFIRM:-0.4} \
    -p lane_change_min_sec:=${CHANGE_MIN_S:-1.5} -p return_duration_sec:=${RETURN_DUR:-3.5} \
    -p pass_clearance_m:=${PASS_CLEARANCE:-0.8} \
    -p side_sector_min_deg:=${SIDE_MIN_DEG:-30.0} -p side_sector_max_deg:=${SIDE_MAX_DEG:-150.0} \
    -p lane_change_cooldown_sec:=${LANE_CHANGE_COOLDOWN:-3.0} \
    -p prefer_side:=${FORCE_SIDE:-auto} -p vehicle_half_width_m:=${VEHICLE_HALF_WIDTH:-0.18} \
    -p clearance_margin_m:=${CLEAR_MARGIN:-0.20} -p min_obstacle_width_m:=${MIN_OBSTACLE_WIDTH:-0.25} \
    -p max_straight_dur:=${MAX_STRAIGHT_DUR:-4.5} \
    > "${LOG}/state_machine.log" 2>&1 & disown

sleep 3
echo "Autonomy up. Detector loads in ~10 s (watch: tail -f ${LOG}/rfdetr.log)."
echo "Stop everything with: scripts/stop_lane_keeping.sh"
