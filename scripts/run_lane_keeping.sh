#!/usr/bin/env bash
# Bring up Gazebo + the spec-compliant lane-keeping pipeline.
#
# Pipeline (closed loop):
#   sim camera /qcar2/front_camera/image
#     -> perception_node   -> /perception/raw_lane_data
#     -> validation_node   -> /planning/validated_target_x
#     -> pid_lane_follower -> /model/qcar2/cmd_vel (Twist)
#     -> sim car
#
# After it boots, the Gazebo GUI window will show the QCar2 driving down the
# lab_track. Tear down with scripts/stop_lane_keeping.sh.

set -e
cd "$(dirname "$0")/.."
WS="$(pwd)"

if [ ! -f "${WS}/install/setup.bash" ]; then
    echo "Workspace not built. Running colcon build first..."
    source /opt/ros/jazzy/setup.bash
    colcon build --symlink-install
fi

source /opt/ros/jazzy/setup.bash
source "${WS}/install/setup.bash"

mkdir -p /tmp/qcar2_pipeline
LOG=/tmp/qcar2_pipeline

# Set up the lane-keeper mode here so later steps can branch on it.
# ai  = HSHL ResNet18 regression (default, but flawed for our scene)
# pid = classical perception + validation + PID
# bev = classical 3-lane detector + lane_controller
# yolo/dl = YOLO best.pt segmentation + PID (DL drives the car)
LANE_KEEPER="${LANE_KEEPER:-ai}"
echo "Lane keeper mode: ${LANE_KEEPER}"
if [ -z "${SHOW_3LANE+x}" ]; then
    if [ "$LANE_KEEPER" = "dl" ] || [ "$LANE_KEEPER" = "yolo" ]; then
        SHOW_3LANE=0
    else
        SHOW_3LANE=1
    fi
fi

# 1. Gazebo sim
echo "[1/6] launching Gazebo (lab_track)..."
nohup ros2 launch qcar2_bringup sim_bringup.launch.py headless:=${HEADLESS:-false} > "${LOG}/sim.log" 2>&1 &
disown
echo "    waiting for /qcar2/front_camera/image..."
until timeout 2 ros2 topic list 2>/dev/null | grep -q '/qcar2/front_camera/image'; do
    sleep 2
done
sleep 3

# 1b. Optional: spawn an obstacle box before any perception/control node starts,
# so the car finds it the moment PID begins driving. Format:
#   SPAWN_OBSTACLE_AT="x,y"          # one box at world (x, y)
#   SPAWN_OBSTACLE_AT="x1,y1;x2,y2"  # multiple, semicolon separated
# Skip when unset/empty. Coordinates are world-frame metres; the QCar2 spawn
# is (0, -6.20) yaw=0, so "1.5,-6.20" is 1.5 m straight ahead at start.
if [ -n "${SPAWN_OBSTACLE_AT:-}" ]; then
    IFS=';' read -ra _OBSTACLES <<< "${SPAWN_OBSTACLE_AT}"
    _idx=0
    for _pt in "${_OBSTACLES[@]}"; do
        _x="${_pt%%,*}"; _y="${_pt##*,}"
        _idx=$((_idx + 1))
        echo "[1b] pre-spawn obstacle #${_idx} at (${_x}, ${_y})..."
        "${WS}/scripts/spawn_box.sh" "${_x}" "${_y}" "obstacle_pre_${_idx}_$$" \
            > "${LOG}/obstacle_${_idx}.log" 2>&1 || true
    done
fi

# 2. Perception + 3. Validation
# Skipped in dl mode because the DL segmentation node publishes target_x
# directly on /planning/validated_target_x; running both would conflict.
if [ "$LANE_KEEPER" != "dl" ] && [ "$LANE_KEEPER" != "yolo" ] && [ "$LANE_KEEPER" != "rfdetr" ]; then
    echo "[2/6] starting perception_node..."
    nohup ros2 run qcar2_autonomy perception_node \
        --ros-args \
        -p use_sim_time:=true \
        -p image_topic:=/qcar2/front_camera/image \
        > "${LOG}/perception.log" 2>&1 &
    disown

    echo "[3/6] starting validation_node..."
    nohup ros2 run qcar2_autonomy validation_node \
        --ros-args \
        -p use_sim_time:=true \
        > "${LOG}/validation.log" 2>&1 &
    disown
else
    echo "[2/6] skipping perception_node (${LANE_KEEPER} mode owns target_x)"
    echo "[3/6] skipping validation_node (${LANE_KEEPER} mode owns target_x)"
fi

# 4. Lane keeper. Mode was selected at the top of the script.
if [ "$LANE_KEEPER" = "ai" ]; then
    echo "[4/6] starting ai_lane_keeper_node (HSHL ResNet18)..."
    nohup ros2 run qcar2_autonomy ai_lane_keeper_node \
        --ros-args \
        -p use_sim_time:=true \
        -p image_topic:=/qcar2/front_camera/image \
        -p cmd_vel_topic:=/model/qcar2/cmd_vel \
        -p model_path:=${WS}/src/qcar2_autonomous_lanes/qcar2_autonomy/weights/resnet18_road_following.pth \
        -p architecture:=resnet18 \
        -p output_mode:=xy \
        -p device:=cpu \
        -p base_speed:=0.25 \
        -p max_angular:=1.0 \
        -p steering_gain:=1.0 \
        -p steering_sign:=-1.0 \
        -p lane_offset:=${LANE_OFFSET:-0.0} \
        > "${LOG}/ai.log" 2>&1 &
    disown
elif [ "$LANE_KEEPER" = "rfdetr" ]; then
    echo "[4/6] starting RF-DETR ONNX (GPU) + PID stack..."
    # onnxruntime-gpu needs CUDA 12 runtime libs on LD_LIBRARY_PATH. We installed
    # them via pip under nvidia/*/lib; collect them and prepend.
    NV_LIBS=$(find /usr/local/lib/python3.12/dist-packages/nvidia -maxdepth 3 -type d -name lib 2>/dev/null | tr '\n' ':')
    export LD_LIBRARY_PATH="${NV_LIBS}${LD_LIBRARY_PATH}"
    nohup ros2 run qcar2_autonomy rfdetr_onnx_lane_node \
        --ros-args \
        -p use_sim_time:=true \
        -p image_topic:=/qcar2/front_camera/image \
        -p target_topic:=/planning/validated_target_x \
        -p onnx_path:=${WS}/weights/car_track_v3_lane.onnx \
        -p classes_path:=${WS}/weights/car_track_v3_lane.classes.txt \
        -p lane_class_name:=lane2 \
        -p provider:=${RFDETR_PROVIDER:-CUDAExecutionProvider} \
        -p confidence_threshold:=${RFDETR_CONF:-0.5} \
        -p target_band_y_ratio:=${RFDETR_BAND:-0.70} \
        -p target_offset_px:=${TARGET_OFFSET_PX:-0.0} \
        > "${LOG}/rfdetr.log" 2>&1 &
    disown
    # Vision (RF-DETR) only fires when pointed straight down the corridor, so it
    # CANNOT steer the avoidance: the moment the car yaws or the obstacle fills
    # the frame, preds->0. The PID therefore lane-keeps via vision only in
    # LANE_KEEP (gate_on_state=true) and publishes to the mux input. During
    # LANE_CHANGE/LANE_RETURN the open-loop S-curve planner owns the wheel.
    nohup ros2 run qcar2_autonomy pid_lane_follower_node \
        --ros-args \
        -p use_sim_time:=true \
        -p cmd_vel_topic:=/qcar2/control/raw_cmd_vel \
        -p base_speed:=${PID_SPEED:-0.20} \
        -p kp:=${PID_KP:-1.4} \
        -p ki:=0.0 \
        -p kd:=0.0 \
        -p max_angular:=${PID_MAX_W:-1.5} \
        -p target_timeout_sec:=1.0 \
        -p lost_creep_speed:=${LOST_CREEP:-0.10} \
        > "${LOG}/pid.log" 2>&1 &
    disown
    # Open-loop bang-zero-bang S-curve: turn out, drive straight (gains lateral),
    # counter-turn to straighten. Bounded yaw, returns to ~original heading, and
    # needs no vision -- robust through the yaw where RF-DETR drops out.
    nohup ros2 run qcar2_autonomy lane_change_planner_node \
        --ros-args \
        -p use_sim_time:=true \
        -p cmd_topic:=/qcar2/control/maneuver_cmd_vel \
        -p cruise_speed:=${MANEUVER_SPEED:-0.30} \
        -p step_angular:=${MANEUVER_OMEGA:-1.0} \
        -p turn_duration:=${MANEUVER_TURN_T:-0.6} \
        -p straight_duration:=${MANEUVER_STRAIGHT_T:-2.2} \
        -p max_angular:=1.5 \
        > "${LOG}/lane_change_planner.log" 2>&1 &
    disown
    # Mux routes by FSM state: planner owns all of CHANGE/RETURN (S-curve + the
    # straight pass-creep), PID owns LANE_KEEP. Single author on the output.
    nohup ros2 run qcar2_autonomy cmd_vel_mux_node \
        --ros-args \
        -p use_sim_time:=true \
        -p lane_keep_topic:=/qcar2/control/raw_cmd_vel \
        -p maneuver_topic:=/qcar2/control/maneuver_cmd_vel \
        -p output_cmd_topic:=/model/qcar2/cmd_vel \
        -p route_by_state:=true \
        > "${LOG}/cmd_mux.log" 2>&1 &
    disown
elif [ "$LANE_KEEPER" = "yolo" ]; then
    echo "[4/6] starting YOLO seg + PID stack (best.pt drives the car)..."
    nohup ros2 run qcar2_autonomy yolo_lane_node \
        --ros-args \
        -p use_sim_time:=true \
        -p image_topic:=/qcar2/front_camera/image \
        -p target_topic:=/planning/validated_target_x \
        -p publish_target:=true \
        -p model_path:=${WS}/src/qcar2_autonomous_lanes/qcar2_autonomy/weights/best.pt \
        -p lane_class_name:=lane2 \
        -p traffic_light_class_name:=traffic_light \
        -p conf_threshold:=${YOLO_CONF:-0.25} \
        -p imgsz:=${YOLO_IMGSZ:-640} \
        -p device:=${YOLO_DEVICE:-cpu} \
        -p lane_width_px:=${LANE_WIDTH_PX:-420.0} \
        > "${LOG}/yolo.log" 2>&1 &
    disown
    nohup ros2 run qcar2_autonomy pid_lane_follower_node \
        --ros-args \
        -p use_sim_time:=true \
        -p cmd_vel_topic:=/model/qcar2/cmd_vel \
        -p base_speed:=0.20 \
        -p kp:=0.8 \
        -p kd:=0.05 \
        -p target_timeout_sec:=2.0 \
        > "${LOG}/pid.log" 2>&1 &
    disown
elif [ "$LANE_KEEPER" = "dl" ]; then
    echo "[4/6] starting DL segmentation + PID stack (YOLO best.pt drives the car)..."
    # 4a. DL segmentation publishes target_x on /planning/validated_target_x
    nohup ros2 run qcar2_autonomy dl_lane_segmentation_node \
        --ros-args \
        -p use_sim_time:=true \
        -p image_topic:=/qcar2/front_camera/image \
        -p target_topic:=/planning/validated_target_x \
        -p publish_target:=true \
        -p model_path:=${WS}/src/qcar2_autonomous_lanes/qcar2_autonomy/weights/best.pt \
        -p lane_class_name:=lane2 \
        -p traffic_light_class_name:=traffic_light \
        -p confidence_threshold:=${YOLO_CONF:-0.25} \
        -p imgsz:=${YOLO_IMGSZ:-640} \
        -p device:=${YOLO_DEVICE:-cpu} \
        -p lane_width_px:=${LANE_WIDTH_PX:-420.0} \
        -p target_offset_px:=${TARGET_OFFSET_PX:-0.0} \
        > "${LOG}/dl_seg.log" 2>&1 &
    disown
    # 4b. PID consumes target_x. Timeout is a little relaxed so occasional
    # slower CPU inference frames do not brake the car unnecessarily.
    nohup ros2 run qcar2_autonomy pid_lane_follower_node \
        --ros-args \
        -p use_sim_time:=true \
        -p cmd_vel_topic:=/model/qcar2/cmd_vel \
        -p base_speed:=0.20 \
        -p kp:=0.8 \
        -p kd:=0.05 \
        -p target_timeout_sec:=2.0 \
        > "${LOG}/pid.log" 2>&1 &
    disown
elif [ "$LANE_KEEPER" = "bev" ]; then
    echo "[4/6] starting classical 3-lane control stack (bev_lane_detector + lane_controller)..."
    # 4a. Bridge: std_msgs/String state -> qcar2_msgs/BehaviorState
    nohup ros2 run qcar2_autonomy state_to_behavior_bridge_node \
        --ros-args \
        -p use_sim_time:=true \
        -p desired_speed:=0.25 \
        > "${LOG}/state_bridge.log" 2>&1 &
    disown
    # 4b. lane_controller: consumes LaneModel + BehaviorState, outputs Twist
    nohup ros2 run qcar2_autonomy lane_controller \
        --ros-args \
        -p use_sim_time:=true \
        -p cmd_topic:=/qcar2/control/raw_cmd_vel \
        -p base_speed:=0.25 \
        -p kp:=1.15 \
        -p max_angular:=0.8 \
        > "${LOG}/lane_controller.log" 2>&1 &
    disown
    # 4c. Mux: only forward lane_controller's Twist during LANE_KEEP.
    # bev mode has no maneuver author, so the maneuver topic stays silent and
    # the DWA below still publishes /model/qcar2/cmd_vel directly during
    # LANE_CHANGE / LANE_RETURN.
    nohup ros2 run qcar2_autonomy cmd_vel_mux_node \
        --ros-args \
        -p use_sim_time:=true \
        -p lane_keep_topic:=/qcar2/control/raw_cmd_vel \
        -p maneuver_topic:=/qcar2/control/maneuver_cmd_vel \
        -p output_cmd_topic:=/model/qcar2/cmd_vel \
        > "${LOG}/cmd_mux.log" 2>&1 &
    disown
else
    echo "[4/6] starting pid_lane_follower_node (fallback)..."
    nohup ros2 run qcar2_autonomy pid_lane_follower_node \
        --ros-args \
        -p use_sim_time:=true \
        -p cmd_vel_topic:=/model/qcar2/cmd_vel \
        -p base_speed:=0.3 \
        -p kp:=0.8 \
        -p kd:=0.05 \
        > "${LOG}/pid.log" 2>&1 &
    disown
fi

# 5. State machine (LiDAR obstacle detection -> /system/current_state)
# rfdetr does an in-controller lateral-offset lane change (no open-loop planner),
# so recover on elapsed time and do NOT gate on world-frame odom-y, which is
# unreliable on a curving track. Other modes keep the odom-based DWA recovery.
if [ "$LANE_KEEPER" = "rfdetr" ]; then
    SM_EXTRA_ARGS="-p lidar_tracked_avoid:=true -p oneway_lane_change:=${ONEWAY_LANE_CHANGE:-true} -p track_range_m:=${TRACK_RANGE:-0.55} -p pass_confirm_sec:=${PASS_CONFIRM:-0.5} -p lane_change_min_sec:=${CHANGE_MIN_S:-1.8} -p return_duration_sec:=${RETURN_DUR:-3.5} -p pass_clearance_m:=${PASS_CLEARANCE:-0.8} -p side_sector_min_deg:=${SIDE_MIN_DEG:-30.0} -p side_sector_max_deg:=${SIDE_MAX_DEG:-150.0} -p lane_change_cooldown_sec:=${LANE_CHANGE_COOLDOWN:-3.0} -p prefer_side:=${FORCE_SIDE:-auto} -p vehicle_half_width_m:=${VEHICLE_HALF_WIDTH:-0.18} -p clearance_margin_m:=${CLEAR_MARGIN:-0.20} -p min_obstacle_width_m:=${MIN_OBSTACLE_WIDTH:-0.25} -p max_straight_dur:=${MAX_STRAIGHT_DUR:-4.5}"
else
    SM_EXTRA_ARGS=""
fi
echo "[5/6] starting state_machine_node..."
nohup ros2 run qcar2_autonomy state_machine_node \
    --ros-args \
    -p use_sim_time:=true \
    -p scan_topic:=/qcar2/lidar/scan \
    -p odom_topic:=/model/qcar2/odometry \
    -p min_lane_change_y:=0.30 \
    -p return_yaw_tolerance_rad:=0.12 \
    -p return_angular_tolerance:=0.10 \
    -p return_settle_sec:=0.75 \
    ${SM_EXTRA_ARGS} \
    > "${LOG}/state_machine.log" 2>&1 &
disown

# 6. DWA local planner (active during LANE_CHANGE and LANE_RETURN).
# Skipped in rfdetr mode -- the quintic lane_change_planner_node owns the
# maneuver topic there, and two authors on the same downstream Twist would race.
if [ "$LANE_KEEPER" != "rfdetr" ]; then
    echo "[6/6] starting dwa_local_planner_node..."
    nohup ros2 run qcar2_autonomy dwa_local_planner_node \
        --ros-args \
        -p use_sim_time:=true \
        -p scan_topic:=/qcar2/lidar/scan \
        -p cmd_vel_topic:=/model/qcar2/cmd_vel \
        -p odom_topic:=/model/qcar2/odometry \
        -p v_min:=0.20 \
        -p v_max:=0.30 \
        -p w_max:=1.2 \
        -p horizon_sec:=2.5 \
        -p collision_clearance:=0.38 \
        -p target_lane_y:=0.60 \
        -p heading_gate:=0.10 \
        -p w_lateral:=6.0 \
        -p w_heading:=2.0 \
        > "${LOG}/dwa.log" 2>&1 &
    disown
else
    echo "[6/6] skipping dwa_local_planner_node (rfdetr uses quintic planner)"
fi

# (optional) 3-lane segmentation debug overlay on /qcar2/lane/debug_image.
# Independent of the control pipeline - it just visualizes which white pixels
# belong to which line (left=red, middle=yellow, right=green) plus the raw
# histogram peaks. Enable with SHOW_3LANE=1.
if [ "${SHOW_3LANE}" = "1" ]; then
    echo "[+] starting bev_lane_detector_node (3-lane debug view)..."
    nohup ros2 run qcar2_autonomy bev_lane_detector_node \
        --ros-args \
        -p use_sim_time:=true \
        -p image_topic:=/qcar2/front_camera/image \
        > "${LOG}/bev_3lane.log" 2>&1 &
    disown
fi

# (optional) DL lane segmentation overlay using YOLO best.pt segmentation weights.
# Publishes a colored per-pixel overlay on /lane_segmentation/debug_image.
# View with: ros2 run rqt_image_view rqt_image_view /lane_segmentation/debug_image
# Disable with SHOW_DL_SEG=0.
if [ "${SHOW_DL_SEG:-1}" = "1" ] && [ "$LANE_KEEPER" != "dl" ] && [ "$LANE_KEEPER" != "yolo" ]; then
    echo "[+] starting dl_lane_segmentation_node (YOLO segmentation)..."
    nohup ros2 run qcar2_autonomy dl_lane_segmentation_node \
        --ros-args \
        -p use_sim_time:=true \
        -p image_topic:=/qcar2/front_camera/image \
        -p publish_target:=false \
        -p model_path:=${WS}/src/qcar2_autonomous_lanes/qcar2_autonomy/weights/best.pt \
        -p lane_class_name:=lane2 \
        -p traffic_light_class_name:=traffic_light \
        -p confidence_threshold:=${YOLO_CONF:-0.25} \
        -p imgsz:=${YOLO_IMGSZ:-640} \
        -p device:=${YOLO_DEVICE:-cpu} \
        -p lane_width_px:=${LANE_WIDTH_PX:-420.0} \
        -p target_offset_px:=${TARGET_OFFSET_PX:-0.0} \
        > "${LOG}/dl_seg.log" 2>&1 &
    disown
fi

sleep 3
echo
echo "Pipeline up. Logs: ${LOG}/{sim,perception,validation,pid}.log"
echo
echo "Watch live metrics:"
echo "    ros2 topic echo /perception/raw_lane_data"
echo "    ros2 topic echo /planning/validated_target_x"
echo "    ros2 topic echo /model/qcar2/cmd_vel"
echo "    ros2 topic echo /model/qcar2/odometry"
echo
echo "Tear down: scripts/stop_lane_keeping.sh"
