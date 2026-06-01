#!/usr/bin/env bash
# Drive the QCar2 with the separated MPC stack:
# LiDAR obstacle detection -> reference planner -> pure MPC tracker.
# REPLACES run_autonomy.sh (do not run both).
# Prereq: sim is up (Terminal 1) and waypoints recorded (scripts/record_path.sh).
#
#   OVERLAY=1  -> also run RF-DETR for the debug overlay (view_overlay.py)
#   TARGET_SPEED=0.3  LOOP=true  WAYPOINTS=...npy  MPC_SIDE=left
set -e
cd "$(dirname "$0")/.."
WS="$(pwd)"
source /opt/ros/jazzy/setup.bash
source "${WS}/install/setup.bash"
LOG=/tmp/qcar2_pipeline; mkdir -p "${LOG}"

WP="${WAYPOINTS:-${WS}/track_waypoints.npy}"
if [ ! -f "${WP}" ]; then
    echo "No waypoints at ${WP}. Record them first: scripts/record_path.sh"
    exit 1
fi

echo "Waiting for the sim (/qcar2/lidar/scan and /model/qcar2/odometry)..."
while true; do
    TOPICS="$(timeout 10 ros2 topic list --no-daemon --spin-time 5 2>/dev/null || true)"
    if grep -qx '/qcar2/lidar/scan' <<< "${TOPICS}" \
        && grep -qx '/model/qcar2/odometry' <<< "${TOPICS}"; then
        break
    fi
    echo "  still waiting for Gazebo bridge topics..."
    sleep 2
done

if [ "${OVERLAY:-1}" = "1" ]; then
    echo "Starting RF-DETR overlay (debug view only, does not drive)..."
    NV_LIBS=$(find /usr/local/lib/python3.12/dist-packages/nvidia -maxdepth 3 -type d -name lib 2>/dev/null | tr '\n' ':')
    export LD_LIBRARY_PATH="${NV_LIBS}${LD_LIBRARY_PATH}"
    nohup ros2 run qcar2_autonomy rfdetr_onnx_lane_node --ros-args \
        -p use_sim_time:=true \
        -p image_topic:=/qcar2/front_camera/image \
        -p onnx_path:=${WS}/weights/car_track_v3_lane.onnx \
        -p classes_path:=${WS}/weights/car_track_v3_lane.classes.txt \
        -p lane_class_name:=lane2 -p publish_target:=false \
        > "${LOG}/rfdetr.log" 2>&1 & disown
fi

# normalize RECORD (accept 1/true/yes) to a launch bool
case "${RECORD:-}" in 1|true|yes|on) REC=true ;; *) REC=false ;; esac
echo "Starting separated MPC stack (waypoints=${WP}). Diagnostics -> ${LOG}/mpc.log  (record_log=${REC})"
ros2 launch qcar2_autonomy mpc.launch.py \
    waypoints_path:=${WP} \
    target_speed:=${TARGET_SPEED:-0.30} \
    prefer_side:=${MPC_SIDE:-left} \
    loop:=${LOOP:-true} \
    record_log:=${REC} \
    2>&1 | tee "${LOG}/mpc.log"
# RECORD=1 logs the run to ~/rosbot_ws/mpc_run_log.npz; then:
#   python3 scripts/plot_mpc_run.py   -> /tmp/mpc_run_plot.png
