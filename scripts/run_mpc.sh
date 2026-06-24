#!/usr/bin/env bash
# Drive the QCar2 with the separated MPC stack:
# LiDAR obstacle detection -> reference planner -> pure MPC tracker.
# REPLACES run_autonomy.sh (do not run both).
# Prereq: sim is up (Terminal 1) and waypoints recorded (scripts/record_path.sh).
#
#   OVERLAY=1       -> also run the HSV+IPM BEV debug overlay
#   LANE_FUSION=1   -> use HSV+BEV lane centering as the MPC reference bias
#   SIM_NOISE=1     -> route MPC through noisy odometry/LiDAR topics
#   START_LANE=right -> use the right-lane route and spawn-relative waypoints
#   TARGET_SPEED=0.55  LOOP=true  WAYPOINTS=...npy  MPC_SIDE=left
set -e
cd "$(dirname "$0")/.."
WS="$(pwd)"
source /opt/ros/jazzy/setup.bash
source "${WS}/install/setup.bash"
LOG=/tmp/qcar2_pipeline; mkdir -p "${LOG}"

truthy() {
    case "${1:-}" in
        1|true|TRUE|yes|YES|on|ON) return 0 ;;
        *) return 1 ;;
    esac
}

START_LANE="${START_LANE:-center}"

if [ "${START_LANE}" = "right" ]; then
    WP="${WAYPOINTS:-${WS}/university_route_rlane.npy}"
    SPAWN_X="${SPAWN_X:-0.0}"
    SPAWN_Y="${SPAWN_Y:--0.25}"
    SPAWN_YAW="${SPAWN_YAW:-0.0}"
else
    WP="${WAYPOINTS:-${WS}/track_waypoints.npy}"
    SPAWN_X="${SPAWN_X:-0.0}"
    SPAWN_Y="${SPAWN_Y:-0.0}"
    SPAWN_YAW="${SPAWN_YAW:-0.0}"
fi
if [ ! -f "${WP}" ]; then
    echo "No waypoints at ${WP}. Record them first: scripts/record_path.sh"
    exit 1
fi

WP_RUN="${WP}"
if [ "${START_LANE}" = "right" ] || truthy "${LOCALIZE_WAYPOINTS:-}"; then
    WP_RUN="${LOG}/local_waypoints.npy"
    python3 - "${WP}" "${WP_RUN}" "${SPAWN_X}" "${SPAWN_Y}" "${SPAWN_YAW}" <<'PY'
import math
import sys

import numpy as np

src, dst, sx, sy, syaw = sys.argv[1:6]
sx, sy, syaw = float(sx), float(sy), float(syaw)
route = np.load(src)
dist = np.hypot(route[0] - sx, route[1] - sy)
start_idx = int(np.argmin(dist))
segment = route[:, start_idx:]
dx = segment[0] - sx
dy = segment[1] - sy
c = math.cos(-syaw)
s = math.sin(-syaw)
local = np.vstack((c * dx - s * dy, s * dx + c * dy))
if local.shape[1] == 0 or np.hypot(local[0, 0], local[1, 0]) > 1e-3:
    local = np.column_stack((np.zeros(2), local))
np.save(dst, local)
print(
    f"Localized waypoints: source={src}, spawn=({sx:.2f},{sy:.2f},{syaw:.2f}), "
    f"start_idx={start_idx}, first_local=({local[0,0]:.3f},{local[1,0]:.3f}) -> {dst}"
)
PY
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

if truthy "${LANE_FUSION:-}"; then LF=true; else LF=false; fi
if truthy "${SIM_NOISE:-${SENSOR_NOISE:-}}"; then
    NOISE=true
    ODOM_TOPIC="${ODOM_TOPIC:-/mpc/noisy_odometry}"
    SCAN_TOPIC="${SCAN_TOPIC:-/mpc/noisy_scan}"
else
    NOISE=false
    ODOM_TOPIC="${ODOM_TOPIC:-/model/qcar2/odometry}"
    SCAN_TOPIC="${SCAN_TOPIC:-/qcar2/lidar/scan}"
fi

if truthy "${OVERLAY:-1}" || [ "${LF}" = "true" ]; then
    echo "Starting Reliable BEV lane node (debug + MPC lane-fusion model)..."
    nohup ros2 run qcar2_autonomy reliable_lane_detector_node --ros-args \
        --params-file "${WS}/src/qcar2_autonomous_lanes/qcar2_autonomy/config/mpc_nodes.yaml" \
        -p use_sim_time:=true \
        -p image_topic:=/qcar2/front_camera/image \
        -p lane_model_topic:=${LANE_MODEL_TOPIC:-/qcar2/lane/model} \
        -p debug_topic:=/qcar2/lane/debug_image \
        -p use_ipm:=true \
        > "${LOG}/bev_lane.log" 2>&1 & disown
fi

# normalize RECORD (accept 1/true/yes) to a launch bool
if truthy "${RECORD:-}"; then REC=true; else REC=false; fi
echo "Starting separated MPC stack (waypoints=${WP_RUN}). Diagnostics -> ${LOG}/mpc.log  (record_log=${REC}, lane_fusion=${LF}, lane_source=bev, noise=${NOISE}, start_lane=${START_LANE})"
ros2 launch qcar2_autonomy mpc.launch.py \
    waypoints_path:=${WP_RUN} \
    target_speed:=${TARGET_SPEED:-0.55} \
    odom_topic:=${ODOM_TOPIC} \
    scan_topic:=${SCAN_TOPIC} \
    prefer_side:=${MPC_SIDE:-left} \
    loop:=${LOOP:-true} \
    record_log:=${REC} \
    sensor_noise_enabled:=${NOISE} \
    max_straight_speed:=${MAX_STRAIGHT_SPEED:-0.80} \
    min_curve_speed:=${MIN_CURVE_SPEED:-0.18} \
    max_lateral_accel_mps2:=${MAX_LATERAL_ACCEL:-0.45} \
    curvature_lookahead_m:=${CURVATURE_LOOKAHEAD:-1.20} \
    lane_fusion_enabled:=${LF} \
    lane_model_topic:=${LANE_MODEL_TOPIC:-/qcar2/lane/model} \
    lane_target_topic:=${LANE_TARGET_TOPIC:-/planning/validated_target_x} \
    lane_fusion_max_correction_m:=${LANE_FUSION_MAX_CORRECTION:-0.45} \
    lane_fusion_max_step_m:=${LANE_FUSION_MAX_STEP:-0.012} \
    lane_fusion_deadband_px:=${LANE_FUSION_DEADBAND_PX:-8.0} \
    lane_px_to_m:=${LANE_PX_TO_M:-0.0015} \
    lane_fusion_gain:=${LANE_FUSION_GAIN:-0.70} \
    lane_fusion_alpha:=${LANE_FUSION_ALPHA:-0.08} \
    lane_fusion_min_confidence:=${LANE_FUSION_MIN_CONFIDENCE:-0.18} \
    lane_fusion_hold_low_confidence:=${LANE_FUSION_HOLD_LOW_CONFIDENCE:-true} \
    lane_fusion_disable_heading_delta_rad:=${LANE_FUSION_HEADING_GATE:-3.20} \
    command_smoothing_enabled:=${COMMAND_SMOOTHING:-true} \
    omega_smoothing_alpha:=${OMEGA_SMOOTHING_ALPHA:-0.80} \
    max_omega_rate_rps2:=${MAX_OMEGA_RATE:-5.0} \
    2>&1 | tee "${LOG}/mpc.log"
# RECORD=1 logs the run to ~/rosbot_ws/mpc_run_log.npz; then:
#   python3 scripts/plot_mpc_run.py   -> /tmp/mpc_run_plot.png
