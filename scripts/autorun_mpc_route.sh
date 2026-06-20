#!/usr/bin/env bash
# One-shot autonomous run: MPC drives the baked university route (START -> up the
# branch -> right turn at the intersection -> around the loop), headless, records
# the run, tears everything down, and plots the driven path. Self-contained.
cd "$(dirname "$0")/.." || exit 1
WS="$(pwd)"
source /opt/ros/jazzy/setup.bash
source "$WS/install/setup.bash"
LOG=/tmp/autorun; mkdir -p "$LOG"
rm -f "$LOG"/sim.log "$LOG"/mpc.log "$LOG"/logger.log "$LOG"/plot.log \
  "$LOG"/fusion.log "$LOG"/bev.log "$LOG"/box.log "$LOG"/local_waypoints.npy
SPEED="${SPEED:-0.22}"
DUR="${DUR:-120}"
START_LANE="${START_LANE:-center}"
if [ "$START_LANE" = "right" ]; then
  WP="${WAYPOINTS:-$WS/university_route_rlane.npy}"
  SX="${SX:-0.0}"
  SY="${SY:--0.25}"
  SYAW="${SYAW:-0.0}"
else
  WP="${WAYPOINTS:-$WS/university_route.npy}"
  SX="${SX:-0.0}"
  SY="${SY:-0.0}"
  SYAW="${SYAW:-0.0}"
fi
LOCALIZE_WAYPOINTS="${LOCALIZE_WAYPOINTS:-auto}"

cleanup() {
  ./scripts/stop_lane_keeping.sh >/dev/null 2>&1
  pkill -9 -f 'gz sim'                     >/dev/null 2>&1
  pkill -9 -f mpc_drive_node              >/dev/null 2>&1
  pkill -9 -f mpc_reference_planner_node  >/dev/null 2>&1
  pkill -9 -f mpc_lidar_obstacle_node     >/dev/null 2>&1
  pkill -9 -f mpc_logger_node             >/dev/null 2>&1
  pkill -9 -f bev_lane_detector           >/dev/null 2>&1
}

echo "[1/6] tearing down any stale sim"
cleanup; sleep 3

echo "[2/6] launching sim headless (university_track @ ${SX},${SY},${SYAW}, start_lane=${START_LANE})"
nohup ros2 launch qcar2_bringup sim_bringup.launch.py \
  world:=university_track x:="$SX" y:="$SY" yaw:="$SYAW" headless:=true \
  >"$LOG/sim.log" 2>&1 &

echo "[3/6] waiting for /model/qcar2/odometry (up to 90s)"
ok=0
for i in $(seq 1 45); do
  if ros2 topic list 2>/dev/null | grep -q '/model/qcar2/odometry'; then ok=1; break; fi
  sleep 2
done
if [ "$ok" != 1 ]; then
  echo "FAIL: sim never published odometry"; echo "--- sim.log ---"; tail -40 "$LOG/sim.log"
  cleanup; exit 1
fi
echo "  odometry is up"; sleep 6

WP_RUN="$WP"
if [ "$LOCALIZE_WAYPOINTS" = "true" ] || {
  [ "$LOCALIZE_WAYPOINTS" = "auto" ] && \
  python3 - "$SX" "$SY" "$SYAW" <<'PY'
import math
import sys
x, y, yaw = (float(v) for v in sys.argv[1:4])
raise SystemExit(0 if abs(x) > 1e-9 or abs(y) > 1e-9 or abs(yaw) > 1e-9 else 1)
PY
}; then
  WP_RUN="$LOG/local_waypoints.npy"
  python3 - "$WP" "$WP_RUN" "$SX" "$SY" "$SYAW" <<'PY'
import math
import pathlib
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
length = float(np.sum(np.hypot(np.diff(local[0]), np.diff(local[1]))))
print(
    f"localized waypoints: start_idx={start_idx}, "
    f"start_world=({route[0,start_idx]:.2f},{route[1,start_idx]:.2f}), "
    f"points={local.shape[1]}, length={length:.2f} m -> {dst}"
)
PY
fi

if [ -n "${OBSTACLE:-}" ]; then
  echo "[obstacle] spawning box at ${OBSTACLE} (world frame)"
  ./scripts/spawn_box.sh ${OBSTACLE} >"$LOG/box.log" 2>&1 || echo "  spawn_box failed (see $LOG/box.log)"
  sleep 2
fi
FUSION_ARGS=""
case "${LANE_FUSION:-}" in
  1|true|yes|on|TRUE|YES|ON)
    echo "[fusion] BEV lane detector + lane fusion ON"
    nohup ros2 run qcar2_autonomy bev_lane_detector_node --ros-args \
      -p use_sim_time:=true -p image_topic:=/qcar2/front_camera/image \
      -p lane_model_topic:=/qcar2/lane/model -p debug_topic:=/qcar2/lane/debug_image \
      -p use_ipm:=true >"$LOG/bev.log" 2>&1 &
    sleep 4
    FUSION_ARGS="lane_fusion_enabled:=true lane_model_topic:=/qcar2/lane/model lane_fusion_max_correction_m:=${LANE_FUSION_MAX_CORRECTION:-0.06} lane_fusion_max_step_m:=${LANE_FUSION_MAX_STEP:-0.006} lane_fusion_gain:=${LANE_FUSION_GAIN:-0.35} lane_fusion_alpha:=${LANE_FUSION_ALPHA:-0.25} lane_fusion_min_confidence:=${LANE_FUSION_MIN_CONFIDENCE:-0.30} lane_fusion_hold_low_confidence:=${LANE_FUSION_HOLD_LOW_CONFIDENCE:-true} lane_fusion_disable_heading_delta_rad:=${LANE_FUSION_HEADING_GATE:-3.20}"
    ;;
esac

NOISE_ARGS=""
case "${SIM_NOISE:-${SENSOR_NOISE:-}}" in
  1|true|yes|on|TRUE|YES|ON)
    echo "[noise] sensor noise ON (LiDAR + gyro/odom): MPC reads /mpc/noisy_odometry + /mpc/noisy_scan"
    NOISE_ARGS="sensor_noise_enabled:=true odom_topic:=/mpc/noisy_odometry scan_topic:=/mpc/noisy_scan"
    ;;
esac

echo "[4/6] launching MPC stack (route, loop=false, speed=$SPEED, fusion=${LANE_FUSION:-0}, noise=${SIM_NOISE:-0})"
echo "      source waypoints: $WP"
echo "      using waypoints:  $WP_RUN"
nohup ros2 launch qcar2_autonomy mpc.launch.py \
  waypoints_path:="$WP_RUN" \
  target_speed:=$SPEED loop:=false \
  command_smoothing_enabled:=${COMMAND_SMOOTHING:-true} \
  omega_smoothing_alpha:=${OMEGA_SMOOTHING_ALPHA:-0.80} \
  max_omega_rate_rps2:=${MAX_OMEGA_RATE:-5.0} \
  $FUSION_ARGS \
  $NOISE_ARGS \
  >"$LOG/mpc.log" 2>&1 &

if [ -n "$FUSION_ARGS" ]; then
  ( for i in $(seq 1 8); do printf "fusion@%s: " "$i"
      timeout 4 ros2 topic echo /mpc/lane_fusion_status --field data --once 2>/dev/null || echo "(none)"
      sleep $((DUR / 8)); done ) >"$LOG/fusion.log" 2>&1 &
fi

echo "[5/6] recording ${DUR}s (logger auto-saves)"
timeout -s INT $((DUR + 15)) ros2 run qcar2_autonomy mpc_logger_node --ros-args \
  -p use_sim_time:=true -p duration_sec:=${DUR}.0 \
  -p out_path:="$WS/mpc_run_log.npz" >"$LOG/logger.log" 2>&1

echo "[6/6] stopping + plotting"
cleanup; sleep 2
python3 scripts/plot_mpc_run.py "$WS/mpc_run_log.npz" "$WP_RUN" \
  >"$LOG/plot.log" 2>&1
echo "--- logger.log tail ---"; tail -3 "$LOG/logger.log"
echo "--- plot.log tail ---"; tail -3 "$LOG/plot.log"
echo "--- modes / problems from mpc.log ---"
grep -iE "planner:|mode|infeasible|BLOCK|solve error" "$LOG/mpc.log" | tail -15
[ -f "$LOG/fusion.log" ] && { echo "--- lane fusion status ---"; cat "$LOG/fusion.log"; }
ls -la "$WS"/mpc_run_plot.png /tmp/mpc_run_plot.png 2>/dev/null
echo "AUTORUN DONE"
