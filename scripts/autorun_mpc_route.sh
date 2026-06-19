#!/usr/bin/env bash
# One-shot autonomous run: MPC drives the baked university route (START -> up the
# branch -> right turn at the intersection -> around the loop), headless, records
# the run, tears everything down, and plots the driven path. Self-contained.
cd "$(dirname "$0")/.." || exit 1
WS="$(pwd)"
source /opt/ros/jazzy/setup.bash
source "$WS/install/setup.bash"
LOG=/tmp/autorun; mkdir -p "$LOG"
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
echo "[4/6] launching MPC stack (route, loop=false, speed=$SPEED)"
echo "      source waypoints: $WP"
echo "      using waypoints:  $WP_RUN"
nohup ros2 launch qcar2_autonomy mpc.launch.py \
  waypoints_path:="$WP_RUN" \
  target_speed:=$SPEED loop:=false \
  >"$LOG/mpc.log" 2>&1 &

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
ls -la "$WS"/mpc_run_plot.png /tmp/mpc_run_plot.png 2>/dev/null
echo "AUTORUN DONE"
