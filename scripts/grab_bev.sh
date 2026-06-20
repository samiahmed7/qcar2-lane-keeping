#!/usr/bin/env bash
# Boot the sim headless, run the BEV lane detector, and grab the raw camera +
# the BEV debug overlay to PNGs for inspection. Then tear down.
#   WORLD=university_track SX=0.0 SY=0.0 ./scripts/grab_bev.sh
cd "$(dirname "$0")/.." || exit 1
WS="$(pwd)"
source /opt/ros/jazzy/setup.bash
source "$WS/install/setup.bash"
LOG=/tmp/grabbev; mkdir -p "$LOG"
WORLD="${WORLD:-university_track}"; SX="${SX:-0.0}"; SY="${SY:-0.0}"; SYAW="${SYAW:-0.0}"

echo "[1/5] teardown"
./scripts/stop_lane_keeping.sh >/dev/null 2>&1
pkill -9 -f 'gz sim' >/dev/null 2>&1; pkill -9 -f bev_lane_detector >/dev/null 2>&1
sleep 3

echo "[2/5] launch sim ($WORLD @ $SX,$SY)"
nohup ros2 launch qcar2_bringup sim_bringup.launch.py \
  world:=$WORLD x:=$SX y:=$SY yaw:=$SYAW headless:=true >"$LOG/sim.log" 2>&1 &

echo "[3/5] wait for camera"
ok=0
for i in $(seq 1 45); do
  ros2 topic list 2>/dev/null | grep -q '/qcar2/front_camera/image' && { ok=1; break; }
  sleep 2
done
[ "$ok" != 1 ] && { echo "FAIL: no camera topic"; tail -20 "$LOG/sim.log"; exit 1; }
sleep 5

echo "[4/5] run BEV detector + grab frames"
nohup ros2 run qcar2_autonomy bev_lane_detector_node --ros-args \
  -p use_sim_time:=true -p image_topic:=/qcar2/front_camera/image \
  -p debug_topic:=/qcar2/lane/debug_image -p use_ipm:=true \
  >"$LOG/bev.log" 2>&1 &
sleep 8
python3 scripts/grab_image.py /qcar2/front_camera/image "$WS/bev_camera_raw.png" 30
python3 scripts/grab_image.py /qcar2/lane/debug_image  "$WS/bev_debug.png" 30
echo "--- a LaneModel sample ---"
timeout 8 ros2 topic echo /qcar2/lane/model --once 2>/dev/null | head -25 || echo "(no LaneModel)"

echo "[5/5] teardown + logs"
./scripts/stop_lane_keeping.sh >/dev/null 2>&1
pkill -9 -f 'gz sim' >/dev/null 2>&1; pkill -9 -f bev_lane_detector >/dev/null 2>&1
echo "--- bev.log tail ---"; tail -8 "$LOG/bev.log"
echo "GRAB DONE"
