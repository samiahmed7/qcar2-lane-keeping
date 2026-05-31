#!/usr/bin/env bash
# Run the QCar2 Gazebo map/world with the car spawned in it.
#
# Usage:
#   ./scripts/run_qcar2_map.sh
#   ./scripts/run_qcar2_map.sh sim
#   ./scripts/run_qcar2_map.sh autonomy
#   ./scripts/run_qcar2_map.sh sim lab_track 0.0 -6.20 0.07 0.0
#
# Modes:
#   sim      Gazebo world + QCar2 + ROS/Gazebo bridges
#   autonomy sim + QCar2 autonomous-lanes skeleton

set -euo pipefail

usage() {
  cat <<'EOF'
Run the QCar2 Gazebo map/world with the car spawned in it.

Usage:
  ./scripts/run_qcar2_map.sh
  ./scripts/run_qcar2_map.sh sim
  ./scripts/run_qcar2_map.sh autonomy
  ./scripts/run_qcar2_map.sh sim lab_track 0.0 -6.20 0.07 0.0

Modes:
  sim      Gazebo world + QCar2 + ROS/Gazebo bridges
  autonomy sim + QCar2 autonomous-lanes skeleton
EOF
}

MODE="sim"
if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ "${1:-}" == "sim" || "${1:-}" == "autonomy" ]]; then
  MODE="$1"
  shift
fi

WORLD="${1:-lab_track}"
X="${2:-0.0}"
if [[ $# -ge 3 ]]; then
  Y="$3"
else
  Y="-6.20"
fi
Z="${4:-0.07}"
YAW="${5:-0.0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ ! -f /opt/ros/jazzy/setup.bash ]]; then
  echo "ROS 2 Jazzy setup not found at /opt/ros/jazzy/setup.bash" >&2
  exit 1
fi

set +u
source /opt/ros/jazzy/setup.bash

if [[ -f "${WS_DIR}/install/setup.bash" ]]; then
  source "${WS_DIR}/install/setup.bash"
else
  set -u
  echo "Workspace install/setup.bash not found. Build first:" >&2
  echo "  cd ${WS_DIR} && colcon build --symlink-install" >&2
  exit 1
fi
set -u

mkdir -p "${HOME}/.ros/log"

GZ_VENDOR_LIB="/opt/ros/jazzy/opt/gz_sim_vendor/lib"
if [[ -d "$GZ_VENDOR_LIB" ]]; then
  export GZ_SIM_SYSTEM_PLUGIN_PATH="${GZ_VENDOR_LIB}${GZ_SIM_SYSTEM_PLUGIN_PATH:+:${GZ_SIM_SYSTEM_PLUGIN_PATH}}"
fi

QCAR2_WORLDS_PREFIX="$(ros2 pkg prefix qcar2_worlds)"
QCAR2_WORLDS_SHARE="${QCAR2_WORLDS_PREFIX}/share/qcar2_worlds"

WORLD_FILE="${QCAR2_WORLDS_SHARE}/worlds/${WORLD}.sdf"
if [[ ! -f "$WORLD_FILE" ]]; then
  echo "World '${WORLD}' not found. Available worlds:" >&2
  find -L "${QCAR2_WORLDS_SHARE}/worlds" -maxdepth 1 -type f -name "*.sdf" -printf "  %f\n" | sed 's/\.sdf$//' >&2
  exit 1
fi

case "$MODE" in
  sim)
    LAUNCH_FILE="sim_bringup.launch.py"
    ;;
  autonomy)
    LAUNCH_FILE="autonomy.launch.py"
    ;;
  *)
    echo "Unknown mode '${MODE}'. Use sim or autonomy." >&2
    exit 1
    ;;
esac

echo "Starting QCar2 ${MODE} in world '${WORLD}' at pose x=${X}, y=${Y}, z=${Z}, yaw=${YAW}"
exec ros2 launch qcar2_bringup "$LAUNCH_FILE" \
  world:="$WORLD" \
  x:="$X" \
  y:="$Y" \
  z:="$Z" \
  yaw:="$YAW"
