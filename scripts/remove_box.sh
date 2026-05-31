#!/usr/bin/env bash
# Remove a previously-spawned test obstacle from the Gazebo world.
#
# Usage:
#   ./scripts/remove_box.sh                # default: removes 'obstacle_box'
#   ./scripts/remove_box.sh <name>         # removes the named model
#   ./scripts/remove_box.sh --all          # removes every test_box_* and obstacle_box

set -euo pipefail

source /opt/ros/jazzy/setup.bash

remove_one() {
  local name="$1"
  echo "Removing model '${name}'..."
  gz service \
    -s /world/lab_track/remove \
    --reqtype gz.msgs.Entity --reptype gz.msgs.Boolean \
    --timeout 2000 \
    --req "name: \"${name}\" type: MODEL" || true
}

if [[ "${1:-}" == "--all" ]]; then
  for name in $(gz model --list 2>/dev/null | awk '/^- (obstacle_box|test_box_)/ {sub(/^- /,""); print}'); do
    remove_one "$name"
  done
  exit 0
fi

remove_one "${1:-obstacle_box}"
