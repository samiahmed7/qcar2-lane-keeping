#!/usr/bin/env bash
# Spawn a tall red test box in the Gazebo world for LiDAR / safety testing.
#
# Usage:
#   ./spawn_box.sh                       # default: x=0.75 y=-6.20 (lane 0, ahead of car start)
#   ./spawn_box.sh <x> <y>               # custom position, world frame
#   ./spawn_box.sh <x> <y> <name>        # custom name (default: test_box_<timestamp>)
#
# The default car spawn pose (simulation.launch.py) is (x=0, y=-6.20, yaw=0),
# so +x is forward and +y is left in the world frame at start.
#
# Useful test positions on qcar_oval:
#   0.75 -6.20    classic obstacle in lane 0, ahead of start (triggers auto LC)
#   2.00 -6.20    farther obstacle in lane 0 (more reaction time)
#   2.00 -5.51    obstacle ALREADY in lane 1 (~0.69 m left) — should stop after LC
#   2.00 -5.85    obstacle straddling the lane-1 path edge — tests path band
#   3.00 -6.20    farther still — exercises slow-zone braking from longer range
#
# Box is 0.25 x 0.25 m wide, 0.35 m tall (tall enough to exceed the 0.19 m
# LiDAR scan height). Static, red, with collision.

set -euo pipefail

X="${1:-0.75}"
Y="${2:--6.20}"
NAME="${3:-test_box_$(date +%s)}"

SDF="<sdf version='1.10'>
  <model name='${NAME}'>
    <static>true</static>
    <link name='link'>
      <pose>0 0 0.175 0 0 0</pose>
      <collision name='collision'>
        <geometry><box><size>0.25 0.25 0.35</size></box></geometry>
      </collision>
      <visual name='visual'>
        <geometry><box><size>0.25 0.25 0.35</size></box></geometry>
        <material>
          <ambient>1.0 0.1 0.1 1.0</ambient>
          <diffuse>1.0 0.1 0.1 1.0</diffuse>
        </material>
      </visual>
    </link>
  </model>
</sdf>"

echo "Spawning '${NAME}' at x=${X} y=${Y}"
ros2 run ros_gz_sim create \
  -name "${NAME}" \
  -allow_renaming true \
  -x "${X}" -y "${Y}" -z 0.0 \
  -string "${SDF}"
