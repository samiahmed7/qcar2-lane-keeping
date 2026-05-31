# QCar2 Autonomous Lanes

ROS 2 workspace project for running a QCar2 in a Gazebo lab track with a
modular lane-perception, lane-keeping, obstacle, behavior, and safety stack.

## Packages

- `qcar2_description`: QCar2 URDF, meshes, and RViz assets.
- `qcar2_worlds`: Gazebo worlds, models, maps, and world launch files.
- `qcar2_bringup`: simulation, real-robot, and combined autonomy bringup.
- `qcar2_autonomy`: Python autonomy nodes and autonomy launch files.
- `qcar2_msgs`: custom ROS messages shared by the autonomy nodes.

## Build

```bash
cd ~/rosbot_ws
colcon build --symlink-install --packages-up-to qcar2_autonomy qcar2_bringup
source install/setup.bash
```

## Run

Simulation only, headless:

```bash
ros2 launch qcar2_bringup sim_bringup.launch.py headless:=true
```

Spawn the obstacle:

```bash
./scripts/spawn_box.sh 2.0 -6.20
```

Start autonomy:

```bash
./scripts/run_autonomy.sh
```

Open the overlay viewer:

```bash
python3 scripts/view_overlay.py
```

For the exact four-terminal demo flow, see the root `README.md`.

Phase 1 lane perception only:

```bash
ros2 launch qcar2_autonomy lane_perception.launch.py
```

Useful topics:

- `/qcar2/front_camera/image`
- `/qcar2/lane/model`
- `/qcar2/lane/debug_image`
- `/qcar2/lidar/scan`
- `/qcar2/obstacle/state`
- `/qcar2/behavior/state`
- `/model/qcar2/cmd_vel`

## Phase 1 Perception

`bev_lane_detector_node` subscribes to `/qcar2/front_camera/image`, thresholds
white lane markings in a lower road ROI, detects lane-line x peaks, and
publishes a `qcar2_msgs/msg/LaneModel` on `/qcar2/lane/model`. The target is
the right lane center, computed only when the middle dashed and right solid
lines are visible. It also publishes a debug image on
`/qcar2/lane/debug_image`.
