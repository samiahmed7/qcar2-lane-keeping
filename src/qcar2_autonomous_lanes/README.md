# QCar2 Autonomous Lanes

Package group for the QCar2 BEV-MPC stack.

The active run path on the `bev-mpc` branch is:

```text
HSV/IPM BEV lane detector + LiDAR obstacle perception + MPC route tracking
```

The root `README.md` contains the exact terminal commands.

## Packages

- `qcar2_description`: QCar2 URDF, meshes, and sensors.
- `qcar2_worlds`: Gazebo worlds, including `lab_track` and `university_track`.
- `qcar2_bringup`: simulation bringup and ROS/Gazebo bridge.
- `qcar2_autonomy`: BEV lane detector, MPC planner, tracker, logger, and helpers.
- `qcar2_msgs`: shared custom messages such as `LaneModel`.

## Build

```bash
cd ~/rosbot_ws
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install --packages-up-to qcar2_autonomy qcar2_bringup
source install/setup.bash
```

## Run Short Form

```bash
# Terminal 1
ros2 launch qcar2_bringup sim_bringup.launch.py world:=university_track x:=0.0 y:=-0.25 yaw:=0.0 headless:=false

# Terminal 2
START_LANE=right LANE_FUSION=1 OVERLAY=1 TARGET_SPEED=0.55 LOOP=false ./scripts/run_mpc.sh

# Terminal 3
ros2 topic echo /mpc/lane_fusion_status
```

Headless record-and-plot:

```bash
START_LANE=right LANE_FUSION=1 SPEED=0.55 DUR=300 ./scripts/autorun_mpc_route.sh
```

## Core Nodes

| Node | Role |
|---|---|
| `bev_lane_detector_node` | HSV/IPM BEV lane extraction -> `/qcar2/lane/model`. |
| `mpc_lidar_obstacle_node` | LiDAR obstacle estimate -> `/mpc/obstacle`. |
| `mpc_reference_planner_node` | Route planner, obstacle behavior, BEV fusion. |
| `mpc_drive_node` | MPC tracker -> `/model/qcar2/cmd_vel`. |
| `mpc_logger_node` | Records `.npz` logs for `plot_mpc_run.py`. |

## Useful Topics

- `/qcar2/front_camera/image`
- `/qcar2/lane/model`
- `/qcar2/lane/debug_image`
- `/qcar2/lidar/scan`
- `/model/qcar2/odometry`
- `/mpc/reference_path`
- `/mpc/target_speed`
- `/mpc/mode`
- `/mpc/lane_fusion_status`
- `/model/qcar2/cmd_vel`
