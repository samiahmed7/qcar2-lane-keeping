# QCar2 BEV-MPC Lane Keeping

ROS 2 Jazzy + Gazebo Harmonic workspace for a Quanser QCar2 lane-keeping and
navigation stack. The current branch focuses on the **HSV + IPM bird's-eye-view
(BEV) lane detector** fused into a **Model Predictive Control (MPC)** route
tracker.

The design is intentionally simple:

- Camera HSV/BEV detects white lane markings and publishes `/qcar2/lane/model`.
- LiDAR detects obstacles and publishes `/mpc/obstacle`.
- The planner builds a route reference, applies a small BEV lane-centering
  correction, and publishes `/mpc/reference_path`.
- MPC tracks that reference and publishes `/model/qcar2/cmd_vel`.

No deep-learning weights are required for the BEV-MPC run path.

## Architecture

```text
/qcar2/front_camera/image
        |
        v
bev_lane_detector_node  -- /qcar2/lane/model
        |                           |
        |                           v
/qcar2/lidar/scan --> mpc_lidar_obstacle_node --> /mpc/obstacle
                                    |
<route>.npy + /model/qcar2/odometry |
        |                           v
        +------------------> mpc_reference_planner_node
                                    |
                                    v
             /mpc/reference_path, /mpc/target_speed, /mpc/mode
                                    |
                                    v
                         mpc_drive_node
                                    |
                                    v
                         /model/qcar2/cmd_vel
```

## Build

```bash
cd ~/rosbot_ws
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install --packages-up-to qcar2_autonomy qcar2_bringup
source install/setup.bash
```

One-time MPC solver dependencies:

```bash
python3 -m pip install --user --break-system-packages cvxpy clarabel
```

## Run: University Track, Right Lane, HSV/BEV + MPC

Use these terminal commands for the main BEV-MPC setup.

Terminal 1: start Gazebo on the university track, spawning the car in the right
lane.

```bash
cd ~/rosbot_ws
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ros2 launch qcar2_bringup sim_bringup.launch.py world:=university_track x:=0.0 y:=-0.25 yaw:=0.0 headless:=false
```

Terminal 2: run MPC navigation with HSV/BEV lane fusion.

```bash
cd ~/rosbot_ws
START_LANE=right LANE_FUSION=1 OVERLAY=1 TARGET_SPEED=0.55 LOOP=false ./scripts/run_mpc.sh
```

Terminal 3: watch the BEV lane model and fusion status.

```bash
ros2 topic echo /qcar2/lane/model
ros2 topic echo /mpc/lane_fusion_status
```

Optional: open the BEV debug image.

```bash
python3 scripts/view_overlay.py /qcar2/lane/debug_image
```

## One-Shot Headless Test

This starts the sim, runs the BEV-MPC stack, records a log, saves a plot, and
cleans up:

```bash
cd ~/rosbot_ws
START_LANE=right LANE_FUSION=1 SPEED=0.55 DUR=300 ./scripts/autorun_mpc_route.sh
```

Outputs:

- `mpc_run_log.npz`
- `mpc_run_plot.png`
- `/tmp/autorun/*.log`

For a full slow headless run on this machine, use a longer duration:

```bash
START_LANE=right LANE_FUSION=1 SPEED=0.55 DUR=900 ./scripts/autorun_mpc_route.sh
```

## Run: Lab Track Obstacle Demo

Terminal 1:

```bash
cd ~/rosbot_ws
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ros2 launch qcar2_bringup sim_bringup.launch.py headless:=true
```

Terminal 2:

```bash
cd ~/rosbot_ws
./scripts/spawn_box.sh 2.0 -6.20
```

Terminal 3:

```bash
cd ~/rosbot_ws
./scripts/run_mpc.sh
```

## Generate Or Refresh The University Track

```bash
cd ~/rosbot_ws
python3 scripts/make_track.py
python3 scripts/make_world.py
python3 scripts/make_route_check.py
colcon build --symlink-install --packages-select qcar2_worlds
```

The important route files are:

- `university_route.npy`: route centerline.
- `university_route_rlane.npy`: right-lane route used by `START_LANE=right`.
- `track_waypoints.npy`: original lab-track route.

## Useful Topics

| Topic | Purpose |
|---|---|
| `/qcar2/front_camera/image` | Raw front camera. |
| `/qcar2/lane/model` | HSV/BEV lane model used by MPC lane fusion. |
| `/qcar2/lane/debug_image` | HSV/BEV debug overlay. |
| `/qcar2/lidar/scan` | 2D LiDAR. |
| `/model/qcar2/odometry` | Vehicle pose and speed. |
| `/mpc/obstacle` | LiDAR obstacle estimate. |
| `/mpc/reference_path` | Route reference after obstacle/lane-fusion corrections. |
| `/mpc/target_speed` | MPC speed target. |
| `/mpc/mode` | Planner state. |
| `/mpc/lane_fusion_status` | Why BEV fusion is active/gated and current offset. |
| `/model/qcar2/cmd_vel` | Final drive command. |

## Useful Scripts

| Script | Purpose |
|---|---|
| `scripts/run_mpc.sh` | Main terminal-run MPC stack. |
| `scripts/autorun_mpc_route.sh` | Headless sim + BEV-MPC + record + plot. |
| `scripts/autorun_bev_fusion.sh` | Lab-track BEV fusion smoke test. |
| `scripts/grab_bev.sh` | Save raw camera and BEV debug PNGs. |
| `scripts/make_track.py` | Generate university route arrays. |
| `scripts/make_world.py` | Generate `university_track.sdf`. |
| `scripts/make_route_check.py` | Generate/check the right-lane route. |
| `scripts/plot_mpc_run.py` | Plot a recorded MPC run. |
| `scripts/view_overlay.py` | View `/qcar2/lane/debug_image` or another image topic. |
| `scripts/spawn_box.sh` | Spawn a LiDAR-visible obstacle. |
| `scripts/stop_lane_keeping.sh` | Stop sim/autonomy processes. |

## Tuning

Primary parameters:

- `src/qcar2_autonomous_lanes/qcar2_autonomy/config/mpc.yaml`: MPC model,
  horizon, and cost weights.
- `src/qcar2_autonomous_lanes/qcar2_autonomy/config/mpc_nodes.yaml`: planner,
  obstacle, lane-fusion, and command-smoothing parameters.

Important environment variables for `scripts/run_mpc.sh`:

- `START_LANE=right`
- `LANE_FUSION=1`
- `TARGET_SPEED=0.55`
- `LOOP=false`
- `LANE_FUSION_GAIN`
- `LANE_FUSION_ALPHA`
- `LANE_FUSION_MAX_CORRECTION`
- `LANE_FUSION_MAX_STEP`
- `LANE_FUSION_HEADING_GATE`
- `COMMAND_SMOOTHING`
- `OMEGA_SMOOTHING_ALPHA`
- `MAX_OMEGA_RATE`

## Notes

- `/model/qcar2/odometry` is spawn-relative in this simulator. For the right-lane
  university route, spawn at `x:=0.0 y:=-0.25 yaw:=0.0` and run with
  `START_LANE=right`.
- Use `LOOP=false` for the university branch route.
- The headless sim can run much slower than wall time on this machine; use a long
  `DUR` for full-route validation.
- The BEV detector is classical HSV/IPM and does not require ML weights.

## Repository

```text
https://github.com/6hammad9/qcar2-lane-keeping
```
