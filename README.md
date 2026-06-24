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

No trained model files are required for the BEV-MPC run path.

## Architecture

```text
/qcar2/front_camera/image
        |
        v
reliable_lane_detector_node  -- /qcar2/lane/model
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

## Run: University Track — HSV/BEV + MPC (current, WORKING)

**Use `my_route_clean.npy`** — the manually-driven right-lane route. It covers the
full track and stays on-road through the roundabout. (Do NOT use
`university_route_rlane.npy` — its computed offset rides the roundabout's outer
edge and the car goes off after the roundabout curve.)

**Terminal 1 — Gazebo with GUI:**

```bash
cd ~/rosbot_ws
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ros2 launch qcar2_bringup sim_bringup.launch.py \
  world:=university_track x:=0.0 y:=-0.25 yaw:=0.0 headless:=false
```

**Terminal 2 — MPC + BEV/HSV stack:**

```bash
cd ~/rosbot_ws
WAYPOINTS=~/rosbot_ws/my_route_clean.npy START_LANE=center \
  LANE_FUSION=1 LOOP=false TARGET_SPEED=0.40 ./scripts/run_mpc.sh
```

`START_LANE=center` uses the file directly (no re-localization); the recording
was made from spawn `(0, -0.25, 0)`, so spawn there.

> **Gotcha:** `scripts/run_mpc.sh` passes launch-arg defaults for the most
> important lane-fusion params. To change fusion strength, edit the defaults in
> `run_mpc.sh` or pass the `LANE_FUSION_*` env vars. The reliable detector also
> loads `config/mpc_nodes.yaml` for its smoothing/tracking parameters.

**Terminal 3 — optional diagnostics:**

```bash
ros2 topic echo /mpc/lane_fusion_status --field data   # fusion active/gated
ros2 topic echo /qcar2/lane/model                      # BEV detector output
```

### Key files

| File | Role |
|---|---|
| `my_route_clean.npy` | **THE route** — manually-driven right-lane waypoints, full track, on-road through roundabout (spawn 0,-0.25) |
| `src/.../qcar2_autonomy/qcar2_autonomy/reliable_lane_detector_node.py` | HSV+BEV sliding-window + polynomial lane detector — publishes `/qcar2/lane/model` |
| `src/.../qcar2_autonomy/qcar2_autonomy/mpc_reference_planner_node.py` | Route planner — follows waypoints, fuses BEV correction on straights |
| `src/.../qcar2_autonomy/qcar2_autonomy/mpc_drive_node.py` | Pure MPC tracker — solves QP, publishes cmd_vel |
| `src/.../qcar2_autonomy/qcar2_autonomy/mpc_lidar_obstacle_node.py` | LiDAR obstacle detector |
| `src/.../qcar2_autonomy/config/mpc_nodes.yaml` | All ROS parameters (fusion gains, gates, confidence threshold) |
| `src/.../qcar2_autonomy/config/mpc.yaml` | MPC solver params (horizon, cost weights, vehicle model) |
| `scripts/run_mpc.sh` | Orchestration — localises waypoints, starts detector, launches MPC |
| `scripts/drive_teleop.py` | Keyboard teleop for recording new waypoints (holds steering) |
| `scripts/record_path.sh` | Records `/model/qcar2/odometry` to a `.npy` waypoint file |
| `scripts/plot_mpc_run.py` | Plots a recorded run vs waypoints (cross-track error, speed, steering) |
| `src/.../qcar2_worlds/worlds/university_track.sdf` | Gazebo world (road tiles, lane markings, roundabout) |

### How BEV lane fusion works

```
camera -> HSV white mask -> IPM bird's-eye warp -> column histogram
       -> find dashed centre + solid right edge -> midpoint = right-lane centre
       -> error_px = midpoint - image_centre
       -> planner applies: correction = -error_px * (lane_width_m / est_width_px)
       -> reference path shifted laterally up to ±30 cm
       -> heading gate disengages fusion at curves/junctions (MPC uses waypoints alone)
```

### Straight-line oscillation fix

The car was seeing the lane lines but still wobbled on straights because BEV
fusion was too aggressive: each small pixel movement in the detected target
shifted the MPC reference, then the vehicle response changed the camera target
again. The fix is to make lane fusion slower and ignore tiny pixel jitter:

- `scripts/run_mpc.sh` now loads `config/mpc_nodes.yaml` into
  `reliable_lane_detector_node`, so detector smoothing is actually used.
- Fusion defaults are softer: `LANE_FUSION_GAIN=0.70`,
  `LANE_FUSION_ALPHA=0.08`, `LANE_FUSION_MAX_STEP=0.012`, and
  `LANE_FUSION_MAX_CORRECTION=0.45`.
- `mpc_reference_planner_node.py` adds `lane_fusion_deadband_px`; the default
  `LANE_FUSION_DEADBAND_PX=8.0` means errors inside +/-8 px do not move the
  MPC reference.
- `config/mpc_nodes.yaml` smooths the detector target/track/width estimates and
  keeps command smoothing enabled, reducing steering chatter.

### To record new waypoints

1. Launch Gazebo (any spawn position).
2. `WAYPOINTS=~/rosbot_ws/my_route_loop.npy SPACING=0.10 ./scripts/record_path.sh`
3. `python3 scripts/drive_teleop.py` — W/S speed, A/D steer (values hold), Space stop.
4. Drive one lap in the right lane; Ctrl-C Terminal 2 to save.
5. Set spawn `x`/`y` to the world position where recording started.

### Re-run with the old right-lane route (pre-computed)

```bash
# Terminal 1 — spawn at the branch start
ros2 launch qcar2_bringup sim_bringup.launch.py \
  world:=university_track x:=0.0 y:=-0.25 yaw:=0.0 headless:=false

# Terminal 2
START_LANE=right LANE_FUSION=1 TARGET_SPEED=0.40 LOOP=false ./scripts/run_mpc.sh
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
- `LANE_FUSION_DEADBAND_PX`
- `LANE_FUSION_HEADING_GATE`
- `COMMAND_SMOOTHING`
- `OMEGA_SMOOTHING_ALPHA`
- `MAX_OMEGA_RATE`

## Notes

- `/model/qcar2/odometry` is **spawn-relative** — always starts at (0,0,0). Waypoints
  must be recorded from the same spawn position used at run time.
- `my_route_loop.npy` was recorded from spawn `x:=2.991 y:=3.818 yaw:=0.0`.
  Use that exact spawn when running with `WAYPOINTS=my_route_loop.npy`.
- `LOOP=true` for the loop-only route; `LOOP=false` for branch+route one-shot.
- The BEV detector is classical HSV/IPM — no trained weights required.
- After editing `mpc_nodes.yaml` or `reliable_lane_detector_node.py`, no rebuild
  is needed (symlink-install). Restart `run_mpc.sh` to pick up changes.
- To plot a recorded run: `python3 scripts/plot_mpc_run.py mpc_run_log.npz my_route_loop.npy`

## Repository

```text
https://github.com/6hammad9/qcar2-lane-keeping
```
