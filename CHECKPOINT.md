# Project Checkpoint - QCar2 Lane-Change Sim

**Current architecture:** one maintained lane-change stack.

| Node | Role |
|---|---|
| `lane_filter` | Camera BEV mask, multi-line lane identity, selected/blended centreline measurements, Clothoid EKF, Stanley-style tracking to `/qcar2/cmd_vel_desired` |
| `lane_change_manager` | Behaviour state, lane index, target lane index, smooth lane-change offset, LiDAR pre-check, automatic obstacle-triggered lane-change start |
| `safety_filter` | Path-aware LiDAR safety gate and only publisher to `/model/qcar2/cmd_vel` |
| `ipm_calibrate` | Click-to-calibrate IPM YAML |

The old centroid follower and old two-line detector were removed from source to
avoid duplicate drivers and ambiguous command ownership.

## Environment

| Item | Value |
|---|---|
| OS | Ubuntu 24.04 on WSL2 |
| ROS distro | ROS2 Jazzy |
| Simulator | Gazebo Harmonic (gz-sim8) |
| Workspace | `~/rosbot_ws` |

Required shell setup:

```bash
source /opt/ros/jazzy/setup.bash
source ~/rosbot_ws/install/setup.bash
export GZ_SIM_SYSTEM_PLUGIN_PATH=/opt/ros/jazzy/opt/gz_sim_vendor/lib
export GZ_SIM_RESOURCE_PATH=$(ros2 pkg prefix qcar2)/share:$GZ_SIM_RESOURCE_PATH
```

## Main Run Sequence

```bash
# Terminal 1
ros2 launch qcar2 simulation.launch.py

# Terminal 2
ros2 launch qcar2_line_tracker lane_filter_avoid.launch.py

# Terminal 3
ros2 run rqt_image_view rqt_image_view
# select /qcar2/lane_filter/debug
```

Lane-change requests:

```bash
ros2 topic pub --once /qcar2/lane_change_request std_msgs/msg/String "{data: LEFT}"
ros2 topic pub --once /qcar2/lane_change_request std_msgs/msg/String "{data: RIGHT}"
ros2 topic pub --once /qcar2/lane_change_request std_msgs/msg/String "{data: ABORT}"
```

`lane_filter_avoid.launch.py` also enables automatic lane change from lane `0`
to lane `1` when the forward LiDAR range drops below `2.20 m`. Manual requests
are useful for isolated lane-switch testing; the obstacle test should no longer
need a manual `LEFT` command.

## Topic Ownership

| Topic | Owner | Purpose |
|---|---|---|
| `/qcar2/front_camera/image` | Gazebo bridge | camera input |
| `/qcar2/lidar/scan` | Gazebo bridge | LiDAR input |
| `/qcar2/lane_selection` | `lane_change_manager` | `[current_lane, target_lane, active, direction]` |
| `/qcar2/lane_target_offset` | `lane_change_manager` | smooth lateral blend offset |
| `/qcar2/lane_filter/state` | `lane_filter` | `[c0, c1, c2, c3, lane_width_obs, age]` |
| `/qcar2/lane_filter/debug` | `lane_filter` | camera + BEV + identities |
| `/qcar2/cmd_vel_desired` | `lane_filter` | pre-safety command |
| `/model/qcar2/cmd_vel` | `safety_filter` | final command to Gazebo |
| `/qcar2/safety_status` | `safety_filter` | LiDAR safety diagnostics |

## Lane Identity

The generated worlds draw two solid white lane boundaries and a dashed white
centreline. The maintained detector is `multi_line_detector.py`:

1. white-mask the BEV image;
2. smooth the bottom-third histogram;
3. find up to three peaks with non-maximum suppression;
4. run sliding windows from each peak;
5. classify each marking as solid/dashed from window hit ratio;
6. assign identities `SOLID_LEFT`, `DASHED_MIDDLE`, `SOLID_RIGHT`.

Lane index convention:

| Lane index | Corridor |
|---|---|
| `0` | `DASHED_MIDDLE + SOLID_RIGHT` |
| `1` | `SOLID_LEFT + DASHED_MIDDLE` |

The default `qcar_oval` generated world has `LO=0.40`, but the current IPM
calibration observes the active corridor as about `0.69 m`. Use the observed
IPM metric width for lane-change blending unless the calibration is redone.

During a lane change, `lane_filter` blends measurements between the current and
target lane centrelines, rather than steering to a target offset while perception
keeps tracking the old lane.

To reduce the previous midpoint `COAST`/sticky feel, `lane_filter` keeps short
identity memory and, only while a lane change is active, accepts weaker
single-line evidence if it still has at least two sliding-window centres.

## LiDAR Notes

The QCar2 LiDAR scans horizontally at about `0.19 m` height, so it does not see
painted lane lines or the low track walls. Use the test obstacle for validation:

```bash
ros2 launch qcar2_line_tracker lidar_test_obstacle.launch.py
ros2 topic echo /qcar2/safety_status
```

`/qcar2/safety_status` reports the active safety minimum as `front_min`, plus
`arc_min`, `corridor_min`, `obj_*`, and `global_min/global_angle` for
diagnostics. If only `global_min` sees the object, tune `front_angle_offset_deg`
or inspect whether the object is outside the forward region being debugged.

The default safety gate uses the fresh `/qcar2/lane_filter/state` polynomial as
a swept driving corridor. A LiDAR return blocks only when it overlaps the band
around `y_path = c0 + c1*x + c2*x^2/2 + c3*x^3/6`; it slows below `1.80 m` and
hard-stops below `1.20 m`. If the lane-filter state is stale or missing, the
filter falls back to the conservative car-centred rectangular corridor. The
default path band is `0.18 m` (`vehicle_half_width_m=0.10` plus
`path_safety_margin_m=0.08`), based on the QCar2 URDF footprint rather than the
earlier over-wide `0.30 m` band. The path check begins at
`path_check_start_x_m=0.30`, while closer returns are handled only by a narrow
near-body stop strip (`near_body_stop_y_m=0.07`) so side returns next to the
vehicle do not masquerade as blocked forward path.

`/qcar2/safety_status` reports `mode=PATH` or `mode=FALLBACK`, plus
`path_min`, `path_y`, `path_err`, `path_band`, `path_start`, `near_min`,
`corridor_min`, `arc_min`, and the diagnostic obstacle cluster fields. In
`mode=PATH`, `arc_min`, `corridor_min`, and `obj_*` can see an old-lane obstacle
without stopping the car; `path_min` is the distance that controls slow/stop.

The lane-change manager starts automatic avoidance earlier, at about `2.20 m`,
so the car has room to shift before the final safety gate has to hard-stop.

The manager also clusters the nearest forward LiDAR returns to estimate
`obj_x`, lateral span `obj_y`, and `obj_w`. If a measured object needs more room,
it publishes a larger temporary offset (`extra`) so `lane_filter` tracks lane
`1` plus extra left clearance instead of clipping the request to one lane width.
The manager can increase that target while the maneuver is active as the LiDAR
gets a better obstacle span estimate, then holds the clearance offset until the
object is no longer ahead.

The final safety gate fails closed on `scan=MISSING` or `scan=STALE`; the
published drive command stays zero until fresh LiDAR returns. The lane-change
manager also treats stale LiDAR as blocked and requires fresh lane-filter
measurements (`meas_age`) before committing a completed lane change.

## Package Layout

```text
src/qcar2/
  urdf/QCar2.urdf
  launch/simulation.launch.py

src/qcar2_worlds/
  worlds/qcar_oval.sdf
  worlds/qcar_oval_wide.sdf
  scripts/generate_track.py

src/qcar2_line_tracker/
  qcar2_line_tracker/lane_filter.py
  qcar2_line_tracker/multi_line_detector.py
  qcar2_line_tracker/clothoid_ekf.py
  qcar2_line_tracker/lane_change_manager.py
  qcar2_line_tracker/safety_filter.py
  qcar2_line_tracker/ipm.py
  qcar2_line_tracker/ipm_calibrate.py
  launch/lane_filter_avoid.launch.py
  launch/lane_filter.launch.py
  launch/ipm_calibrate.launch.py
  launch/lidar_test_obstacle.launch.py
```
