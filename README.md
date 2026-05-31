# QCar2 Lane-Change Simulation

ROS2 Jazzy + Gazebo Harmonic workspace for a Quanser QCar2 driving on an oval
track, staying between white lane markings, stopping for LiDAR obstacles, and
changing lanes when the avoid stack sees a forward obstacle.

## Current Autonomous Lanes Work

The newer scaffold lives in `src/qcar2_autonomous_lanes`. Phase 1 perception is
implemented in `qcar2_autonomy/qcar2_autonomy/bev_lane_detector_node.py`.

It subscribes to `/qcar2/front_camera/image`, detects the three white lane
lines, targets the right lane center between the middle dashed line and right
solid line, and publishes:

- `/qcar2/lane/model` (`qcar2_msgs/msg/LaneModel`)
- `/qcar2/lane/debug_image` (`sensor_msgs/msg/Image`)

Build and run perception:

```bash
cd ~/rosbot_ws
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install --packages-select qcar2_msgs qcar2_autonomy
source install/setup.bash
ros2 launch qcar2_autonomy lane_perception.launch.py
```

## Quick Start

```bash
# Terminal 1 - simulation
cd ~/rosbot_ws
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ros2 launch qcar2_bringup sim_bringup.launch.py [headless:=true/false]

Terminal 2 — obstacle

cd ~/rosbot_ws
source /opt/ros/jazzy/setup.bash
source install/setup.bash
./scripts/spawn_box.sh 2.0 -6.20

Terminal 3 — autonomy

cd ~/rosbot_ws
./scripts/run_autonomy.sh
Wait for: Autonomy up. Detector loads in ~10 s

Terminal 4 — viewer

cd ~/rosbot_ws
source /opt/ros/jazzy/setup.bash
source install/setup.bash
python3 scripts/view_overlay.py
```

## Node Ownership

Each node has one job:

| Node | Owns | Does not own |
|---|---|---|
| `lane_filter` | camera mask, multi-line identity, selected lane centreline, EKF, path tracking | behaviour decisions, final safety veto |
| `lane_change_manager` | lane index, target lane index, lane-change state, smooth blend offset | image processing, steering control |
| `safety_filter` | path-aware LiDAR swept-volume check, stale-command stop, final `/model/qcar2/cmd_vel` | lane perception, lane-change planning |
| `ipm_calibrate` | one-off perspective calibration | driving |

The command path is:

```text
lane_change_manager -> /qcar2/lane_selection
lane_change_manager -> /qcar2/lane_target_offset
lane_filter         -> /qcar2/cmd_vel_desired
safety_filter       -> /model/qcar2/cmd_vel
```

Only `safety_filter` should publish the real Gazebo drive command.

## Lane Identity

The track has three white lane markings:

```text
SOLID_LEFT     DASHED_MIDDLE     SOLID_RIGHT
```

The centre marking is dashed in `qcar2_worlds/scripts/generate_track.py`, so
the detector can identify it without relying on colour. `lane_filter` uses a
multi-line detector instead of the old left-half/right-half detector:

```text
BEV white mask
  -> histogram peak finder
  -> up to 3 sliding-window line tracks
  -> solid/dashed classification
  -> lane corridor selected by lane index
```

Lane index convention:

| Lane index | Corridor |
|---|---|
| `0` | `DASHED_MIDDLE + SOLID_RIGHT` |
| `1` | `SOLID_LEFT + DASHED_MIDDLE` |

With the current IPM calibration, the lane filter observes this corridor as
about `0.69 m` wide. The lane-change manager uses that internal metric width as
its initial shift distance and then keeps refining it from the observed line
spacing.

## Lane Change Commands

`lane_filter_avoid.launch.py` enables automatic obstacle avoidance by default:
when the front LiDAR range drops below `2.20 m` in lane `0`, the manager starts
a left lane change into lane `1`. Manual requests are still available:

```bash
# Change one lane left
ros2 topic pub --once /qcar2/lane_change_request std_msgs/msg/String "{data: LEFT}"

# Change one lane right
ros2 topic pub --once /qcar2/lane_change_request std_msgs/msg/String "{data: RIGHT}"

# Abort active shift back toward the current lane
ros2 topic pub --once /qcar2/lane_change_request std_msgs/msg/String "{data: ABORT}"

# Watch state, target lane, offset, LiDAR gate, and rejection reason
ros2 topic echo /qcar2/lane_change_status
```

For a clean left change from the default right lane, the status should move
through roughly:

```text
state=KEEP lane=0 target=0 offset=+0.000 scan=OK blocked=False
state=PREPARE lane=0 target=1 offset=+0.000
state=CHANGING lane=0 target=1 offset=+0.xxx
state=KEEP lane=1 target=1 offset=+0.000
```

If it stays in `HOLD_TARGET`, watch `c0` and `meas_age`: perception has not yet
accepted that the car is centred in the target lane.

During a lane change, `lane_change_manager` publishes a smooth lateral blend.
`lane_filter` converts that into blended centreline measurements between the
current lane and target lane. This avoids the old one-frame role flip where the
dashed middle line changed from one lane boundary to the other.

If the debug/log output briefly shows `COAST` near the dashed middle line, that
means the detector could not produce at least two centreline measurements for
that frame. The current code keeps short identity memory and accepts weaker
single-line evidence only while a lane change is active, so midpoint dropouts
should be shorter and less sticky than before.

## LiDAR Sanity Check

The QCar2 LiDAR scans horizontally at roughly `0.19 m` above the ground. It will
not detect painted lane lines, and it will not detect the low `0.08 m` track
walls. Use a tall test box when checking obstacle detection:

```bash
# Terminal 1: sim
ros2 launch qcar2 simulation.launch.py

# Terminal 2: lane stack
ros2 launch qcar2_line_tracker lane_filter_avoid.launch.py

# Terminal 3: spawn a tall box directly in front of the default start pose
ros2 launch qcar2_line_tracker lidar_test_obstacle.launch.py

# Terminal 4: confirm scans and safety gate
ros2 topic echo /qcar2/safety_status
ros2 topic echo /qcar2/lane_change_status
```

If `/qcar2/safety_status` says `scan=MISSING`, the LiDAR bridge is not running
or simulation is not up. If it says `scan=STALE`, the bridge stopped updating.
Both states force the final drive command to zero. If it says `scan=OK` but
the object is not showing in `arc_min`, `corridor_min`, `obj_*`, or
`global_min`, it is probably below scan height, too close for the `0.15 m`
LiDAR minimum range, or outside the sensor view. If the object appears at a
consistent non-zero angle in `global_angle`, tune `front_angle_offset_deg` in
`lane_filter_avoid.launch.py`.

When `/qcar2/lane_filter/state` is fresh, the safety filter checks each LiDAR
return against the EKF path corridor instead of a fixed straight-ahead box:

```text
y_path = c0 + c1*x + c2*x^2/2 + c3*x^3/6
blocked if abs(y - y_path) <= vehicle_half_width + path_safety_margin
```

Default safety tuning starts braking when a point is inside that path corridor
at `path_min < 1.80 m` and hard-stops below `1.20 m`. If the lane-filter state
is stale or missing, safety falls back to the conservative car-centred corridor.
The default path band is `0.18 m` (`0.10 m` simulated vehicle half-width plus
`0.08 m` margin), based on the QCar2 URDF collision/visual width. The path
check starts at `x=0.30 m`, roughly past the front bumper; closer points use
only a narrow central near-body stop strip (`x <= 0.30 m`, `|y| <= 0.07 m`) so
side returns while passing a box do not stop the car. `arc_min`, `corridor_min`,
and `obj_*` are still reported for debugging, but in `mode=PATH` they do not
stop the car unless a point overlaps the planned path. Seeing
`mode=PATH path_min=1.10m blocked=True` means the LiDAR obstacle stop is
working. Seeing `obj_*` or `corridor_min` near the old-lane obstacle while
`path_min=inf blocked=False` means the car sees the object but can pass it.

In the full avoid stack, obstacle avoidance should begin earlier than braking:
`lane_change_manager` triggers the left lane change around a `2.20m` forward
obstacle range, and `safety_filter` remains the final veto if the object still
overlaps the driven path.

`/qcar2/lane_change_status` also reports `meas_age`; lane changes only complete
when the lane-filter measurement is fresh.

For obstacle avoidance, the lane-change manager also estimates the nearest
LiDAR obstacle cluster:

```text
obj_x    nearest forward face distance
obj_y    lateral obstacle span in the car frame
obj_w    estimated lateral obstacle width
extra    temporary extra lane-change offset used for clearance
```

When `extra` is non-zero, the car does not aim only for lane `1` centre. It aims
for lane `1` plus that extra left clearance and holds the offset until the
obstacle is no longer ahead. The clearance target can grow during the maneuver
as the LiDAR gets a better view of the obstacle face.
added when the measured obstacle span actually requires it; if lane `1` centre
has enough clearance, `extra` should stay `0.000`.

## Other Launches

```bash
# Observer only: camera + lane_filter debug, no cmd_vel output
ros2 launch qcar2_line_tracker lane_filter.launch.py

# Calibrate the IPM perspective transform
ros2 launch qcar2_line_tracker ipm_calibrate.launch.py

# Drive on the wider-lane track
ros2 launch qcar2 simulation.launch.py world:=qcar_oval_wide y:=-6.275
```

## Key Topics

| Topic | Type | Carries |
|---|---|---|
| `/qcar2/front_camera/image` | `sensor_msgs/Image` | 640x480 camera |
| `/qcar2/lidar/scan` | `sensor_msgs/LaserScan` | 720-sample 360 degree LiDAR |
| `/qcar2/lane_filter/debug` | `sensor_msgs/Image` | camera + BEV + lane identity debug |
| `/qcar2/lane_filter/state` | `std_msgs/Float32MultiArray` | `[c0, c1, c2, c3, lane_width_obs, age]` |
| `/qcar2/lane_change_request` | `std_msgs/String` | `LEFT`, `RIGHT`, `ABORT`, `KEEP` |
| `/qcar2/lane_selection` | `std_msgs/Int32MultiArray` | `[current_lane, target_lane, active, direction]` |
| `/qcar2/lane_target_offset` | `std_msgs/Float32` | smooth lateral blend offset |
| `/qcar2/cmd_vel_desired` | `geometry_msgs/Twist` | pre-safety drive command |
| `/qcar2/safety_status` | `std_msgs/String` | LiDAR gate diagnostics |
| `/model/qcar2/cmd_vel` | `geometry_msgs/Twist` | final Gazebo drive command |

See [CHECKPOINT.md](CHECKPOINT.md) for the longer technical reference.
