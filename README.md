# QCar2 Lane-Keeping Simulation

ROS2 Jazzy + Gazebo Harmonic workspace for the Quanser QCar2 driving autonomously around a closed oval track.

## Quick start
.
```bash
# Terminal 1 — simulation
source /opt/ros/jazzy/setup.bash && source ~/rosbot_ws/install/setup.bash
export GZ_SIM_SYSTEM_PLUGIN_PATH=/opt/ros/jazzy/opt/gz_sim_vendor/lib
export GZ_SIM_RESOURCE_PATH=$(ros2 pkg prefix qcar2)/share:$GZ_SIM_RESOURCE_PATH
ros2 launch qcar2 simulation.launch.py

# Terminal 2 — lane keeping driver (default, smooth, no calibration needed)
source /opt/ros/jazzy/setup.bash && source ~/rosbot_ws/install/setup.bash
ros2 launch qcar2_line_tracker lane_follower.launch.py

# Terminal 3 — debug viewer
ros2 run rqt_image_view rqt_image_view   # select /qcar2/debug/image
```

That's it — the car drives itself around the oval, staying in the right lane between the centre dashes and the outer solid line.

## What's in the workspace

```
rosbot_ws/
├── src/
│   ├── qcar2/                    Robot description (URDF + meshes + sim launch)
│   ├── qcar2_worlds/             Parametric oval track + variants
│   │   ├── worlds/qcar_oval.sdf      default — narrow lane, frequent dashes
│   │   └── worlds/qcar_oval_wide.sdf wider lane, longer dash gaps (stress test)
│   └── qcar2_line_tracker/       Two lane-keeping pipelines
│       ├── lane_follower.py          ★ DEFAULT — centroid + PD driver
│       ├── lane_filter.py            EKF observer/driver (work-in-progress)
│       ├── clothoid_ekf.py             Clothoid Extended Kalman Filter
│       ├── lane_detector.py            Sliding-window line finder in BEV
│       ├── ipm.py                      IPM warp utility
│       └── ipm_calibrate.py            Click-to-calibrate tool for IPM
├── README.md          (you are here)
├── CHECKPOINT.md      Full technical reference + roadmap
└── ISSUES_AND_FIXES.md
```

## The two lane-keeping drivers

### `lane_follower` — the default

The driver this README launches above. Simple and reliable:
- White-mask the camera frame (HSV)
- Find the leftmost and rightmost lane edges in a look-ahead band
- Track lane width dynamically; tolerate dash gaps
- EMA-smoothed midpoint → PD on pixel error

No calibration needed. Tuned for our oval. **This is what the workspace is committed to use right now.**

### `lane_filter` — the EKF pipeline (in development)

Same purpose, very different architecture:
- Inverse-Perspective-Map the camera to a bird's-eye view (requires one-off calibration)
- Sliding-window lane-line detection in BEV, seeded by EKF prediction
- **Clothoid Extended Kalman Filter** consumes lane points + vehicle odometry, produces a clothoid lane model `y(s) = c0 + c1·s + (c2/2)·s² + (c3/6)·s³`
- Confidence-aware: handles single-line scenarios, ambiguous detections, and full perception loss (COAST mode)
- Stanley controller: `δ = k_head·c1 + atan(k_lat·c0/v) + k_ff·κ`

This is the architecture being matured for **real QCar2 hardware** and **multi-vehicle V2V** scenarios where the EKF's metric state, uncertainty signal, and prediction-through-dropouts properties become essential.

For now it's still being tuned, so the centroid driver is the documented default.

## Other launches

```bash
# Drive on the wider-lane track (more challenging — bigger dash gaps)
ros2 launch qcar2 simulation.launch.py world:=qcar_oval_wide y:=-6.275

# Run the EKF observer in parallel with the centroid driver (debug view only)
ros2 run qcar2_line_tracker lane_filter
# Then view /qcar2/lane_filter/debug

# Run the EKF as the driver (must stop lane_follower first)
ros2 launch qcar2_line_tracker lane_filter_drive.launch.py

# Calibrate the IPM perspective transform (one-off)
ros2 launch qcar2_line_tracker ipm_calibrate.launch.py
```

## Generating track variants

The track is procedural:

```bash
python3 src/qcar2_worlds/scripts/generate_track.py \
    --lo 0.55 --tw 1.6 --dash-period 0.9 \
    --world-name qcar_oval_custom \
  > src/qcar2_worlds/worlds/qcar_oval_custom.sdf
colcon build --packages-select qcar2_worlds --base-paths src
```

`--help` for all flags.

## Key topics

| Topic | Type | Carries |
|---|---|---|
| `/qcar2/front_camera/image` | sensor_msgs/Image | 640×480 30 Hz camera |
| `/qcar2/lidar/scan` | sensor_msgs/LaserScan | 720-sample 360° lidar |
| `/model/qcar2/cmd_vel` | geometry_msgs/Twist | drive command |
| `/model/qcar2/odometry` | nav_msgs/Odometry | vehicle pose + velocity |
| `/qcar2/debug/image` | sensor_msgs/Image | annotated centroid debug |
| `/qcar2/lane_filter/debug` | sensor_msgs/Image | composite EKF debug view |
| `/qcar2/lane_filter/state` | std_msgs/Float32MultiArray | `[c0, c1, c2, c3, lane_width_obs, age]` |

See [CHECKPOINT.md](CHECKPOINT.md) for the full technical reference, EKF architecture, mode states, and roadmap.

## Roadmap

- ☐ Phase A4 — finish tuning Clothoid EKF process/measurement noise
- ☐ LiDAR safety filter (emergency stop)
- ☐ Behaviour mode interface for V2V
- ☐ Multi-vehicle namespacing
- ☐ V2V messaging between leader and follower QCar2
- ☐ Lane-change controller
