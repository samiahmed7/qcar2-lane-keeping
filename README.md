# QCar2 ML Steering Lane Keeping

ROS 2 Jazzy + Gazebo Harmonic workspace for a Quanser QCar2 lane-keeping and
obstacle-avoidance stack.

The current milestone runs a single QCar2 in the `lab_track` Gazebo world,
detects lane markings with an ML steering pipeline, uses LiDAR to trigger an
obstacle-avoidance lane switch, and then continues in the new lane. This stack
is being prepared for deployment on the real Quanser QCar2 after simulation
validation.

## Current Stack

- `qcar2_description`: QCar2 URDF, meshes, sensors, and Ackermann setup.
- `qcar2_worlds`: Gazebo `lab_track` world and map assets.
- `qcar2_bringup`: simulation bringup and ROS/Gazebo bridge launch files.
- `qcar2_autonomy`: perception, ML lane steering, obstacle behavior, lane-change
  planning, command muxing, and safety nodes.
- `qcar2_msgs`: shared custom ROS messages.

Large model weights are hosted on Hugging Face:

```text
https://huggingface.co/HammadNaseer/qcar2-ml-steering-weights
```

See [docs/ML_STEERING_WEIGHTS.md](docs/ML_STEERING_WEIGHTS.md) for class names,
sample scores, and download instructions.

## Build

```bash
cd ~/rosbot_ws
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install --packages-up-to qcar2_autonomy qcar2_bringup
source install/setup.bash
```

If the ML weights are not already present locally:

```bash
scripts/download_ml_steering_weights.sh
```

Expected weight paths:

```text
weights/car_track_v3_lane.onnx
weights/car_track_v3_lane.classes.txt
```

## Run The ML Obstacle-Crossing Demo

### Terminal 1 - Gazebo Headless

```bash
cd ~/rosbot_ws
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ros2 launch qcar2_bringup sim_bringup.launch.py headless:=true
```

Wait for this line:

```text
Camera images for [qcar2::base_link::front_camera] advertised on [/qcar2/front_camera/image]
```

### Terminal 2 - Spawn Obstacle

```bash
cd ~/rosbot_ws
source /opt/ros/jazzy/setup.bash
source install/setup.bash
./scripts/spawn_box.sh 2.0 -6.20
```

### Terminal 3 - Autonomy

```bash
cd ~/rosbot_ws
./scripts/run_autonomy.sh
```

Wait for:

```text
Autonomy up. Detector loads in ~10 s
```

### Terminal 4 - Viewer

```bash
cd ~/rosbot_ws
source /opt/ros/jazzy/setup.bash
source install/setup.bash
python3 scripts/view_overlay.py
```

## What The Demo Does

1. Gazebo runs the QCar2 and publishes camera/LiDAR topics.
2. A box is spawned ahead of the starting lane at `(x=2.0, y=-6.20)`.
3. `run_autonomy.sh` starts:
   - RF-DETR ONNX lane detector
   - PID lane follower
   - LiDAR-tracked obstacle state machine
   - open-loop lane-change planner
   - command mux
4. The car detects the obstacle with LiDAR.
5. The planner switches the car into the adjacent lane.
6. The state machine stays in the new lane instead of returning to the first
   lane.
7. The ML tracker regains control after the lane-change maneuver completes.

## Key Topics

| Topic | Purpose |
|---|---|
| `/qcar2/front_camera/image` | Front camera stream from Gazebo. |
| `/qcar2/lidar/scan` | 2D LiDAR scan. |
| `/planning/validated_target_x` | ML lane target consumed by PID. |
| `/system/current_state` | High-level lane state: `LANE_KEEP`, `LANE_CHANGE`, `LANE_RETURN`. |
| `/qcar2/control/raw_cmd_vel` | Lane-keep command before muxing. |
| `/qcar2/control/maneuver_cmd_vel` | Lane-change planner command. |
| `/model/qcar2/cmd_vel` | Final Gazebo drive command. |

## Useful Scripts

| Script | Purpose |
|---|---|
| `scripts/run_autonomy.sh` | Starts autonomy only after Gazebo is already running. |
| `scripts/spawn_box.sh` | Spawns a tall LiDAR-visible test obstacle. |
| `scripts/view_overlay.py` | Shows the live RF-DETR/debug overlay. |
| `scripts/stop_lane_keeping.sh` | Stops the running autonomy/sim helper processes. |
| `scripts/download_ml_steering_weights.sh` | Downloads model weights from Hugging Face. |

## Notes

- The default autonomy mode uses one-way lane switching for the obstacle pass.
  It does not return to the original lane after crossing.
- The detector can take about 10 seconds to warm up after `run_autonomy.sh`.
- Headless Gazebo is preferred because the GUI plus GPU inference can overload
  a single laptop GPU.
- Large weights, Roboflow caches, videos, and virtual environments are excluded
  from Git and should stay on Hugging Face or local disk.

## Repositories

Code:

```text
https://github.com/6hammad9/qcar2-lane-keeping
```

Weights:

```text
https://huggingface.co/HammadNaseer/qcar2-ml-steering-weights
```
