# Autonomous Indoor Navigation and V2V Overtaking on the Quanser QCar 2

> **Cartographer localization • Model Predictive Control • LiDAR overtaking • Depth-camera emergency stop • V2V communication**

This repository contains the QCar 2 side of the V2V project, built on **ROS 2 Humble**. The car follows a recorded reference path with a Model Predictive Controller, overtakes a vehicle in its lane using its LiDAR, stops on its own if the depth camera sees something close, and receives the other vehicle's state over a V2V UDP link. A browser dashboard shows the camera feeds, V2V and safety state, and a live track map.

---

## Contents

1. [Quick start: `run_qcar2_stack.sh`](#quick-start-run_qcar2_stacksh)
2. [Dashboard](#dashboard)
3. [Hardware and software](#hardware-and-software)
4. [Architecture](#architecture)
5. [Tuning switches](#tuning-switches)
6. [Changing code on the car](#changing-code-on-the-car)
7. [Manual operation (without the launcher)](#manual-operation-without-the-launcher)
8. [Mapping and recording a new trajectory](#mapping-and-recording-a-new-trajectory)
9. [Troubleshooting](#troubleshooting)

---

## Quick start: `run_qcar2_stack.sh`

`run_qcar2_stack.sh` is the normal way to run the car. It starts the whole stack, waits until the MPC is ready, and then gives you a menu to start and stop driving.

### 1. Connect

```bash
ssh qcar2                   # or: ssh nvidia@192.168.0.53
```

`~/.ssh/config` on the lab server has a `qcar2` entry (IP `192.168.0.53`, user `nvidia`, key `~/.ssh/id_ed25519_qcar2`). One-time key setup, from the lab server (asks for the car's password once):

```bash
ssh-copy-id -i ~/.ssh/id_ed25519_qcar2.pub nvidia@192.168.0.53
ssh qcar2 echo ok           # should connect with no password prompt
```

If `ssh-copy-id` isn't available:

```bash
cat ~/.ssh/id_ed25519_qcar2.pub | ssh nvidia@192.168.0.53 \
  'mkdir -p ~/.ssh && chmod 700 ~/.ssh && cat >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys'
```

### 2. Launch

```bash
export ROS_DOMAIN_ID=42
~/ros2_ws_video/run_qcar2_stack.sh
```

> **Run it in your own interactive terminal.** The menu reads single keypresses, and the E-Stop has to stay one key away from your hands. Don't start it from a script or a non-interactive `ssh host "command"`: you would get a running stack with no way to stop it.

The QCar 2 always uses `ROS_DOMAIN_ID=42`. `.bashrc` sets it for interactive logins; export it yourself in any other shell.

### 3. Drive

The car does **not** move when the stack comes up. Once the script prints `stack up. Nothing drives until you press 'r'`, use the menu:

| Key | Action |
| --- | --- |
| `r` | **Resume**: start driving. Refused until `path_mpc` reports `Localization stable`. |
| `e` | **E-Stop**: sends SIGINT to `qcar2_hardware` and confirms it exited. The motors are zeroed; driving is over until you relaunch. |
| `l` | **Relaunch** hardware and localization after an E-Stop. The other nodes keep running. |
| `s` | **Status**: which nodes are running, the last `path_mpc` line, and a duplicate check. |
| `q` | **Quit**: full shutdown of everything the script started. |

`Ctrl+C` also runs the full shutdown.

**Why the E-Stop kills the hardware node:** publishing `/motion_enable false` does not reliably stop the car mid-drive (`notes.md`, Issue 9). Killing `qcar2_hardware` with `-9` doesn't stop the motors either, because the motor-zeroing code is in its destructor, which SIGKILL skips (Issue 17). Only SIGINT works. If the script warns that `qcar2_hardware` did not exit, **cut physical power to the car**.

### What the launcher starts

| Order | Process | Log in `~/qcar2_run_logs/` |
| --- | --- | --- |
| 1 | `ros2 launch qcar2_nodes qcar2_cartographer_launch.py` (hardware, IMU, LiDAR, camera, Cartographer, robot_state_publisher) | `localization.log` (previous run kept as `.prev`) |
| 2 | `lidar_overtake`: LiDAR obstacle detection and the overtake state machine | `lidar_overtake.log` |
| 3 | `depth_emergency_node`: RealSense emergency stop | `depth_emergency.log` |
| 4 | `sound_node`: audio notifications | `sound_node.log` |
| 5 | `v2v_receiver`: V2V link (params: `config/v2v_params.yaml`) | `v2v_receiver.log` |
| 6 | `v2v_dashboard.py --role qcar2`: browser dashboard | `dashboard.log` |
| 7 | `path_mpc`: MPC path follower (started with `PYTHONUNBUFFERED=1`) | `path_mpc.log` |

Before starting localization, the script clears leftover processes from an earlier launch. It stops the RealSense camera gently first, because a `-9` on it can wedge the USB stream until a power cycle. At the end it checks for duplicate nodes.

> `l` restarts **only** hardware and localization. After changing any node's code, press `q` and run the script again. See [Changing code on the car](#changing-code-on-the-car).

---

## Dashboard

Open **`http://192.168.0.53:8090/`** in a browser on the lab network. The page shows:

- Status pills for each robot
- Both camera feeds, at equal fixed height
- V2V link and motion tables, with the **Gap (m)** row highlighted
- Safety state, and QCar 2's encounter state
- Active ROS nodes on each robot
- Live track map (reference path, QCar 2 in green, the other vehicle in orange)

The layout fits one screen so it can be screen-recorded. Below 1100 px width it switches back to a scrolling layout.

| Setting | Where | Effect |
| --- | --- | --- |
| `--cam-h` | CSS in `v2v_dashboard.py` | Height of both camera panes (default `38vh`) |
| `--cam-trim` | CSS in `v2v_dashboard.py` | Pixels trimmed from each camera image (default `30px`) |
| `--port` | command line | HTTP port (default `8090`) |

**The page is generated by the running process.** After editing `v2v_dashboard.py`, restart the dashboard. A browser refresh alone won't change anything.

To start the dashboard by hand, source ROS first; it needs `rclpy`:

```bash
cd ~/ros2_ws_video
source /opt/ros/humble/setup.bash && source install/setup.bash && export ROS_DOMAIN_ID=42
python3 v2v_dashboard.py --role qcar2 --peer-host 192.168.0.100 \
  --trajectory ~/ros2_ws_video/track_run_cartographer_final_leftshift.npy
```

---

## Hardware and software

| Component | Description |
| --- | --- |
| Vehicle | Quanser QCar 2 |
| LiDAR | RPLidar A2M12 |
| Depth camera | Intel RealSense D435i |
| Compute | NVIDIA Jetson |
| Sensors | Wheel encoders, IMU |

| Software | Version |
| --- | --- |
| Ubuntu | 20.04.6 LTS (on the Jetson) |
| ROS 2 | Humble Hawksbill |
| Python | 3.8.10 |
| Localization | Cartographer (frozen `.pbstream` map) |
| MPC solver | CasADi + IPOPT |
| Other | Nav2 map server, NumPy, SciPy, Matplotlib |

---

## Architecture

### Repository layout

```text
.
├── README.md                  this file
├── run_qcar2_stack.sh         all-in-one launcher with Resume/E-Stop menu
├── v2v_dashboard.py           browser dashboard (both robots)
├── seed_cartographer.py       pose reseeding helper (currently disabled in the launcher)
├── notes.md                   findings and the issues/fixes history (the why)
├── memory.md                  session log
├── utils/                     trajectory recording and path-processing scripts
└── src/
    ├── qcar2_nodes/           hardware drivers and launch files (Cartographer, IMU, LiDAR, camera)
    ├── qcar2_interfaces/      message definitions
    ├── qcar_science_night_pkg/
    │   ├── config/            v2v_params.yaml, sounds, AMCL/SLAM configs
    │   ├── launch/            older AMCL-based launch files
    │   └── qcar_science_night_pkg/
    │       ├── path_mpc_node.py            MPC path follower
    │       ├── lidar_overtake_node.py      LiDAR sectors and overtake decisions
    │       ├── overtake_state_machine.py   LK / LC_LEFT / LC_RIGHT / WAIT / E-STOP
    │       ├── lidar_sector_analyzer.py    front, left, right and emergency boxes
    │       ├── depth_emergency_node.py     depth-camera emergency stop
    │       ├── v2v_receiver_node.py        V2V UDP receiver and gap computation
    │       ├── v2v_common.py               shared path projection and gap math
    │       └── sound_node.py               audio notifications
    └── qcar_lane_pkg/         earlier camera lane-keeping work
```

### Nodes (`ros2 run qcar_science_night_pkg <name>`)

| Executable | Purpose |
| --- | --- |
| `path_mpc` | Follows the reference path with a nonlinear MPC (kinematic bicycle model, CasADi/IPOPT). Applies the overtake lateral offset and speed limits. |
| `lidar_overtake` | Splits the LiDAR scan into front, left, right and emergency sectors, and runs the overtake state machine. |
| `depth_emergency_node` | Stops the car when the depth camera sees something inside its region of interest. |
| `v2v_receiver` | Receives the other vehicle's state over UDP. Computes the along-path gap between the two vehicles, bumper to bumper. |
| `sound_node` | Plays audio for events such as detecting an obstacle. |
| `amcl_nudge`, `lane_centering_node`, `reverse_trailer_mpc` | Older or experimental; not started by the launcher. |

### Overtake state machine

The state names follow the IDEAM paper's modes. The car's lateral offset is `overtake_offset` in `LC_LEFT` and 0 in every other state.

| State | Meaning | Goes to | When |
| --- | --- | --- | --- |
| `LK` | Lane keeping | `LC_LEFT` | Obstacle confirmed, left lane clear, overtaking allowed |
| | | `WAIT_FOR_CLEAR` | Obstacle confirmed but it can't pass (left lane blocked, curve too tight, or too close) |
| `LC_LEFT` | In the left lane, passing | `LC_RIGHT` | Front clear, right side empty, heading stable, minimum time in the lane reached |
| | | `LC_RIGHT` (abort) | Left lane becomes blocked while the original lane is empty |
| | | `WAIT_FOR_CLEAR` | Left lane blocked and the original lane occupied |
| `LC_RIGHT` | Returning to the original lane | `LK` | Return confirmed |
| `WAIT_FOR_CLEAR` | Stopped | `LK` | Obstacle no longer confirmed |
| | | `LC_LEFT` | Passing becomes possible |
| `EMERGENCY_STOP` | Stopped, something is too close | `WAIT_FOR_CLEAR` | Emergency box is clear again |

From **any** state, something inside the emergency box sends the car to `EMERGENCY_STOP`.

### Control flow

```text
RPLidar ──▶ Cartographer (frozen map) ──▶ map → base_link ──▶ path_mpc ──▶ QCar 2 motors
   │                                                            ▲
   └──▶ lidar_overtake ── offset, speed cap, drive state ───────┤
                                                                │
RealSense D435i ──▶ depth_emergency_node ── emergency stop ─────┤
                                                                │
V2V peer ── UDP ──▶ v2v_receiver ── gap, follow-speed cap ──────┘
```

---

## Tuning switches

These values set how overtaking and following behave. All are in `src/qcar_science_night_pkg/qcar_science_night_pkg/`.

| Setting | File | Value | Effect |
| --- | --- | --- | --- |
| `REQUIRE_FRONT_DISTANCE_TO_OVERTAKE` | `lidar_overtake_node.py` | `True` | The overtake starts only while the obstacle is between the commit floor and the detection distance. |
| `front_stop_straight_m` | `lidar_overtake_node.py` | `1.00` m | Distance on straights at which an obstacle ahead is detected. |
| `overtake_start_min_distance` | `lidar_overtake_node.py` | `0.80` m | Commit floor (plus a 0.05 m margin). Keep it well above the 0.70 m emergency distance, or the emergency stop triggers before an overtake can start. |
| `overtake_offset` | `lidar_overtake_node.py` | `0.45` m | Lateral offset during the pass. The left lane's middle is 0.40 m, since lanes are 0.40 m wide. Don't go below ~0.40, or the two robots get too close. |
| `no_obstacle_confirm_required` | `lidar_overtake_node.py` | `15` | Ticks with a clear front before returning to the lane. This is the only delay before cutting back in. |
| `V2V_FOLLOW_SPEED_CAP_ENABLED` | `path_mpc_node.py` | `False` | When `True`, the other vehicle's speed and gap limit QCar 2's speed while following. |
| `v_curve_min` | `path_mpc_node.py` | `0.60` m/s | Keep above **0.5625**. Below that, the MPC preview halves from 1.50 m to 0.75 m and the steering oscillates in tight curves. |
| `CURVE_AWARE_ROI_ENABLED` | `depth_emergency_node.py` | `True` | Narrows the depth-camera region in curves so walls on the inside don't trigger stops. |

The emergency stop (0.70 m on straights, 0.65 m in curves) and the hard stop (0.40 m) are independent of these switches and always active.

---

## Changing code on the car

A running node keeps the code it started with. **Rebuilding doesn't change a running process.** After any change you have to restart the node that uses it.

```bash
# 1. Copy the changed file to the car
scp src/qcar_science_night_pkg/qcar_science_night_pkg/lidar_overtake_node.py \
    qcar2:~/ros2_ws_video/src/qcar_science_night_pkg/qcar_science_night_pkg/

# 2. Rebuild just that package
ssh qcar2 "source /opt/ros/humble/setup.bash && cd ~/ros2_ws_video && colcon build --packages-select qcar_science_night_pkg"

# 3. Load it: in the launcher, press e, then q, then run ./run_qcar2_stack.sh again
```

To restart one node without restarting the whole stack, **first make sure the car isn't driving** (`tail -2 ~/qcar2_run_logs/path_mpc.log`). While `lidar_overtake` restarts, there is no obstacle detection.

```bash
ssh qcar2 "pkill -9 -f 'qcar_science_night_pkg/lib/qcar_science_night_pkg/lidar_overtake'"
ssh qcar2 "source /opt/ros/humble/setup.bash && source ~/ros2_ws_video/install/setup.bash && export ROS_DOMAIN_ID=42 && cd ~/ros2_ws_video && setsid nohup ros2 run qcar_science_night_pkg lidar_overtake --ros-args -r __node:=lidar_overtake > ~/qcar2_run_logs/lidar_overtake.log 2>&1 < /dev/null & disown"
ssh qcar2 "ps aux | grep -E 'lidar_overtake|depth_emergency|path_mpc |sound_node' | grep -v grep"   # no duplicates
```

The same pattern works for `path_mpc`, `depth_emergency_node` and `sound_node`: swap the executable and log name. For `path_mpc`, also set `PYTHONUNBUFFERED=1` (see [Troubleshooting](#troubleshooting)).

---

## Manual operation (without the launcher)

### Check for duplicates first

```bash
ps aux | grep -E 'qcar2_hardware|lidar_overtake|depth_emergency|path_mpc|sound_node|v2v_receiver|cartographer' | grep -v grep
```

### Start the nodes, one terminal each

In every terminal: `ssh qcar2`, then `cd ~/ros2_ws_video`.

```bash
# Terminal 1 — localization (also starts hardware, LiDAR and camera)
ros2 launch qcar2_nodes qcar2_cartographer_launch.py

# Terminal 2 — LiDAR obstacle avoidance
ros2 run qcar_science_night_pkg lidar_overtake --ros-args -r __node:=lidar_overtake

# Terminal 3 — depth-camera emergency stop
ros2 run qcar_science_night_pkg depth_emergency_node --ros-args -r __node:=depth_emergency_node

# Terminal 4 — sound (optional; if it's silent, apply the audio fix below)
ros2 run qcar_science_night_pkg sound_node --ros-args -r __node:=sound_node

# Terminal 5 — V2V receiver
ros2 run qcar_science_night_pkg v2v_receiver --ros-args \
  --params-file install/qcar_science_night_pkg/share/qcar_science_night_pkg/config/v2v_params.yaml

# Terminal 6 — MPC (wait for "Localization stable. MPC enabled.")
PYTHONUNBUFFERED=1 ros2 run qcar_science_night_pkg path_mpc --ros-args -r __node:=path_mpc
```

### Start driving

```bash
ros2 topic pub --once /mission_restart std_msgs/msg/Bool "{data: true}"   # only if a previous run already finished
ros2 topic pub --once /motion_enable std_msgs/msg/Bool "{data: true}"
```

A single `--once` can be lost before discovery completes. If nothing happens, publish again. The launcher sends it three times for this reason.

`/mission_restart` matters after a run has finished. The MPC sets `mission_done` at the end of the run, and after that, `/motion_enable` alone does nothing.

### Stop safely mid-drive

```bash
# 1. Stop qcar2_hardware gracefully — NEVER use -9 here
ps aux | grep qcar2_hardware | grep -v grep     # get its PID
kill -2 <PID>                                   # SIGINT
ps aux | grep qcar2_hardware | grep -v grep     # must print nothing

# 2. Only once qcar2_hardware has exited, clean up the rest (-9 is safe for these)
pkill -9 -f 'cartographer_node|cartographer_occupancy_grid_node|qcar2_nodes/lib|path_mpc|lidar_overtake|depth_emergency_node|sound_node|v2v_receiver|teleop_twist_keyboard|dist_to_start.py|qcar2_trajectory_recorder.py|ros2 launch qcar2_nodes'
```

The LiDAR may keep spinning afterwards. Only a power cycle stops it.

### Duplicate `qcar2_hardware` after a failed launch

List both instances with `ps aux | grep qcar2_hardware | grep -v grep`. Keep the **older** one; it's the one holding the GPIO. Kill only the newer duplicate's two PIDs (wrapper and binary). Never use `pkill -f qcar2_hardware` here, because it matches both.

```bash
kill -9 <wrapper_PID> <binary_PID>
```

### Launch detached over non-interactive SSH

Non-interactive shells don't read `.bashrc`, so source ROS and export the domain explicitly:

```bash
ssh qcar2 "source /opt/ros/humble/setup.bash && source ~/ros2_ws_video/install/setup.bash && export ROS_DOMAIN_ID=42 && cd ~/ros2_ws_video && setsid nohup ros2 launch qcar2_nodes qcar2_cartographer_launch.py > ~/qcar2_run_logs/localization.log 2>&1 < /dev/null & disown"
```

Use the same form for each node in the list above. **Keep a way to stop the car within reach.** The E-Stop menu only exists in `run_qcar2_stack.sh`.

### Audio fix (ALSA levels reset on every power cycle)

```bash
amixer sset 'DSPK1 Audio Channels' 2
amixer sset 'DSPK1 FIFO Threshold' 63
```

### Shut down the Jetson

```bash
sudo shutdown now
```

---

## Mapping and recording a new trajectory

### Build a new map (when the track layout changes)

```bash
ros2 launch qcar2_nodes qcar2_cartographer_original_launch.py
```

Drive the whole track slowly and smoothly with teleop (remapped to `/cmd_vel_nav`), covering every corridor and ending near the start. **Save the map before stopping Cartographer:**

```bash
ros2 service call /write_state cartographer_ros_msgs/srv/WriteState "{filename: '/home/nvidia/ros2_ws_video/track_map_new.pbstream'}"
ros2 run nav2_map_server map_saver_cli -f /home/nvidia/ros2_ws_video/track_map_new
```

Then stop Cartographer with `Ctrl+C` and relaunch localization to use the new map.

### Record and process a trajectory

```bash
# Terminal 1: localization
ros2 launch qcar2_nodes qcar2_cartographer_launch.py

# Terminal 2: record one clean closed lap (uses the map → base_link TF)
python3 utils/qcar2_trajectory_recorder.py --ros-args -p trajectory_file:=<path>.csv

# Terminal 3: live distance and direction back to the start, for closing the loop
python3 utils/dist_to_start.py <start_x> <start_y>

# Terminal 4: drive
ros2 run teleop_twist_keyboard teleop_twist_keyboard --ros-args -r cmd_vel:=cmd_vel_nav

# Turn the recording into the MPC reference path (--closed for a full loop)
python3 utils/make_final_mpc_path.py --input <path>.csv --output <path>_final.npy --closed --smoothing 0.0002

# If curvature is noisy at that low smoothing, clean it separately
python3 utils/smooth_curvature.py --input <path>_final.npy --output <path>_final_smoothed.npy --max-curvature 2.5
```

The resulting `.npy` has columns `x, y, yaw, curvature`. To use it, point `forward_trajectory_file` in `path_mpc_node.py` and `--trajectory` in `run_qcar2_stack.sh` at the new file.

### Older AMCL-based localization (sanity checks only, not used for driving)

```bash
ros2 launch qcar_science_night_pkg science_night_slam.launch.py
```

Wait for `AMCL cannot publish a pose ... Please set the initial pose...`, then set the pose:

```bash
ros2 topic pub --once /initialpose geometry_msgs/msg/PoseWithCovarianceStamped "{
  header: {frame_id: 'map'},
  pose: {
    pose: {
      position: {x: 0.0, y: 0.0, z: 0.0},
      orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}
    },
    covariance: [0.25, 0, 0, 0, 0, 0,  0, 0.25, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0.06]
  }
}"
```

Drive a short distance with some turning so it converges, then check `ros2 topic echo /amcl_pose --once`. Aim for x, y and yaw covariance below ~0.02.

### RViz (optional, needs X11 forwarding)

```bash
ssh -X nvidia@192.168.0.53
cd ~/Documents/ACC_Development/isaac_ros_common
export ROS_DOMAIN_ID=42
./scripts/run_dev.sh /home/nvidia/Documents/ACC_Development/Development
```

Inside the container, run `export ROS_DOMAIN_ID=42 && rviz2`. Set **Fixed Frame → map**, then **Add → By topic** and add `/map`, `/scan` and `/camera/color_image`.

---

## Troubleshooting

| Symptom | Cause and fix |
| --- | --- |
| A code or config change has no effect | The running node still has the old code. Restart it: `e`, `q` and relaunch. Rebuilding alone isn't enough. The same applies to the dashboard. |
| `r` is refused, or "MPC did not report ready" | `path_mpc` hasn't logged `Localization stable` yet. Check `~/qcar2_run_logs/path_mpc.log`. If you start `path_mpc` by hand with output going to a file, set `PYTHONUNBUFFERED=1`, otherwise that line can sit unwritten in Python's buffer. |
| Car doesn't move after `r` | Check that `qcar2_hardware` is running (`s`). If a run already finished, publish `/mission_restart` first. |
| Car stops unexpectedly | `lidar_overtake.log` shows `state=WAIT_FOR_CLEAR` or `EMERGENCY_STOP` with the sector distances. Also check `depth_emergency.log`. |
| No overtake, car waits behind the obstacle | In `lidar_overtake.log`, check `allow_raw` (the MPC blocks overtaking in tight curves), `L` (left lane clear) and `enough_dist`. |
| Two copies of a node are running | Run the duplicate check (`s`). Kill the newer copy. For `qcar2_hardware`, follow [Duplicate `qcar2_hardware`](#duplicate-qcar2_hardware-after-a-failed-launch). |
| Steering jerks back and forth in tight curves | `v_curve_min` has dropped below 0.5625 m/s. See [Tuning switches](#tuning-switches). |
| Dashboard camera pane freezes briefly | The log shows `empty frame (0x0, 0 bytes)`: the camera node is publishing empty frames. If it's frequent, restart the camera (relaunch localization). |
| Dashboard won't start by hand: `No module named 'rclpy'` / `'yaml'` | ROS isn't sourced, or a Python virtualenv is active. Source `/opt/ros/humble/setup.bash` in a shell without a venv. |
| Localization never becomes stable | Check `localization.log`. Make sure the car starts on the mapped track. Cartographer relocalizes globally, since pose seeding is disabled. |
| No sound | Apply the [audio fix](#audio-fix-alsa-levels-reset-on-every-power-cycle). The levels reset on every power cycle. |

---

## Acknowledgements

This project builds on Cartographer, Navigation2, CasADi and IPOPT, with custom software for path following, LiDAR overtaking, emergency stopping and V2V communication on the Quanser QCar 2. The overtake state names follow Shu, Zhou and Zhang, *Agile Decision-Making and Safety-Critical Motion Planning for Emergency Autonomous Vehicles*, IEEE T-ITS 26(9), 2025.
