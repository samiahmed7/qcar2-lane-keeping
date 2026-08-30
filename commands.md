# QCar Commands Reference

Pure command reference — every runnable command used on the car, no narrative. See `notes.md` for the why/how-it-works behind any of these (especially the safe-stop procedure — Issue 17 there explains why `qcar2_hardware` must never get `-9`).

`.bashrc` auto-sources the workspace and `ROS_DOMAIN_ID=42` on an **interactive** login shell. Non-interactive `ssh host "command"` invocations do NOT get this — export it explicitly in those.

---

## Connect

Key-based, once installed (see below):

```bash
ssh qcar2
```

`~/.ssh/config` on `rosbot-server` has a `qcar2` host entry (IP
`192.168.0.53`, user `nvidia`) using a dedicated key,
`~/.ssh/id_ed25519_qcar2`. Everything in this file also works with the
full `ssh nvidia@192.168.0.53` form if `qcar2` isn't set up on the
machine you're running from.

**One-time setup, from `rosbot-server`** (needs the car reachable and its
password once — after this, no password prompts):

```bash
ssh-copy-id -i ~/.ssh/id_ed25519_qcar2.pub nvidia@192.168.0.53
ssh qcar2 echo ok    # should connect with no password prompt
```

If `ssh-copy-id` isn't available:

```bash
cat ~/.ssh/id_ed25519_qcar2.pub | ssh nvidia@192.168.0.53 \
  'mkdir -p ~/.ssh && chmod 700 ~/.ssh && cat >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys'
```

## Mapping from scratch (if the track layout changes again)

```bash
ros2 launch qcar2_nodes qcar2_cartographer_original_launch.py
```

Drive the full track slowly and smoothly (teleop, remapped to `/cmd_vel_nav`), covering every corridor, ending back near the start. Save the map **before** stopping Cartographer:

```bash
ros2 service call /write_state cartographer_ros_msgs/srv/WriteState "{filename: '/home/nvidia/ros2_ws_izhan/track_map_new.pbstream'}"
ros2 run nav2_map_server map_saver_cli -f /home/nvidia/ros2_ws_izhan/track_map_new
```

Then `Ctrl+C` Cartographer and re-launch the localization stack to use the fresh map.

## Recording a new trajectory

```bash
# Terminal 1: localization (same as above)
ros2 launch qcar2_nodes qcar2_cartographer_launch.py

# Terminal 2: record one clean closed lap (map -> base_link TF)
python3 utils/qcar2_trajectory_recorder.py --ros-args -p trajectory_file:=<path>.csv

# Terminal 3: live distance/direction back to the start point, once closing the loop
python3 utils/dist_to_start.py <start_x> <start_y>

# Terminal 4: drive (teleop)
ros2 run teleop_twist_keyboard teleop_twist_keyboard --ros-args -r cmd_vel:=cmd_vel_nav

# Then process into the final MPC path (--closed since it's a full loop):
python3 utils/make_final_mpc_path.py --input <path>.csv --output <path>_final.npy --closed --smoothing 0.0002

# If curvature comes out noisy from that low smoothing, clean it up separately:
python3 utils/smooth_curvature.py --input <path>_final.npy --output <path>_final_smoothed.npy --max-curvature 2.5
```

## Running the car — 5 terminals

In every terminal: `ssh nvidia@192.168.0.53` then `cd ~/ros2_ws_izhan`.

```bash
# Terminal 1 — Localization (also brings up hardware/LiDAR/camera)
ros2 launch qcar2_nodes qcar2_cartographer_launch.py

# Terminal 2 — LiDAR obstacle avoidance
ros2 run qcar_science_night_pkg lidar_overtake --ros-args -r __node:=lidar_overtake

# Terminal 3 — Depth-camera emergency stop
ros2 run qcar_science_night_pkg depth_emergency_node --ros-args -r __node:=depth_emergency_node

# Terminal 3b — Sound (optional, easy to forget; re-apply the mixer fix below first if silent)
ros2 run qcar_science_night_pkg sound_node --ros-args -r __node:=sound_node

# Terminal 4 — MPC controller (look for "QCar MPC ready: loop_path=True, target_laps=N")
ros2 run qcar_science_night_pkg path_mpc --ros-args -r __node:=path_mpc
```

Check for duplicates, then trigger (Terminal 5) — commands below.

## Launching everything detached over a single non-interactive SSH session

Same 5 nodes as above, but backgrounded so they survive the SSH session closing, with output going to per-node log files. Needs `ROS_DOMAIN_ID=42` exported explicitly on each (non-interactive shells don't get `.bashrc`'s auto-source).

```bash
ssh nvidia@192.168.0.53 "source /opt/ros/humble/setup.bash && source ~/ros2_ws_izhan/install/setup.bash && export ROS_DOMAIN_ID=42 && cd ~/ros2_ws_izhan && setsid nohup ros2 launch qcar2_nodes qcar2_cartographer_launch.py > /home/nvidia/qcar2_launch.log 2>&1 < /dev/null & disown"

ssh nvidia@192.168.0.53 "source /opt/ros/humble/setup.bash && source ~/ros2_ws_izhan/install/setup.bash && export ROS_DOMAIN_ID=42 && cd ~/ros2_ws_izhan && setsid nohup ros2 run qcar_science_night_pkg lidar_overtake --ros-args -r __node:=lidar_overtake > /home/nvidia/lidar_overtake.log 2>&1 < /dev/null & disown"

ssh nvidia@192.168.0.53 "source /opt/ros/humble/setup.bash && source ~/ros2_ws_izhan/install/setup.bash && export ROS_DOMAIN_ID=42 && cd ~/ros2_ws_izhan && setsid nohup ros2 run qcar_science_night_pkg depth_emergency_node --ros-args -r __node:=depth_emergency_node > /home/nvidia/depth_emergency.log 2>&1 < /dev/null & disown"

ssh nvidia@192.168.0.53 "source /opt/ros/humble/setup.bash && source ~/ros2_ws_izhan/install/setup.bash && export ROS_DOMAIN_ID=42 && cd ~/ros2_ws_izhan && setsid nohup ros2 run qcar_science_night_pkg sound_node --ros-args -r __node:=sound_node > /home/nvidia/sound_node.log 2>&1 < /dev/null & disown"

ssh nvidia@192.168.0.53 "source /opt/ros/humble/setup.bash && source ~/ros2_ws_izhan/install/setup.bash && export ROS_DOMAIN_ID=42 && cd ~/ros2_ws_izhan && setsid nohup ros2 run qcar_science_night_pkg path_mpc --ros-args -r __node:=path_mpc > /home/nvidia/path_mpc.log 2>&1 < /dev/null & disown"
```

Verify: wait for `path_mpc.log` to show `Localization stable. MPC enabled.` before triggering `/motion_enable`.

## Check for duplicates (before any launch or test)

```bash
ps aux | grep -E 'qcar2_hardware|lidar_overtake|depth_emergency|path_mpc|sound_node|cartographer' | grep -v grep
```

## Start driving (once the stack is up)

```bash
ros2 topic pub --once /mission_restart std_msgs/msg/Bool "{data: true}"   # only if a previous run already finished
ros2 topic pub --once /motion_enable std_msgs/msg/Bool "{data: true}"
```

## Stop safely, mid-drive

```bash
# 1. Graceful stop of qcar2_hardware — NEVER -9 here
ps aux | grep qcar2_hardware | grep -v grep     # get its PID
kill -2 <PID>                                   # SIGINT
ps aux | grep qcar2_hardware | grep -v grep     # must print nothing — confirms it exited
tail -3 ~/qcar2_hardware.log                    # or the launch log — should show "qcar2 exit"

# 2. Only after qcar2_hardware is confirmed dead, clean up everything else (-9 is safe for these)
pkill -9 -f 'cartographer_node|cartographer_occupancy_grid_node|qcar2_nodes/lib|qcar2_launch|path_mpc|lidar_overtake|depth_emergency_node|sound_node|teleop_twist_keyboard|dist_to_start.py|qcar2_trajectory_recorder.py|ros2 launch qcar2_nodes'

# LiDAR may keep physically spinning after this — power-level behavior, only a power-cycle stops it.
```

## Shut down the Jetson itself

```bash
sudo shutdown now   # needs the nvidia user's sudo password
```

---

## Iterating on one node's code without restarting everything

Most tuning work is "change one value, redeploy, retest" — restarting the whole stack each time is slow and drops obstacle detection while `lidar_overtake` relaunches.

```bash
# 1. Sync just the changed file
scp src/qcar_science_night_pkg/qcar_science_night_pkg/lidar_overtake_node.py \
    nvidia@192.168.0.53:~/ros2_ws_izhan/src/qcar_science_night_pkg/qcar_science_night_pkg/

# 2. Rebuild just that package (must cd into the workspace first — 8 workspace dirs exist on this machine)
ssh nvidia@192.168.0.53 "source /opt/ros/humble/setup.bash && cd ~/ros2_ws_izhan && colcon build --packages-select qcar_science_night_pkg"

# 3. Kill only the one node that changed (exact path pattern, not just the executable name)
ssh nvidia@192.168.0.53 "pkill -9 -f 'qcar_science_night_pkg/lib/qcar_science_night_pkg/lidar_overtake'"

# 4. Relaunch it detached (setsid+nohup survives the SSH session closing; output goes to a log file)
ssh nvidia@192.168.0.53 "source /opt/ros/humble/setup.bash && source ~/ros2_ws_izhan/install/setup.bash && export ROS_DOMAIN_ID=42 && cd ~/ros2_ws_izhan && setsid nohup ros2 run qcar_science_night_pkg lidar_overtake --ros-args -r __node:=lidar_overtake > /home/nvidia/lidar_overtake.log 2>&1 < /dev/null & disown"

# 5. Verify — no duplicates, log looks clean
ssh nvidia@192.168.0.53 "ps aux | grep -E 'lidar_overtake|depth_emergency|path_mpc |sound_node' | grep -v grep"
ssh nvidia@192.168.0.53 "tail -5 ~/lidar_overtake.log"
```

Same pattern for `path_mpc`, `depth_emergency_node`, `sound_node` — swap the executable name and log filename. Each node's log lives at `~/<node_name>.log`.

Before touching anything, always check the car isn't currently mid-drive: `tail -2 ~/path_mpc.log`.

## Duplicate `qcar2_hardware` cleanup (a second launch failed but left a zombie)

Identify both PIDs first (`ps aux | grep qcar2_hardware | grep -v grep` — two lines instead of one), keep the older one (earlier start time = the one that actually got the GPIO), kill only the newer duplicate's two PIDs directly — never `pkill -f qcar2_hardware` here, it would match both:

```bash
kill -9 <wrapper_PID> <binary_PID>
```

## Audio fix (ALSA mixer levels reset on every power cycle)

```bash
amixer sset 'DSPK1 Audio Channels' 2
amixer sset 'DSPK1 FIFO Threshold' 63
```

## AMCL-based flow (older, localization sanity-check only — not used for driving)

```bash
ros2 launch qcar_science_night_pkg science_night_slam.launch.py
```

Wait for `AMCL cannot publish a pose or update the transform. Please set the initial pose...` (expected). Then, in a second terminal:

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

Drive a short distance with some turning (third terminal, teleop) to help it converge, then check:

```bash
ros2 topic echo /amcl_pose --once
```

Target x/y/yaw covariance all below ~0.02.

## RViz visualization (optional, needs X11 forwarding)

```bash
ssh -X nvidia@192.168.0.53
cd ~/Documents/ACC_Development/isaac_ros_common
export ROS_DOMAIN_ID=42
./scripts/run_dev.sh /home/nvidia/Documents/ACC_Development/Development
```

Inside the container: `export ROS_DOMAIN_ID=42 && rviz2`. Set **Fixed Frame → map**, then **Add → By topic** and add `/map`, `/scan`, `/amcl_pose` (or `/camera/color_image` etc).
