# Project Checkpoint — QCar2 Lane-Keeping Sim

**Status:** Two working lane-keeping pipelines side by side.

| Driver | What it is | Where it lives |
|---|---|---|
| `lane_follower` (legacy) | Centroid in BEV + dynamic lane width + EMA + PD on pixel error | `qcar2_line_tracker/lane_follower.py` |
| `lane_filter` (production-style) | IPM warp + sliding-window detection + Clothoid EKF + Stanley controller | `qcar2_line_tracker/lane_filter.py` (+ `ipm.py`, `lane_detector.py`, `clothoid_ekf.py`) |

The `lane_filter` driver is the one being matured for transfer to real QCar2 + multi-vehicle V2V.

---

## Environment

| Item | Value |
|---|---|
| OS | Ubuntu 24.04 on WSL2 |
| ROS distro | ROS2 Jazzy |
| Simulator | Gazebo Harmonic (gz-sim8) |
| Workspace | `~/rosbot_ws` |
| Real-time factor | ~20–25 % (WSL2 software rendering) |

Required env vars before every launch:
```bash
source /opt/ros/jazzy/setup.bash
source ~/rosbot_ws/install/setup.bash
export GZ_SIM_SYSTEM_PLUGIN_PATH=/opt/ros/jazzy/opt/gz_sim_vendor/lib
export GZ_SIM_RESOURCE_PATH=$(ros2 pkg prefix qcar2)/share:$GZ_SIM_RESOURCE_PATH
```

---

## Package layout

```
rosbot_ws/
├── src/
│   ├── qcar2/
│   │   ├── urdf/QCar2.urdf
│   │   ├── meshes/                        5 STL meshes
│   │   ├── launch/
│   │   │   ├── simulation.launch.py       Gazebo + RSP + spawn + bridges (cmd_vel, odom, tf, lidar, clock)
│   │   │   └── rsp.launch.py
│   │   ├── CMakeLists.txt, package.xml
│   │
│   ├── qcar2_worlds/
│   │   ├── worlds/
│   │   │   ├── qcar_oval.sdf              "default" track — LO=0.40, dash period 0.45 m
│   │   │   └── qcar_oval_wide.sdf         wider lanes — LO=0.55, dash period 0.90 m
│   │   ├── scripts/generate_track.py      Parametric track generator (CLI args)
│   │   ├── CMakeLists.txt, package.xml
│   │
│   └── qcar2_line_tracker/
│       ├── qcar2_line_tracker/
│       │   ├── lane_follower.py          ★ Legacy centroid+PD driver
│       │   ├── lane_filter.py            ★ EKF observer/driver (current focus)
│       │   ├── clothoid_ekf.py             Clothoid Extended Kalman Filter (no ROS deps)
│       │   ├── lane_detector.py            Sliding-window line finder for BEV
│       │   ├── ipm.py                      IPM warp/unwarp utility (reads ipm.yaml)
│       │   └── ipm_calibrate.py            Click-to-calibrate tool
│       ├── launch/
│       │   ├── lane_follower.launch.py      Centroid driver (legacy)
│       │   ├── lane_filter.launch.py        EKF observer (drive_enable=false, parallel debug)
│       │   ├── lane_filter_drive.launch.py  EKF observer + driver (drive_enable=true)
│       │   ├── ipm_calibrate.launch.py      Bridge + calibration tool
│       │   └── camera_bridge.launch.py
│       ├── config/ipm.yaml                  IPM calibration (persisted from `src/` so it survives builds)
│       ├── setup.py, setup.cfg, package.xml
│
├── README.md
├── CHECKPOINT.md
└── ISSUES_AND_FIXES.md
```

---

## Major topics

| Topic | Direction | Carries |
|---|---|---|
| `/qcar2/front_camera/image` | Gz → ROS | 640×480 30 Hz camera |
| `/qcar2/lidar/scan` | Gz → ROS | 720-sample 360° lidar |
| `/model/qcar2/cmd_vel` | ROS → Gz | drive command (linear.x, angular.z) |
| `/model/qcar2/odometry` | Gz → ROS | vehicle pose + velocity |
| `/qcar2/debug/image` | published by `lane_follower` | annotated centroid debug |
| `/qcar2/lane_filter/debug` | published by `lane_filter` | composite: camera + BEV + EKF curve |
| `/qcar2/lane_filter/state` | published by `lane_filter` | `[c0, c1, c2, c3, lane_width_obs, age]` |

---

## Run sequence

### A) **Legacy centroid driver** (simple, no calibration needed)

```bash
# Terminal 1 — simulation
ros2 launch qcar2 simulation.launch.py

# Terminal 2 — driver + camera bridge
ros2 launch qcar2_line_tracker lane_follower.launch.py

# Terminal 3 — debug viewer
ros2 run rqt_image_view rqt_image_view   # → /qcar2/debug/image
```

### B) **EKF observer only** (lane_filter watches; centroid drives)

```bash
# Terminal 1 — simulation
ros2 launch qcar2 simulation.launch.py

# Terminal 2 — centroid driver (also brings up camera bridge)
ros2 launch qcar2_line_tracker lane_follower.launch.py

# Terminal 3 — EKF observer in parallel (drive_enable=false)
ros2 run qcar2_line_tracker lane_filter

# Terminal 4 — debug viewer → /qcar2/lane_filter/debug
ros2 run rqt_image_view rqt_image_view
```

### C) **EKF driver** (lane_filter drives directly; current focus)

```bash
# Terminal 1 — simulation
ros2 launch qcar2 simulation.launch.py world:=qcar_oval_wide y:=-6.275

# Terminal 2 — IMPORTANT: do NOT run lane_follower.launch.py here

# Terminal 3 — EKF driver (drive_enable=true)
ros2 launch qcar2_line_tracker lane_filter_drive.launch.py

# Terminal 4 — debug viewer → /qcar2/lane_filter/debug
ros2 run rqt_image_view rqt_image_view
```

---

## IPM calibration

The EKF pipeline requires a one-off perspective-transform calibration.

```bash
# With Gazebo running:
ros2 launch qcar2_line_tracker ipm_calibrate.launch.py
# 's' to capture frame, click 4 points (BL→BR→TR→TL on the lane lines),
# 's' again to save. The YAML is written to both install/ AND src/ so it
# survives `colcon build`.
```

Default settings: `real_width_m = 0.80` (outer-lane-line spacing on the default track), `real_length_m = 1.50`.

For the `qcar_oval_wide` track use `real_width_m:=1.10` instead.

---

## Track generator

The track is procedural — `qcar2_worlds/scripts/generate_track.py` produces an SDF from parameters:

```bash
python3 src/qcar2_worlds/scripts/generate_track.py \
    --lo 0.55 --tw 1.6 \
    --dash-len 0.30 --dash-period 0.90 \
    --world-name qcar_oval_wide \
  > src/qcar2_worlds/worlds/qcar_oval_wide.sdf
colcon build --packages-select qcar2_worlds --base-paths src
```

Available flags: `--s` (straight half-length), `--r` (corner radius), `--tw` (wall-to-wall width), `--lo` (lane-line offset from centreline), `--dash-len`, `--dash-period`, `--n` (corner segments).

---

## EKF pipeline — what each piece does

```
camera_frame ──▶ IPM.warp ──▶ LaneDetector.detect ──▶ centreline measurements
                              (sliding windows, EKF-seeded)        │
                                                                    ▼
            ClothoidEKF.predict   ◀────── /model/qcar2/odometry (v, ω)
                  │
                  ▼
            ClothoidEKF.update ◀────── centreline measurements
                  │
                  ▼
            EKF state [c0, c1, c2, c3]
                  │
                  ▼
            Stanley controller
                  │
                  ▼
            /model/qcar2/cmd_vel       (only if drive_enable=true)
```

### State vector

`x = [c0, c1, c2, c3]ᵀ`, where the lane centreline relative to the vehicle is

```
y(s) = c0 + c1·s + (c2/2)·s² + (c3/6)·s³
```

with `s` = distance ahead of the vehicle (m).

### Sliding-window detection

- Bird's-eye-view white mask (HSV).
- **EKF-seeded**: starting x-positions for the sliding windows come from the EKF's prediction of where the lane lines should be, NOT just "left half / right half of the image". This prevents the detector from splitting a single curved line into two halves on tight corners.
- Returns per-window centres + total pixel counts.

### Confidence-aware measurement logic

A side is "confident" iff:
1. Total tracked pixels ≥ `confident_pixels_per_side` (default 600)
2. At least `min_windows_confident` windows (default 3) recentred
3. The recentred x-positions follow a coherent line (residual std < 30 px)

Five operating modes:

| Mode | Condition | Behaviour |
|---|---|---|
| `BOTH` | Both sides confident, separation plausible | Use real centreline; EMA-update lane width estimate |
| `LEFT_ONLY` | Only left confident | Project left line inward by half-width |
| `RIGHT_ONLY` | Only right confident | Project right line inward by half-width |
| `AMBIG→L` / `AMBIG→R` | Both confident but separation < 50 % expected (same line seen twice) | Pick the stronger side, project inward |
| `COAST` | Neither side confident | EKF predict-only (motion model + last state) |

### Stanley controller

```
δ = k_head · c1                      # align with lane direction
  + atan(k_lat · c0 / max(v, 0.05))  # pull back to lane centre
  + k_ff · κ(s_lookahead)            # anticipate upcoming curvature
```

With output saturation `±0.5 rad` and rate limit `±0.06 rad/frame ≈ ±1.8 rad/s`. In `COAST` mode the controller produces only half its last commanded steering at crawl speed — fail-soft, not fail-hard.

---

## What's done

- Robot description (URDF + Ackermann steering plugin)
- Camera + lidar sensors publishing on Gz side
- ROS↔Gz bridges
- Parametric track generator + two world variants (`qcar_oval`, `qcar_oval_wide`)
- Centroid lane keeper (legacy, simple)
- IPM calibration tool + persistent YAML config
- Sliding-window lane detector with EKF seeding
- Clothoid EKF (4-D state, χ² outlier rejection)
- Confidence-aware measurement fusion + AMBIGUOUS-line handling
- Stanley controller with curvature feed-forward + rate limit + COAST safety

## What's next (planned)

- **Phase A4** — Tune EKF Q/R for dropout recovery, dash gaps, tight curves.
- LiDAR safety filter (emergency stop on front-arc obstacles).
- Behaviour mode interface (`/qcar2/behavior_mode`: DRIVE/STOP/SLOW/CHANGE_LANE).
- Lane-status publisher (`/qcar2/lane_status`).
- Image watchdog (camera-loss detection).
- Multi-vehicle namespacing.
- V2V messaging between leader and follower QCar2.
- Lane-change controller.
