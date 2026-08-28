# QCar Technical Notes — Pipeline, Nodes, and Debugging History

Narrative reference: key findings, progress, and the full issues/fixes history — the how/why. `MY_README.md` is the short status log. **All runnable commands live in `commands.md`** — including the safe start/stop procedure; the reasoning behind that procedure is in Issue 17 below.

---

## 1. Key Findings

**Pipeline:** map the room once (Cartographer) → save the map → localize live against it → record a trajectory by driving once → process it into a smoothed closed-loop path → `path_mpc` autonomously follows that path, with LiDAR overtaking and depth-camera emergency stop active. Everything runs natively on the QCar's Jetson (Ubuntu, ROS2 Humble), `ROS_DOMAIN_ID=42`, workspace `~/ros2_ws_izhan`.

**Where things live:**

| Path | What |
|---|---|
| `src/qcar2_nodes/` | Hardware layer + Cartographer launch/config (mostly original dev) |
| `src/qcar_science_night_pkg/` | **The package actually used for driving** — all the Python nodes below |
| `src/qcar_lane_pkg/` | Older/alternate approach, not used |
| `utils/` | Standalone trajectory recording/processing scripts, not `ros2 run`-able |
| `track_map_new.*`, `track_run_cartographer_final.npy` | Current map + the path `path_mpc` actually loads |

Node → file mapping isn't always 1:1 — check each package's `setup.py` `entry_points` (e.g. executable `object_detection_node` actually runs `object_detetion.py`, misspelled).

**The nodes:**

| Node | Role |
|---|---|
| `qcar2_hardware` (C++) | Motor/sensor interface. Kill this to force a guaranteed hard stop. |
| `nav2_qcar2_converter` | `/cmd_vel_nav` → `/qcar2_motor_speed_cmd`. Why teleop/MPC publish to `/cmd_vel_nav`, not `/cmd_vel`. |
| `lidar`, `rgbd` | Publish `/scan` and `/camera/{color,depth}_image` |
| `cartographer_node` | Pure-localization mode against the frozen saved map, publishes `map→odom→base_link` TF (no separate EKF node — Cartographer's extrapolator does `odom→base_link` too) |
| `lidar_overtake_node.py` | LiDAR obstacle detection + overtake state machine (below) |
| `depth_emergency_node.py` | Depth-camera proximity e-stop, independent of the LiDAR one. 2-hit debounce to latch, 3-hit to clear, obstacle/emergency clear independently (Issue 7) |
| `path_mpc_node.py` | The driving brain — MPC path-follower (CasADi, kinematic bicycle model, 12.5Hz). Pose via TF lookup, not a topic. Only drives if `motion_enabled` AND `lidar_motion_safe` AND not `depth_emergency` AND not `mission_done` |
| `sound_node.py` | Plays `.wav` clips on `/qcar2/sound_event` — easy to forget in the launch, nothing depends on it |
| `lane_centering_node.py` | Camera-based lane centering, fully implemented and pre-tuned, never actually tested |

**Overtaking logic** — plain rule-based state machine, not ML. `LidarSectorAnalyzer` turns the raw scan into front/left/right box distances; `OvertakeStateMachine` switches states off those numbers, with debounce counters so single noisy readings can't cause flicker:

```
DRIVE ──(obstacle ahead)──► WAIT_FOR_CLEAR ──(left clear + allowed)──► OVERTAKE_LEFT
  ▲                              │                                          │
  │                     (blocked/not allowed,                               │
  │                      just wait)                                  (passed it,
  │                                                                  right side clear)
  └──────────────── RETURN_RIGHT ◄───────────────────────────────────────┘

Any state ──(something dangerously close)──► EMERGENCY_STOP ──(clear again)──► WAIT_FOR_CLEAR
```

Whether an overtake is even attempted also depends on `/allow_overtake` (published by `path_mpc`, curvature-based — allowed on straights, not corners).

Current tunables (`lidar_overtake_node.py` unless noted):

| Variable | Value | Note |
|---|---|---|
| `front_stop_straight_m` | 1.10 | Reaction distance on a straight — must stay equal to `overtake_start_distance` below (Issue 14) |
| `overtake_start_distance` (`LidarSectorAnalyzer(...)` arg) | 1.10 | The *actual* gate for `obstacle_ahead` — defaults to 0.75 if not explicitly passed (Issue 14) |
| `overtake_start_min_distance` | 0.85 | Floor to *start* an overtake — must stay above `emergency_stop_straight_m` (0.70) and below the two distances above, with margin |
| `emergency_stop_straight_m` / `_curve_m` | 0.70 / 0.65 | Full emergency-stop distance |
| `front_stop_curve_m` | 0.45 | Shorter reaction distance on a corner |
| `hard_stop_front_distance` | 0.40 | Always stop below this, no matter what |
| `overtake_offset` | 0.55 | Sideways shift (m) while overtaking |
| `obstacle_confirm_required` / `left_clear_confirm_required` | 2 / 2 | Debounce readings needed (was 1/1 — too twitchy) |
| `overtake_curve_limit` / `overtake_mean_curve_limit` (`path_mpc_node.py`) | 0.35 / 0.25 | Max peak/mean curvature for overtaking to be permitted |
| `loop_end_tolerance_m` (`path_mpc_node.py`) | 0.90 | Loop-closure tolerance — must stay above `overtake_offset` (Issue 15) |

**"fast-config"** — speed-tuning pass (2026-08-25), validated over multiple 3-lap runs. Baseline (table above / this section's un-labeled values) is `v_max=0.60`; this is the deployed, current alternative:

| Variable | File | Value | Baseline |
|---|---|---|---|
| `v_max` | `path_mpc_node.py` | 0.75 | 0.60 |
| `v_curve_min` | `path_mpc_node.py` | 0.60 | 0.35 |
| `w_speed_tracking` | `path_mpc_node.py` | 120.0 | 90.0 |
| `overtake_maneuver_v` | `path_mpc_node.py` | 0.58 | (new — see below) |
| `startup_v_max` | `path_mpc_node.py` | 0.28 | 0.24 |
| `track_error_min_v` | `path_mpc_node.py` | 0.23 | 0.20 |
| `reverse_speed` | `path_mpc_node.py` | -0.12 | -0.10 |
| `end_stop_speed` | `path_mpc_node.py` | 0.07 | 0.06 |

All LiDAR distances (`lidar_overtake_node.py`) and depth-camera thresholds (`depth_emergency_node.py`) are **unchanged from baseline** — see lessons below for why.

Lessons from this pass:
- **Don't scale LiDAR/depth safety distances with `v_max`.** Tried it (v² scaling, matching real stopping-distance physics) and it broke twice: `front_stop_curve_m`/`hard_stop_front_distance` picked up a real wall at a tight corner (front_min 0.48-0.60m) that the original values safely ignored, and separately `front_stop_straight_m`/`overtake_start_distance` picked up a real static object (likely the desk/cabinet near the loop-seam section) at 0.97-1.69m that also used to be safely below threshold. Root cause: actual curve speed is capped by `v_curve_min`/curvature regardless of `v_max` (measured ~0.4-0.44 m/s even at `v_max=0.75`), and physically-required stopping margin even at full `v_max=0.75` is only ~0.35m (`v²/2·max_decel`) — nowhere close to needing the scaled-up distances. The original values already had huge headroom; scaling them just exposed close-by track geometry as false obstacles. Depth-camera thresholds have the same story for a different reason (ROI reads ~0.5-0.6m on open floor from the camera's own mount angle — see Issue-style note in `depth_emergency_node.py`).
- **`v_curve_min` alone stops helping once steering saturates.** Pushed it 0.35→0.54 incrementally; by 0.54 `delta` was pinned at ~93% of `max_steer` (0.58 rad) at the track's tightest corner, and further pushes just widened the gap between an unreachable `target_v` and the actual achieved speed — no real gain, some tracking degradation. `w_speed_tracking` (90→120) is a better lever for the same goal: it lets the MPC trade tracking-error weight for speed specifically where the two are in tension, without loosening straight-line tracking.
- **Unexplained, not reproduced:** one specific test at `v_max=0.75`/`v_curve_min=0.60`/`w_speed_tracking=120` showed a real deceleration failure through the track's tightest curvature (actual `v` stuck at 0.69-0.75 despite `target_v` computing 0.58-0.60, `track_err` up to 0.062m, depth-emergency stops 5-6/lap instead of ~1/lap) — sustained over ~10 seconds, not a transient. The exact same variable values, retested later after a full car power-cycle, ran clean at full speed with no such issue. Never isolated why — noted here in case it recurs.
- `overtake_maneuver_v` is a new variable, consolidating what used to be `target_v = 0.3` hardcoded in three separate places in `apply_state_machine_reference()` (`path_mpc_node.py`). It went stale during the `v_max`/`v_curve_min` scaling (overtaking became disproportionately slow relative to normal driving) purely because the two copies weren't touched. First raised to `0.46`, the very next overtake at the loop seam showed a severe tracking excursion, so it was reverted to `0.3` out of caution — but the same excursion was then reproduced at `0.3` too, clearing this variable. Retested `0.46` in isolation (nothing else changed): one self-recovering excursion in lap 1 (same ~0.45m peak as the `0.3` runs), laps 2-3 clean.
- Once the hardcoded-clip bug below was fixed (making this variable actually control achieved speed), pushed further: `0.55` ran clean; `0.58` ran clean (`track_err` peaked 0.550-0.569m across two separate test sessions, but `v` held steady the whole time, no decline); `0.60`, `0.62`, and `0.70` **all three** produced an identical near-stall in the track's tightest curvature (`v` declining toward ~0.05-0.14 over 100+ indices, steering pegged at `max_steer`, ~0.5m off line) — self-recovered every time so far, but three-for-three above `0.58` is a repeatable failure boundary, not luck. **Settled on `0.58`** — the highest value with no decline signature in any test. Do not push this past `0.58` without first fixing the loop-seam yaw transient below; every test above that value has hit the same wall.
- **Correction — the real reason overtake speed wasn't reaching `overtake_maneuver_v`**: not a control-damping issue (two live "fixes" chasing that theory — a loop-closure heading gate, and slowing `avoidance_alpha`'s ramp rate — both deployed and tested, neither changed the outcome even slightly). The actual cause was a **second, separate hardcoded clip** on the final published `v`, a few hundred lines away from `overtake_maneuver_v`: `if drive_state in [OVERTAKE_LEFT, RETURN_RIGHT, OBSTACLE_SLOW]: v = clip(v, 0.0, 0.3)`. This silently overrode the MPC's output regardless of `target_v`, which is why `v` sat at exactly `0.30` in every single test no matter what else changed. Changed the literal `0.3` to `self.overtake_maneuver_v` — confirmed live, `v` now tracks `target_v=0.46` for the whole maneuver instead of pinning at 0.30. Lesson: when a value won't move no matter what you change, grep for other hardcoded copies of the same number before re-diagnosing the mechanism.
- **Known open issue, unrelated to the above, not caused by this pass:** an overtake triggered right at the loop-closure seam (`idx=0`) still reliably produces a transient tracking excursion (`track_err` up to ~0.5m, `yaw_err` swinging ±20-30°) before self-recovering within a few seconds — this part was real and is still present after the clip fix above, it just no longer suppresses achieved speed during it. Reproduced at multiple `overtake_maneuver_v` values (0.3 and 0.46) and with both a slow and fast `avoidance_alpha` ramp, so it isn't caused by either. Reads as the pre-existing Issue 15 fragility (loop-seam index-reset vs. an active lateral offset), just exposed more often because a real obstacle happens to sit right at that section of track. Worth its own fix (likely in how the reference/closest_idx handles the transition at idx≈0 while `RETURN_RIGHT`/blend-back is still active).

**Why Cartographer, not AMCL:** AMCL converged fine numerically (covariance < 0.02) but drifted ~1.7m over a full lap while still reporting low covariance — confidently wrong. Cartographer's pose-graph optimization pulls earlier estimates back into alignment in a way AMCL's particle filter can't; first recording closed a loop to 2.1cm.

---

## 2. Progress

- ✅ Physical hardware driving, mapping, localization, trajectory recording/processing
- ✅ MPC autonomous driving, LiDAR obstacle avoidance, depth-camera emergency stop
- ✅ Obstacle voice announcement (English, full-stop only, not on a successful overtake)
- ✅ Autonomous overtaking — confirmed working over multiple 3-lap runs (detect → swing out → pass → return to lane). Took fixing Issues 14-16 to get reliable.
- 🔲 Roundabout not captured in the recorded trajectory yet — missed in every recording attempt, not a tooling problem
- 🔲 Lane centering (camera-based) implemented and configured, never tested alongside MPC
- 🔲 Overtake reaction could still be tuned further (distance/timing) — works reliably now, not fully polished
- 🔲 `/motion_enable false` still doesn't reliably stop the car mid-drive (Issue 9, unresolved) — `target_laps` completing or gracefully stopping `qcar2_hardware` (`kill -2`, NOT `-9` — see Issue 17) remain the only reliable stops

---

## 3. Issues & Fixes

### Issue 1 — Duplicate nodes in the launch file
`qcar2_cartographer_original_launch.py` both included `qcar2_launch.py` (which already starts `nav2_qcar2_converter` and `fixed_lidar_frame`) and separately re-declared those same two nodes. Fixed by removing the duplicates.

### Issue 2 — `/cmd_vel` vs `/cmd_vel_nav`
`nav2_qcar2_converter` subscribes to `/cmd_vel_nav`, not the standard `/cmd_vel`. Manual teleop must remap: `--ros-args -r cmd_vel:=cmd_vel_nav`.

### Issue 3 — Every process crashing with `std::bad_alloc`
Root cause: ROS2 domain collision (laptop on Jazzy, car on Humble, both on default `ROS_DOMAIN_ID=0`) corrupted DDS discovery packets. Fixed by isolating the car onto `ROS_DOMAIN_ID=42`.

### Issue 4 — Stale map references
Three different files pointed at three different outdated maps. Remapped from scratch, updated references to point at the fresh map.

### Issue 5 — RViz Map display QoS mismatch
`map_server` publishes with `Durability: Transient Local`; RViz's default Map display QoS didn't match, showing "No map received" even though data was flowing. Fixed by setting Durability Policy manually or adding the display "By topic" (auto-negotiates QoS).

### Issue 6 — Spline smoothing cutting corners
First processed trajectory used `smoothing=0.03`, visibly rounding off real corners (most noticeable at the bottom curve). Dropped to `0.0002` — hugs the raw recorded points much more closely. Trade-off: noisier curvature (2nd derivative amplifies position noise). Fixed separately with `smooth_curvature.py`.

### Issue 7 — `depth_emergency_node` getting permanently stuck
Obstacle and emergency flags shared one "clear" counter — a persistent *mild* obstacle (e.g. a nearby wall) could keep the *strict* emergency stop latched forever, since both needed the same all-clear condition. Fixed by giving each flag an independent clear-counter. Also fixed a `division by zero` crash on malformed/empty depth frames (now skipped instead of crashing into a default "no danger" state).

### Issue 8 — Localization drifting worse with every lap
`qcar2_2d_localization.lua` had `POSE_GRAPH.optimize_every_n_nodes = 0` — Cartographer's pure-localization mode was doing live scan-matching against the frozen map with **no periodic pose-graph correction**, so small drift in feature-poor sections accumulated lap after lap. Fixed by setting it to `20`. Verified with a 5-lap run afterward — tracking error stayed flat instead of climbing.

### Issue 9 — `/motion_enable` topic collision
Publishing `false` to `/motion_enable` mid-drive never stopped the car. Root cause: `lidar_overtake` had its *own* publisher on the same topic, continuously re-asserting its own safety signal — the manual `false` got overwritten within milliseconds. Fixed by moving `lidar_overtake`'s signal to `/lidar_motion_safe`, with `path_mpc` now requiring both signals to drive.

**Still unresolved**: even after this fix, manually publishing `/motion_enable false` mid-drive still hasn't reliably stopped the car in testing. The reliable stop mechanism remains either `target_laps` completing naturally, or gracefully stopping `qcar2_hardware` directly (see Issue 17 — the actual motor interface, but only a catchable signal triggers its cleanup). Worth investigating further — possibly still a callback-group/QoS issue, or something else entirely.

Also repeatedly found **duplicate `lidar_overtake`/`depth_emergency_node` instances** left running from earlier restarts, fighting each other. Always check for duplicates before a test (`commands.md`).

### Issue 10 — Obstacle voice announcement + no audio output
Wanted "Obstacle detected" spoken only on a full stop, not a successful overtake. `lidar_overtake_node.py`'s sound trigger originally fired on `OVERTAKE_LEFT` too — changed to only `EMERGENCY_STOP` and `WAIT_FOR_CLEAR` (both are states where `path_mpc` fully stops; the corner-blocked case specifically lands in `WAIT_FOR_CLEAR`, not `EMERGENCY_STOP`, so both were needed).

Separately, no audio came out at all even though `sound_node` logged successful playback every time. Turned out to be an ALSA mixer level issue on the Jetson (`DSPK1 Audio Channels` and `DSPK1 FIFO Threshold` were at 0%), not a code bug — fix command in `commands.md`. Doesn't persist across reboot — reapply after any power cycle.

Also regenerated `obstacle.wav` in English via `pico2wave` (already installed on the car) — original was in German, backed up as `obstacle_de_backup.wav`.

### Issue 11 — Trajectory recording pitfalls
Closing a recorded loop precisely (needed for the closed-spline fit) is harder than it sounds — eyeballing physical distance back to the start point wasn't accurate enough, several attempts landed 0.5–1.7m short. Built `dist_to_start.py` to solve this with live numeric feedback instead. Also: the roundabout feature on the physical track has not made it into any recording yet across several attempts — likely just missed while driving, not a tooling problem.

### Issue 12 — Environment/workspace issues (early session)
- Hardcoded `/home/nvidia/ros2_ws/...` paths (old, stale workspace) in multiple launch/config files — repointed at `~/ros2_ws_izhan`.
- `.bashrc` was auto-sourcing the wrong workspace (9 duplicate lines) and defaulting `ROS_DOMAIN_ID=1` — silently overrode manual sourcing. Fixed.
- Leftover crashed processes holding the RealSense USB handle (`EACCES`) and LiDAR GPIO — required killing stale processes before relaunch.

### Issue 13 — `/motion_enable true` doing nothing after a completed run

After a run finishes (`target_laps` reached, or forward trajectory complete), `path_mpc_node.py` sets an internal `self.mission_done = True` latch and its `_control_loop` bails out immediately (`if self.mission_done: self.stop(); return`) — **before** it even looks at `/motion_enable`. The log message printed at that point (`"Target laps complete. Stopping. Press motion_enable to restart."`) is misleading: publishing `/motion_enable true` again does nothing while `mission_done` is set. The actual reset is a separate topic, `/mission_restart` (`commands.md`), which resets `mission_done`, `completed_laps`, and MPC state — only *then* does `/motion_enable true` actually start driving again.

### Issue 14 — Overtaking never triggering, even on a straight, even after raising `front_stop_straight_m`

Root cause was one layer deeper than it first looked. `lidar_overtake_node.py` doesn't compute `obstacle_ahead` itself — it delegates to `LidarSectorAnalyzer.analyze()`, which has its *own* internal gate:

```python
obstacle_ahead = (
    front_count >= self.min_front_points
    and front_min > 0.0
    and front_min <= self.overtake_start_distance   # defaults to 0.75
)
```

`lidar_overtake_node.py` was never passing `overtake_start_distance` into the `LidarSectorAnalyzer(...)` constructor, so it silently used the class default of `0.75`. Meanwhile `front_stop_straight_m` only feeds `limit_status_by_path_context`, which can *suppress* `obstacle_ahead` when the car's too far away — it can never make it trigger farther out than the analyzer's own 0.75m ceiling. Raising `front_stop_straight_m` alone did nothing.

Worse: once `overtake_start_min_distance` (the *separate* "far enough to start overtaking" floor) got raised to `0.85` — above the analyzer's unpatched `0.75` ceiling — the two conditions became mutually exclusive: `obstacle_ahead` can only be `True` when `front_min ≤ 0.75`, but starting an overtake requires `front_min ≥ 0.85`. Not a rare timing miss — a permanent logical deadlock. Overtaking could never trigger, at any distance, until fixed.

**Fix:** explicitly pass `overtake_start_distance=` into the `LidarSectorAnalyzer(...)` constructor, kept equal to `front_stop_straight_m` (both `1.10`). Also bumped `front_x_max` (the analyzer's own box geometry) to `1.35` so the box itself doesn't clip before the distance gate does. **If either value gets tuned again, keep `front_stop_straight_m` and the analyzer's `overtake_start_distance` equal, and both above `overtake_start_min_distance` with real margin (≥0.2m)** — easy trap to fall back into.

### Issue 15 — After overtaking near the loop seam, the car "kept going straight" instead of returning to lane

Happened specifically when an overtake was triggered near idx 525–539 (the last ~15 points of the 540-point closed-loop trajectory) — not a rare spot to hit, since it's one of only 3 places on the whole track where the curvature gate even permits overtaking (mean curvature across the full track is ~0.685, only ~22% of the track is straight enough).

Two things compound:

1. `PathUtils.closest_point()` (`path_utils.py`) hard-clamps its search to `min(len(trajectory), previous_idx + 120)` and never returns less than `previous_idx` — so once the car's near the trajectory's last index, `closest_idx` can never advance past it and can never go back down. The only way back to `idx=0` is the explicit loop-closure block in `path_mpc_node.py`'s `_control_loop`.
2. That loop-closure block requires `distance_to_end_physical <= self.loop_end_tolerance_m`, which was `0.12` (12cm) — far tighter than `overtake_offset` (`0.55m`, how far sideways the car shifts during `OVERTAKE_LEFT`). So if the car is still laterally offset from an overtake when it reaches the last index, it fails the 12cm check, the loop never resets, `closest_idx` stays pinned, the reference window degenerates to a single repeated point, and the MPC has nothing left to steer back toward — the car just crawls forward until it happens to drift back within 12cm on its own (not guaranteed to happen quickly, or at all).

**Fix:** raised `loop_end_tolerance_m` from `0.12` to `0.90` — comfortably above `overtake_offset` — so the loop/lap-reset can fire even while the car's still laterally displaced from a maneuver near the seam.

### Issue 16 — A second `qcar2_hardware` launch fails, but leaves a broken duplicate running

`qcar2_hardware` opens the QCar's HIL (hardware-in-the-loop) interface over GPIO, which only one process can hold at a time. Launching a second instance while one's already running (e.g. relaunching it for "one more lap run" in a new terminal, forgetting one's already up) fails to open hardware:

```text
[ERROR] [...] [qcar2_hardware]: hil_open error: A GPIO is in use by another application. ... Try rebooting the target.
```

but the node does **not** exit on this error — it logs it and continues into `"Starting qcar2 loop..."` anyway. So you end up with a second, fully alive `qcar2_hardware` process on the ROS graph that never actually connected to hardware — the same duplicate-node risk as Issue 9, just for the hardware node instead of the safety nodes.

**Fix:** identify both PIDs (two lines instead of one), keep the older one (earlier start time = the one that actually got the GPIO), and kill only the newer duplicate's two PIDs (the `ros2 run` wrapper and the binary it spawned) directly — never `pkill -f qcar2_hardware` here, since that pattern matches both and would kill the working one too. Exact commands in `commands.md`.

**Better: avoid it entirely** — check first (`commands.md`) before ever relaunching `qcar2_hardware`.

### Issue 17 — `kill -9` on `qcar2_hardware` while driving does NOT stop the car (caused a spin-in-circles, then later a straight-line collision)

Previously documented (wrongly) as the "guaranteed hard stop." In practice, killing the live, HIL-connected `qcar2_hardware` with `SIGKILL` while the car was actively moving produced two real incidents in testing: once the car spun in circles, once it drove straight ahead until it physically bumped into something.

**Root cause**, found in `qcar2_hardware.cpp`: the only code that ever tells the physical HIL board to stop is the `~QCar2()` destructor (lines ~255-290) — it explicitly writes `0.0` to both the steering (`channel 1000`) and throttle (`channel 11000`) channels via `hil_write_other()` before calling `hil_close()`. That destructor only runs when the process shuts down through a path it can catch and unwind from — `rclcpp`'s signal handler turns `SIGINT`/`SIGTERM` into `rclcpp::shutdown()`, which makes `executor.spin()` in `main()` return, so `qcars_node` goes out of scope and its destructor runs. `SIGKILL` (`-9`) is delivered by the kernel directly to the process and can never be caught, blocked, or unwound from — the destructor simply never runs. The Quanser HIL board itself has no deadman-switch/watchdog that auto-zeros on its own (`hil_watchdog_clear()` is called once at startup, disabling/clearing it, not arming an auto-stop) — it just keeps outputting whatever speed/steering value the 15ms `speed_controller()` timer last wrote, forever, until something explicitly writes zero or the board loses power.

**Fix:** use `kill -2` (`SIGINT`) on `qcar2_hardware`'s PID, never `-9`, and confirm it actually exited (PID gone from `ps`, log shows `qcar2 exit`) before treating the car as stopped. Exact commands in `commands.md`. If a graceful stop ever hangs and doesn't exit within a couple seconds, don't wait on more remote commands — cut physical power to the car immediately.

### Other gotchas worth remembering

- **`/tmp` gets wiped on every car reboot** — any helper launch scripts placed there need recreating.
- **Editing files on the car via SSH does not show up in this git repo** until explicitly `scp`/`rsync`'d down — easy to forget and end up with GitLab out of sync with what's actually running.
- **`ros2 topic pub --once`** without an active subscriber match can silently do nothing if timing is off — always verify with `ros2 topic echo` after, don't assume success from the publisher's own output alone.
- **`cd ~/ros2_ws_izhan` is NOT required for `ros2 run`/`ros2 launch`/`ros2 topic ...`** — those resolve through the sourced environment, not the current directory. **It IS required before `colcon build`** — there are 8 separate workspace directories under `~` on this machine, several containing their own copy of `qcar_science_night_pkg`, so building from the wrong place fails with `Duplicate package names not supported`.
- The full-reset `pkill` pattern's `qcar2_launch` doesn't match the actual script name `qcar2_cartographer_launch.py` — the parent `ros2 launch` process can survive a "full reset" unless `ros2 launch qcar2_nodes` is also in the pattern (already fixed in `commands.md`).
