# QCar 2 Session Log

Narrative session log for QCar 2 work, in the same spirit as the ROSbot 3
repo's `memory.md`. Complements, rather than replaces, the existing docs:

- **`notes.md`** — the numbered troubleshooting log (Issues 1-17). Specific,
  reproducible failures and their fixes. Read it before changing overtake
  or safety thresholds; several values there are hard-won and reverting
  them re-breaks documented bugs.
- **`commands.md`** / **`MY_README.md`** / **`DEVELOPER_README.md`** —
  how to actually run things.

---

# V2V Follow/Overtake Optimization + Localization Recovery (2026-08-27/28)

Two long hardware sessions driving QCar 2 autonomously alongside ROSbot 3.
The ROSbot-side story lives in that repo's `memory.md`.

### Workspace strategy: `ros2_ws_sami`, with `ros2_ws_izhan` as fallback

All of this work was done in a **separate `~/ros2_ws_sami` workspace**,
created by copying `ros2_ws_izhan`'s `src/`, so the known-good build stayed
untouched and instantly available if anything went wrong. That decision
paid off repeatedly.

Deliberate detail: map, pbstream and reference trajectory still point at
`ros2_ws_izhan`'s copies, so the two workspaces are an identical-input
comparison — only code and thresholds differ. Only the *log* outputs were
repointed into `ros2_ws_sami` so izhan's logs could never be clobbered.

**Deployment gotcha that cost time twice:** `lidar_overtake` and
`v2v_receiver` are started once in `launch_stack()` and are **not**
restarted by the script's relaunch (`l`). A `colcon build` alone does
nothing to the running process — they must be killed and relaunched
(`pkill -2 -f "qcar_science_night_pkg/<node>"`, then `ros2 run ...`), which
can be done live without disrupting driving. Non-interactive SSH also needs
`source install/setup.bash` in addition to `/opt/ros/humble/setup.bash`;
`.bashrc` only covers real interactive logins.

### Following: three separate bugs, found in sequence

The goal was for QCar 2 to hold a steady distance behind ROSbot 3.

1. **The speed cap was discarded entirely while following.** `path_mpc`
   applies the caps, then calls `apply_state_machine_reference()`, which
   assigns `target_v = overtake_maneuver_v` outright in `OBSTACLE_SLOW` —
   and `OBSTACLE_SLOW` *is* the state the car sits in while following. The
   V2V cap was computed, published, received, and then silently
   overwritten with a flat 0.60 m/s regardless of gap. Fixed by
   re-applying the caps *after* the state machine, so they are the last
   word on `target_v` (reverse exempted).
2. **The control law could not hold a distance even when applied.** It was
   a gap→speed ramp with no notion of the leader's speed, so it settled
   wherever the ramp happened to cross ROSbot 3's actual speed — about
   1.07 m for a 2.0 m request. It also topped out above `v_max`, so it was
   not even *binding* until the gap was already down to ~1.64 m. Replaced
   with `cap = leader_speed + gain*(gap - target_gap)`: at the target gap
   the cap **is** the leader's speed, so the distance holds. ROSbot 3's
   speed was already on the V2V wire, just unused.
3. **It then jerked.** The raw cap only changes when a UDP packet lands
   (~10 Hz) while the control loop runs faster, so `target_v` stepped on
   nearly every packet. Fixed with an EMA on the QCar side — the same
   pattern already used for `avoidance_offset_filtered` and
   `track_error_filtered` in that file, just missed when the cap was added.
   The `-1.0` "no restriction" sentinel is excluded from the blend.

Final: target gap **0.5 m**, hard-stop floor **0.4 m**. This deliberately
**decouples** the floor from `lidar_overtake`'s `overtake_start_min_distance`
(still 0.85). They were matched on purpose before, but that value cannot
follow the floor down — it must stay clear of `emergency_stop_straight_m`
(0.70) or overtake-commit loses the race to `EMERGENCY_STOP` again, which
is exactly documented Issue 14. Following-tight and overtake-ready are now
two independent distances.

### Overtaking

- **Blocked overtake lane meant a full stop, always.** The `OVERTAKE`
  branch's only escape was `WAIT` (a hard stop), with no regard for the
  lane the car came from — so it stopped dead even when the original lane
  was completely clear. **Demonstrated on hardware by the supervisor**, who
  stood in the overtake lane mid-pass. Now aborts back into the original
  lane when that lane is empty; `WAIT` is reserved for both lanes blocked.
  Deliberately not gated on `min_overtake_steps` or the lead check (an
  abort is an escape, not a finished pass), nor on the 3-tick confirm
  counter — falling through to `WAIT` would trap the car, since `WAIT`
  cannot re-enter `OVERTAKE` while the left lane is blocked.
  Verified offline against that exact scenario; **not yet hardware-tested.**
- **Curve blindness.** `front_stop_curve_m` is 0.45 m, so `obstacle_ahead`
  went false beyond that in curves and the state machine never even
  considered overtaking through a turn. Now `/v2v/gap` can assert
  `obstacle_ahead` where LiDAR cannot — gated to a safe window and to a
  live, on-path link, so it can never trigger the hard-stop path on its
  own. Range is **context-aware**: the wide ceiling applies on CURVE only,
  because on straights it was starting the whole maneuver from ~1.9 m out,
  far beyond LiDAR's own 1.0 m reach.
- **Commit-distance deadlock.** After tightening the follow gap, the car
  could no longer overtake at all — stuck in `WAIT_FOR_CLEAR` with the
  distance counter pinned at zero. Cause: the commit check used LiDAR
  `front_min`, which measures to ROSbot 3's *nearest surface point* and is
  therefore structurally closer than the pose-to-pose V2V gap by roughly a
  vehicle length (0.77 vs 1.05 for the same moment). Tight following held
  `front_min` permanently under the threshold. Now prefers the V2V gap when
  trustworthy, falling back to LiDAR otherwise.
- **Debounce added** to that check — it had been a raw instantaneous
  threshold crossing, unlike `obstacle_ahead`/`left_clear` which are both
  counter-confirmed, so one noisy sample could commit to an overtake.
- **`overtake_offset` widened 0.55 → 0.70 m** after repeated stalls
  alongside a static ROSbot 3. **Not verified against real lane width** —
  confirm visually that the car stays on the drivable surface.

### Localization: cartographer flip-flopping between two hypotheses

The car kept leaving the track, worst on hard curves. From
`mpc_tracking_log.csv`, the pose was jumping between two hypotheses ~4-5 m
apart — origin `(0.12, 0.00, ±180°)` and `(-4.1, -1.0, ~-100°)` — *within a
single cartographer session* (one `Added trajectory` per session, so not a
restart). Two stretches of the track look alike to the LiDAR, and
cartographer kept accepting false global constraint matches.

Why worst on curves: cartographer has **no motion prior at all** here.
`use_odometry = false`, `use_imu_data = false` — the IMU is even remapped
into the node (`/imu:=/qcar2_imu`) and then ignored by the lua. It is pure
scan matching, which is weakest exactly when yaw rate is high.

Fixes: tightened `min_score` 0.70→0.78 and
`global_localization_min_score` 0.75→0.90 (only those two, so a test result
stays attributable); added pose-jump detection in `path_mpc` that stops,
re-arms the localization gate, resets the startup speed ramp and forces a
global path re-search; and gave `closest_point()` a `global_search` mode,
because it previously searched a forward-only window and clamped the index
so it could **never move backwards** to where the car actually was.

**Diagnostic worth reusing:** `mpc_tracking_log.csv` carries
`x,y,yaw,idx,track_error,target_v,v,mean_curvature` per cycle. Scan for
consecutive-row discontinuities. A *frozen* pose (identical for many cycles
while `v > 0`) means TF has gone stale. Note `track_error` stays **low**
through all of it — the car tracks its wrong-but-self-consistent belief
perfectly — so track_error is useless as a fault signal here.

### Pose seeding on relaunch — built, then DISABLED

To restore e-stop→relaunch recovery after suppressing global relocalization,
`path_mpc` persists its last trusted pose and the launcher snapshots it
*before* killing cartographer (necessary: `path_mpc` survives the relaunch
and would otherwise overwrite the good pose with the new origin pose ~3 s
later). `seed_cartographer.py` then does `finish_trajectory` +
`start_trajectory` with `use_initial_pose`.

**It is disabled at the source.** Its first real use put the pose ~180° out:
`SetInitialTrajectoryPose` is relative to the frozen trajectory's own
internal frame, **not** the ROS `map` TF frame that `path_mpc` reads
(confirmed in cartographer's own headers). x/y were close; yaw was wrong.
Deleting the `.seed` file was not enough to stay disabled — the launcher
recreates it from `POSE_FILE` on every relaunch — so the call itself is
stubbed out. Relaunch falls back to normal global relocalization, which is
proven. Verify the pose math (round-trip test while stationary) before
re-enabling.

### Other bugs fixed

- **`v2v_receiver` crashed** and nothing noticed. `rclpy.spin()` raised
  `ExternalShutdownException`, which was not caught (only `KeyboardInterrupt`
  was), so it propagated past a `finally` that unconditionally called
  `rclpy.shutdown()` again → "rcl_shutdown already called" → ugly death.
  Fixed. **Consequence while it was dead:** `path_mpc` has no staleness
  watchdog on `/v2v/follow_speed_cap`, so it simply froze the last cap value
  indefinitely. That watchdog is still missing — worth adding.
- **Resume silently did nothing.** `ros2 topic pub --once /motion_enable`
  can fire before DDS discovery matches the CLI publisher to `path_mpc`'s
  subscriber, dropping the message with no error. Two real Resume presses
  were no-ops. Now published three times over ~1 s (idempotent). The same
  bug very likely exists in `ros2_ws_izhan`'s copy of the script.
- **Camera wedged after `kill -9`** on the RealSense `rgbd` process — same
  class as Issue 17's `qcar2_hardware`. Needed a power cycle. The launcher
  now SIGINTs it and confirms exit before any `-9` sweep.

### BLOCKER: the V2V gap reads ~2x the real distance

`/v2v/gap` reported **1.051 m** against **57 cm** measured by tape. Ruled
out: curvature/arc-length (path is straight there; chord == along-path),
stale recordings, `path_spacing` (arc length is built from real segment
norms, not nominal spacing), and QCar 2's own localization (its dashboard
position matches physical). **Next step is checking ROSbot 3's own reported
position** — that separates its AMCL from the map-frame transform, and the
arithmetic favours AMCL (ROSbot 3 placed ~48 cm too far along the path;
the transform's worst residual is 12.2 cm).

This blocks the supervisor's requested gap-magnitude filtering work, which
is built directly on that number. A filter fed a 2x-wrong distance rejects
enterable gaps and accepts unenterable ones.

Separately: along-path gap is **blind to lateral offset**, so during a pass
it is a poor proximity metric by construction (0.661 m reported vs 19 cm
physical while swerved). Use the LiDAR side sectors for lateral clearance.

### Status

Nothing above is hardware-verified beyond the observations noted. The
localization and follow work was exercised live over many laps; the
overtake-abort fix is offline-verified only. `ros2_ws_izhan` remains
untouched and is the fallback.
