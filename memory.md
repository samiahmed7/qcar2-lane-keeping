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

### Landmark points for a fresh V2V calibration re-check (2026-08-31, in progress)

The current live transform (`frame_tx=1.3814, frame_ty=0.4986,
frame_tyaw=-3.030697`, see the ROSbot 3 repo's `TODO.md` B1 section) was
found this session to have ~22 cm of real-world error (computed gap
`-0.450m`/dashboard `-0.481m` vs. `0.25m` measured with an iPhone Measure
app, both vehicles stopped ~25 cm apart). Re-measuring physical landmark
points to redo the transform, same idea as the abandoned 2026-08-05
two-point method but with points actually recorded this time.

- **P1 = Start Point** (physical marker at/near the recorded trajectory's
  start). QCar 2's live `map -> base_link` TF while parked at P1:
  `x=0.0903, y=-0.0448, yaw=-178.88 deg` (QCar 2's own map frame).
  Cross-checked against `track_run_cartographer_final.npy` (the recorded
  540-point reference path): 3.9 cm from waypoint idx 0
  `(0.0568, -0.0243)`, 2.5 cm from waypoint idx 539 `(0.0866, -0.0203)` —
  the loop-closure point next to idx 0. Both well within noise, so P1 is
  confirmed to sit right at the trajectory's recorded start/end.
- **P2 (superseded first attempt)**: originally placed at
  `x=-4.8037, y=-1.2753, yaw=-87.2 deg`. Diagnosed as a real, converged
  pose (a wiggle test barely moved it) that was simply ~0.68m outside the
  trajectory bbox, so it didn't render on the dashboard
  (`render_track_png()` computes pixel position with no clamping) — not a
  problem in itself, but abandoned in favor of a closer-to-P1, lower
  curvature point (below) once the transform math showed a bad scale
  mismatch using the first P1/P2 pair (see ROSbot 3 repo's `memory.md`).
- **P2 (current)**: QCar 2 manually placed ~2m from P1 along the flat
  straight, 2026-08-31. Live `map -> base_link` TF:
  `x=-1.5794, y=-0.0092, yaw=178.50 deg`. Cross-checked against
  `track_run_cartographer_final.npy`: 16cm from waypoint idx 55/540,
  arc length 1.652m from P1 (idx 0), curvature 0.038 (low — a repeatable,
  easy-to-mark spot, unlike the first P2 attempt which ended up on a
  curve). A larger P1-P2 baseline than the first attempt (1.65m vs. the
  ~0.06m near-miss they'd started with) reduces sensitivity to the
  ~2-5cm pose noise seen throughout this calibration exercise.

  **Getting here required relaunching the QCar 2 stack, with two real
  incidents worth remembering:**
  1. A stray duplicate `lidar_overtake` process survived a `pkill` because
     a chained multi-command Bash tool call had already been moved to
     background by a 120s timeout partway through — the *local* tool call
     stopped, but each already-issued `ssh ... "cmd & disown"` had already
     detached remotely and kept running. Lesson: after any backgrounded
     multi-step remote launch sequence, always re-check `ps aux` on the
     target host before trusting it's the only thing running, and use
     `TaskStop` on the local task promptly to stop further steps in the
     chain from firing.
  2. **The stack was launched without `ROS_DOMAIN_ID=42` exported**
     (defaulted to domain 0) — caught before driving because `ros2 topic
     list` came back nearly empty against the expected domain. **User
     directive: always export `ROS_DOMAIN_ID=42` for QCar 2 commands.**
  3. A closed-loop monitoring script (meant to stop the car at a target
     arc length) crashed on its very first iteration (an f-string
     dict-key bug) *before* reaching its stop logic — `path_mpc` kept
     driving normally the whole time the crash was being diagnosed and
     fixed, so QCar 2 travelled ~6.7m (to trajectory idx 223, a sharp
     curve near a wall/panel) instead of stopping at the intended ~2m
     target. No collision, car and localization were fine (post-incident
     pose matched the trajectory to 4.4cm), but the intended stop point
     was blown through with no live guard in effect. **Lesson: verify a
     monitoring/stop script actually runs to at least one full loop
     iteration in a dry run before trusting it to gate real motion** —
     `/motion_enable` is not self-limiting; only the script's own logic
     was supposed to stop it, and that logic silently never ran.
  User chose to reposition QCar 2 by hand rather than have it driven
  again after this.

  **Both landmark pairs now collected — ready to solve the 2-point rigid
  transform** (rotation from the angle between the two landmark vectors
  in each frame, then translation) and write the result into QCar 2's
  `config/v2v_params.yaml` (`frame_tx/frame_ty/frame_tyaw`):

  | Robot | x | y | yaw |
  |---|---|---|---|
  | P1 — QCar 2 | 0.0903 | -0.0448 | -178.88 deg |
  | P1 — ROSbot 3 | 1.2470 | 0.3109 | -1.80 deg |
  | P2 — QCar 2 | -1.5794 | -0.0092 | 178.50 deg |
  | P2 — ROSbot 3 | *not yet measured at this revised P2* | | |

  **Next step:** place ROSbot 3 at this (revised) physical P2 point, read
  its AMCL pose, then solve the transform.
  (The full P1-P5 landmark table and the 5-point least-squares fit attempt
  — RMS residual 19.8cm, decided not worth writing over the live ICP
  transform — live in the ROSbot 3 repo's `memory.md`, not repeated here.)

### V2V gap was reporting center-to-center, not physical distance (2026-08-31)

`/v2v/gap` (`v2v_receiver_node.py`, `signed_gap_along()`) measures
`base_link`-to-`base_link` distance, but real-world checks (iPhone Measure
app) are bumper-to-bumper. QCar 2's `base_link` sits at its chassis center
(`chassis_length=0.425m` in `qcar.urdf.xacro`, box centered on
`base_link`) and ROSbot 3's does too (`0.197m` collision box, same). Added
`qcar2_overhang_m` (0.2125) + `rosbot3_overhang_m` (0.0985) as new
`v2v_receiver_node.py` params, subtracted from `|gap|` (sign preserved) in
`_process()`. Live test: 25cm physical gap now reads `-0.049` instead of
the old `-0.481` — the ~31cm overhang was exactly the earlier over-report;
the remaining ~20cm gap is the frame-transform's own calibration error
(see the P1-P5 section above), a separate, already-known problem.

### Follow/overtake distance tuning: raised then reverted (2026-08-31)

Tried raising `follow_target_gap_m` (0.5->0.8) and the matched pair
`follow_min_gap_m`/`hard_stop_front_distance` (0.4->0.6) for a larger
safety margin. **Caused a real regression**: the track's typical real
approach gap to ROSbot 3 sits in the 0.50-0.71m range, all below the new
0.8m target, so `compute_follow_speed_cap()`'s proportional term
(`leader_speed + gain*(gap-target)`) stayed permanently pinned near zero
— QCar 2 could never build enough speed to reach the overtake decision at
all, independent of what `lidar_overtake`'s own state machine said.
**Reverted all three back to 0.5/0.4/0.4.**

Also added a proper fix while diagnosing this: `path_mpc_node.py`'s
V2V-cap re-application after `apply_state_machine_reference()` was
unconditional across `OBSTACLE_SLOW`/`OVERTAKE_LEFT`/`RETURN_RIGHT` (now
`LK`/`LC_LEFT`/`LC_RIGHT` after D3) — gated it to only apply during
`LK`, so raising the follow distance again in the future won't be able to
reintroduce this exact deadlock during an active pass. Also found and
removed `"OBSTACLE_SLOW"` as dead code in the same area — nothing has
ever published it; the gate above used to reference it by accident and
was effectively a no-op until repointed at the real state.

### B3 staleness watchdog + D3 state rename (2026-08-31)

Both done, see the ROSbot 3 repo's `TODO.md` (B3, D3 checkboxes) for the
full writeup — `path_mpc_node.py` now falls back to "no restriction" if
`/v2v/follow_speed_cap` goes stale for >1s, and `DRIVE`/`OVERTAKE_LEFT`/
`RETURN_RIGHT` are renamed to `LK`/`LC_LEFT`/`LC_RIGHT` (IDEAM Eqs. 31-33)
across `overtake_state_machine.py`, `path_mpc_node.py`, and
`lidar_overtake_node.py`. No `LP` yet (that's D2, not built).

### depth_emergency_node curve-blindness deadlock — fixed, revertible (2026-08-31)

**Symptom**: QCar 2 stopped dead on a curve even with clear road (LiDAR:
`obs=False`, plenty of room left) because a wall/room-divider panel at an
angle filled enough of the *fixed* 30%-70% ROI to latch
`/depth_emergency_stop=true` (min=0.44m, obs_ratio=0.70). Self-sustaining:
`required_clear_hits=3` needs 3 consecutive clear readings, impossible
while stopped and still facing the same wall.

**Fix 1** (`depth_emergency_node.py`): subscribe to `/allow_overtake`
(already published by `path_mpc`) and narrow the ROI to 42%-58% of frame
width whenever curvature is high. No signal for curve *direction* exists,
so this narrows symmetrically rather than guessing a side. Gated behind
`CURVE_AWARE_ROI_ENABLED = True` at module level — flip to `False` (or
`git revert`) to fully restore old behavior. Verified live: `estop`
cleared, car resumed driving through the exact curve that had been stuck.

**Fix 2, a cold-start variant of the same bug** (`path_mpc_node.py`):
`_control_loop()` returned immediately on `depth_emergency=True`, before
ever reaching the curvature computation that publishes `/allow_overtake`
— so on a fresh `path_mpc` restart sitting *already* in a curve,
`/allow_overtake` never got a fresh reading and stayed at its default
"straight" assumption forever, permanently defeating Fix 1. Moved
`self.publish_overtake_permission()` to run *before* the `depth_emergency`
early-return (safe: `self.closest_idx`/`self.trajectory` are always
initialized in `__init__`, and reusing the last-known `closest_idx` while
stationary is correct since the car hasn't moved).

**A third, deeper bug found but NOT fixed**: `self.closest_idx` resets to
`0` on any plain `path_mpc` restart and only does a narrow windowed search
(`[idx-10, idx+120]` in `PathUtils.closest_point`) — so restarting while
the car is stopped far from trajectory idx 0 (as happened here, idx~503)
can never recover the true position; curvature computed near idx 0 is
wrong (reports "straight" when actually mid-curve). Worked around this
session by physically repositioning QCar 2 near P1 before restarting
`path_mpc`, rather than fixing the startup index-seeding live. Real fix
would be a one-time global search on the first post-launch pose, mirroring
the existing jump-recovery `global_search` mechanism.

### Trajectory reference edited to add wall clearance — ros2_ws_sami only (2026-08-31)

The recorded reference line through the idx~470-530 curve ran too close to
the same wall (real near-miss even with Fix 1/2 above already applied:
`min=0.41m, obs_ratio=1.000, estop_ratio=0.334` — LiDAR agreed,
`front=0.40/right=0.40`). Per explicit instruction, **`ros2_ws_izhan`'s
original `track_run_cartographer_final.npy` was never touched** — copied
to a new file, `~/ros2_ws_sami/track_run_cartographer_final_leftshift.npy`,
with a smooth 12cm leftward shift (raised-cosine taper, idx 445-535,
peak at idx~495) applied via the left-normal `(-sin(yaw), cos(yaw))`, then
yaw+curvature recomputed locally (same method as `PathUtils.
add_yaw_and_curvature`/`compute_curvature`) and patched back into only
that index window — everything outside idx[455,545] is byte-identical to
the original, confirmed via `np.allclose`. Peak curvature in the edited
window rose to 1.93 (track's prior max: 1.66), but `path_mpc`'s
`v_curve = v_max/(1+kappa)` already saturates to its `v_curve_min` floor
for any kappa above ~0.25, so this has zero effect on commanded speed —
only geometry changed, not the speed profile.

Three references updated to point at the new file (must stay in sync so
gap/on_path mean the same thing across nodes): `path_mpc_node.py`'s
`self.forward_trajectory_file`, `config/v2v_params.yaml`'s
`trajectory_file`, `run_qcar2_stack.sh`'s dashboard `--trajectory` arg.
Loop-closure check passed clean after the edit (0.030m, 0.0 deg yaw diff
at idx539->0), confirming the seam wasn't disturbed. Revert: point all
three back at `~/ros2_ws_izhan/track_run_cartographer_final.npy`.

### Overtake commit-margin deadlock — separate from the curve-wall issue (2026-08-31)

With the above fixed, QCar 2 reached a *different* stall: fully stopped
behind a parked ROSbot 3 on a straight, LiDAR confirming the left lane
clear (`L=True`, `left=-1.00`, nothing detected) and `/allow_overtake=true`,
yet `overtake_allowed` stayed `False` forever
(`state=WAIT_FOR_CLEAR`, `motion=False`).

Root cause: `enough_distance_to_overtake` needs
`commit_check_distance >= overtake_start_min_distance (0.85) +
overtake_commit_margin_m (0.10)` = 0.95m. The car's own LiDAR-based
approach/stop logic (`front_stop_straight_m` + braking dynamics —
*independent* of this margin) settled it at `front_min=0.92m`, just 3cm
short. Since `WAIT_FOR_CLEAR` is a hard stop (`motion=False`), the car can
never close that last 3cm on its own — a permanent deadlock distinct from
every other issue fixed today, even though the symptom ("stuck behind
ROSbot 3") looked similar at first glance.

Considered and rejected: raising `follow_target_gap_m` again (a
completely separate subsystem — only feeds `path_mpc`'s speed ramp, no
control over where the LiDAR-based stop actually settles; already burned
once today on this exact parameter, see above).

**Fix**: trimmed `overtake_commit_margin_m` 0.10 -> 0.05 in
`lidar_overtake_node.py`, so the commit floor becomes 0.90 — comfortably
below the observed 0.92 settling point. The margin was added
2026-08-28 specifically to guard against a single noisy sample crossing
the raw 0.85 threshold; the separate debounce counter
(`overtake_distance_confirm_required=3`, untouched) still independently
guards against that, so this only shrinks the *extra* buffer on top of
the debounce, it doesn't remove noise protection entirely. Deployed and
built; not yet hardware-verified against a real overtake completing
(car was driving normally post-deploy, hadn't re-encountered ROSbot 3 at
the boundary distance yet).
