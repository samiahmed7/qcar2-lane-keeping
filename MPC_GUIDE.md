# QCar2 MPC Overtaking — How It Works & How We Got Here

This is the guide to the cleaned-up, MPC-only autonomy stack: what each file
does, the math behind it, and **how every problem we hit was diagnosed and
fixed**. Read top to bottom.

---

## 1. What the system does (one paragraph)

The car follows a **pre-recorded centreline** (a lap of waypoints) using **Model
Predictive Control (MPC)**. A **LiDAR** node watches for obstacles ahead; when one
sits on the path, a **planner** bends the reference path into a smooth S-curve
around it, and the MPC tracks that shifted path. After passing, the path returns
to the centreline. Steering and speed are computed every cycle by solving a small
optimization problem.

It is **map-based path tracking + LiDAR-triggered MPC overtaking** — not
vision-based lane keeping. (The ML/RF-DETR node is only a debug overlay now.)

---

## 2. The pipeline (3 nodes)

```
/qcar2/lidar/scan ─► mpc_lidar_obstacle_node ─► /mpc/obstacle
                                                     │
/model/qcar2/odometry ──────────────────────────────┤
track_waypoints.npy ─► mpc_reference_planner_node ◄──┘
                          │  builds shifted reference path + picks speed
                          ▼
                   /mpc/reference_path, /mpc/target_speed, /mpc/mode
                          │
/model/qcar2/odometry ─► mpc_drive_node (the MPC) ─► /model/qcar2/cmd_vel
```

| File | Role |
|---|---|
| `mpc/cvxpy_mpc.py` | The MPC solver (the math engine). Bicycle model + cost + constraints, solved with CVXPY/CLARABEL. |
| `mpc/overtake_planner.py` | Turns "obstacle at X" into a smooth lateral-offset reference path (the S-curve). |
| `mpc/path_utils.py` | Builds a dense path from sparse waypoints; samples a look-ahead reference. |
| `mpc_lidar_obstacle_node.py` | LiDAR → obstacle (x, y, radius, distances). Perception only. |
| `mpc_reference_planner_node.py` | Behaviour FSM + builds the reference the MPC tracks. The "brain". |
| `mpc_drive_node.py` | Pure tracker: runs the MPC solve, outputs Twist. The "hands". |
| `mpc_logger_node.py` | Records a run to `.npz` for plotting. |
| `record_path_node.py` | Records a driven lap → `track_waypoints.npy`. |
| `rfdetr_onnx_lane_node.py` | ML lane segmentation — **debug overlay only**, does not drive. |

Run it: `scripts/run_mpc.sh` (after `sim_bringup` + `spawn_box.sh`).
Plot it: `RECORD=1 ./scripts/run_mpc.sh` then `python3 scripts/plot_mpc_run.py`.

---

## 3. The math (what MPC actually computes)

### 3a. Vehicle model — kinematic bicycle
State `z = [x, y, v, ψ]` (position, speed, heading); control `u = [a, δ]`
(acceleration, steering angle). Continuous dynamics:

```
ẋ = v·cos(ψ)
ẏ = v·sin(ψ)
v̇ = a
ψ̇ = v·tan(δ) / L        L = wheelbase = 0.256 m
```

MPC needs a *linear* model each step, so we linearize (first-order Taylor) around
the current guess and discretize with Euler (`dt = 0.2 s`):

```
z_{k+1} ≈ A_k·z_k + B_k·u_k + C_k
```

`A_k = I + dt·∂f/∂z`, `B_k = dt·∂f/∂u`, `C_k` = the affine remainder that makes
the linear step match the true nonlinear step at the operating point.

> **Bug we fixed here:** the inherited code had `∂ψ̇/∂v = v·tan(δ)/L`. The correct
> Jacobian is `tan(δ)/L` (no extra `v`). Verified by finite-difference
> (error 0.41 → 3e-11). The affine `C_k` hid it at the operating point, but the
> optimizer's velocity→yaw sensitivity was wrong. Now correct.

### 3b. The cost function (what "good driving" means)
Over a horizon of `N` steps we minimize:

```
J = Σ_k [ ‖e_k‖²_Q + ‖u_k‖²_R + ‖u_k − u_{k−1}‖²_Rd ]  +  ‖e_N‖²_Qf
```

- `e_k` = tracking error in the **path (Frenet) frame**: `[along-track, cross-track,
  speed, heading]`. Cross-track = sideways distance from the lane → minimizing it
  **is** lane-centering.
- `Q` weights tracking, `R` penalizes control effort, `Rd` penalizes *changes* in
  control (smooth steering). `Qf` = terminal weight.

Current weights (`config/mpc.yaml`):
```
Q  (state)      = [8, 60, 20, 30]   # along, CROSS-TRACK, speed, heading
R  (input)      = [5, 5]            # accel, steer
Rd (input rate) = [20, 20]          # d_accel, d_steer  (smoothness/damping)
```
Cross-track (60) is the dominant term — the car fights hardest to stay centred.

### 3c. Constraints
- dynamics (above) at every step
- `|v| ≤ max_speed`, `|a| ≤ max_acc`, `|δ| ≤ max_steer (0.5 rad)`
- rate limits: `|Δa|/dt ≤ max_d_acc`, `|Δδ|/dt ≤ max_d_steer`

Solved as a Quadratic Program by CLARABEL. "Iterative MPC" = we re-linearize and
re-solve a few times (`max_iter=2`) so the linear model converges onto the real
curved dynamics.

---

## 4. The overtake math (the S-curve)

The MPC only *tracks*; the **planner** decides the geometry. Avoidance is encoded
in the *reference path*, not as a constraint (a half-plane constraint is
degenerate when the obstacle sits exactly on the path — the avoidance direction
becomes 0/0).

Steps (`overtake_planner.py`):
1. **Project obstacle onto path:** find nearest path point → arc-length `s_obs`
   and lateral offset `e_obs`.
2. **Clearance needed:** `d = r_obstacle + w_car/2 + margin`.
3. **Target offset:** `e* = e_obs + side·d` (side = +1 left / −1 right).
4. **Smooth profile** `e(s)` along the path: 0 → ramp up → hold across the box →
   ramp down → 0, using a raised-cosine ramp:
   `e = e*·0.5·(1 − cos(π·t))` over the ramp length.
5. **Shift the path sideways** by `e(s)` (along the path's left-normal), and set
   the reference heading `ψ_ref = ψ_path + atan(de/ds)`.

> **Why ramp length matters (the "90° turn" bug):** peak reference heading =
> `atan(e*·π / (2·ramp_L))`. With `ramp = 0.9 m` and `e* ≈ 0.46 m` that's **39°**
> (a violent turn). We set **`ramp = 2.0 m` → ~20°** (a real car-like lane
> change). Pure config change.

---

## 5. Every problem we hit, and the fix

### Problem 1 — "90-degree snap turn" during avoidance
**Cause:** the lateral shift ramp was too short (0.9 m), so the reference demanded
39° of heading change over a tiny distance.
**Fix:** lengthen ramps to 2.0 m (`mpc_nodes.yaml`). Peak heading → 20°.
**How verified:** `peak_heading = atan(e*·π/(2·L_ramp))`, computed for several
ramp lengths, picked the one giving ~20°.

### Problem 2 — car drove into the obstacle / wrong dodge side
**Cause:** the obstacle sat dead-centre on the path, so the avoidance direction
was numerically degenerate (~0/0), flipping randomly on 1 cm of LiDAR noise.
**Fix:** the planner picks a side with a **deadband** — if the obstacle is within
±deadband of centre it commits to `prefer_side` (default left); otherwise it dodges
away from the obstacle's measured offset, using LiDAR `left_clear`/`right_clear`
when on `auto`.

### Problem 3 — violent, *growing* sine-wave weaving (±110 cm)
This was the hardest. The car oscillated worse and worse around the centreline.
**Cause:** classic **gain-vs-delay instability**. The non-DPP QP takes ~0.5 s to
solve, so every command is ~0.5 s stale. With high tracking gain, the car
overcorrects on old information → oscillation that grows.
**What made it worse (my own mistakes):** I first raised `Q_cross` to 100 (more
gain → 78% steering saturation) and dropped `max_iter` to 1 (under-converged
linearization). Both *added* energy to the oscillation.
**Fix (counter-intuitive):** *lower* the gain, *add damping*:
```
Q_cross  100 → 60     (less aggressive)
Q_head    40 → 30
Rd        10 → 20     (penalize steering CHANGES = damping)
horizon, max_iter:  back to 3.0 s, 2   (low lag, converged)
```
**How verified (the key method):** an **offline forward-simulation that models the
0.5 s control lag** (a 2-tick delay buffer). It predicted RMS 0.9 cm, stable. The
live run then matched: ~2 cm tracking, no weaving. *Lesson: tune against an
offline model with the lag included — not via 10-minute live runs.*

### Problem 4 — reference jumped to the wrong side of the loop
On a closed loop, "nearest path point" can snap to the *return* leg when two legs
pass close. The MPC then yanked toward the wrong leg.
**Fix:** `nearest_index_near()` — search for the nearest point only within a
**progress window** (a little behind to a couple metres ahead of the last index),
so the reference always advances along the leg the car is actually on.

### Problem 5 — sim freezes
**Cause:** Gazebo GUI + CUDA inference + solver all contend for one laptop GPU.
**Fix:** always run **headless** (`headless:=true`); watch via `view_overlay.py`.

---

## 6. Speed limits (measured)

Offline sweep with the lag model, current tuning:

| Speed | Tracking RMS | Status |
|---|---|---|
| 0.25 m/s | ~1 cm | stable (default) |
| 0.40 m/s | ~3 cm | works, looser |
| 0.60 m/s | — | **solver fails** (linearization invalid) |

Higher speed needs the **DPP refactor** (makes the solve ~20 ms instead of 0.5 s,
killing the lag) — not done yet. That's the single highest-value future change.

---

## 7. Known gaps (honest)
- **No "road fully blocked → STOP" state.** With `prefer_side: left` it will try to
  overtake even with no room. Needs an explicit EMERGENCY_STOP.
- **Off-lane obstacles ignored** (only those within `current_lane_half_width=0.38 m`
  of the path trigger avoidance).
- **Depends on odometry** (perfect in sim; a real car needs localization or the
  ML→MPC fusion where vision supplies the live centreline).
- **Walls in `lab_track.sdf` are paint** — no collision, below LiDAR height — so
  there is no physical safety net; good tracking is what keeps the car on-track.
```
