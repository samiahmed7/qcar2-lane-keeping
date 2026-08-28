# Final Notes — `ros2_ws_sami` (IDEAM_sim)

What was implemented in Sami's workspace, what it proves, and where it stops short.

Branch: `samiahmed-v2v-IDEAM_sim` (commits `0a4be90`, `50f56cf`). The branch tree contains
**only** `IDEAM_sim/` — no ROS packages, no hardware code. Despite the workspace name, nothing
here runs on the QCar; it is a standalone Python simulation study.

---

## 1. What it is

A from-scratch, runnable reconstruction of the simulation study in:

> Y. Shu, J. Zhou, F. Zhang, *"Agile Decision-Making and Safety-Critical Motion Planning for
> Emergency Autonomous Vehicles"*, IEEE T-ITS 26(9), pp. 13750–13766, 2025.

Upstream reference: [YimingShu-teay/IDEAM](https://github.com/YimingShu-teay/IDEAM) — Windows-only,
partly commented out, with the aggregation code missing. The work here makes it run on Linux
under modern Python, proves the port is faithful, and reproduces the paper's headline table.

**The headline result: 11 of the 12 columns of Table III reproduce over the full
200-scenario benchmark**, and the simulation is verified *bit-identical* to the authors' own
code, step by step, across all three constraint states of the motion planner.

---

## 2. What was implemented

Three layers, one direction of data flow per timestep:

```
traffic.py ──► groups.py ──► lsgm.py ──► routing.py ──► planner.py ──► dynamics.py
   HDVs        gap groups    decision    lane+intent       MPC          propagate
                                                             ▲
                          cbf.py ────────────────────────────┘
                       ellipse tangent
```

`simulate.py` drives the loop, `metrics.py` records it, `render.py` draws it.

### Package `IDEAM_sim/ideam/` — 16 modules, ~3.1k lines

| File | What it does |
|---|---|
| `config.py` | Every tunable constant from the paper's Tables I–II — vehicle mass, Pacejka tyre coefficients, IDM/PID gains, LSGM thresholds, MPC horizon and weights, CBF gains, ellipse semi-axes, actuator limits. **Single place to change model parameters.** |
| `track.py` | The closed three-lane circuit (Section VI-A) — 200 m straights, four quarter arcs, 3.5 m lanes. Seven concentric paths, lane occupancy test, Frenet re-projection between lanes, and `rebuild(R)` for a different bend radius (extension only). |
| `geometry.py` | Frenet ↔ Cartesian conversions and vehicle footprints for clearance checks. |
| `dynamics.py` | Ego vehicle: Frenet kinematics (Eq. 1–3), chassis model with Pacejka lateral tyre forces (Eq. 4–10), and the `A, B, C` linearisation (Eq. 11) the MPC consumes. |
| `traffic.py` | The fifteen HDVs — IDM longitudinal, PID lateral lane tracking, five per lane. Also computes ego clearance `S_o` and the Fig. 11 braking-leader scenario. |
| `prediction.py` | Constant-acceleration forecast of one HDV over the horizon. Feeds every constraint. |
| `groups.py` | Gap-group construction (Section IV-A) building the six graph nodes `L1 L2 C1 C2 R1 R2`, plus constraint extraction (Section V-B) and `inquire_C_state`. |
| `lsgm.py` | The decision layer (Algorithms 1–2): conditional DFS over the six-node graph with `LevelCheck` pruning, `risk_assessment` (Eq. 20 temporal-gap test), gap magnitude, long/short-term efficiency. |
| `routing.py` | Chosen group → reference lane + lane-change intent (`K`/`L`/`R`) + re-projected ego state. |
| `cbf.py` | Ellipse handling for the DHOCBF (Eq. 25–27) — closest-boundary projection and the tangent half-space that keeps the MPC convex. |
| `planner.py` | The safety-critical MPC. Three constraint states — lane-keeping (Eq. 31), lane-probing (Eq. 32), lane-changing (Eq. 33) — with longitudinal/lateral DCBFs, ellipse DHOCBFs, boundary and actuator constraints, and a three-stage relaxation ladder for infeasibility. |
| `simulate.py`, `metrics.py`, `render.py`, `scenarios.py` | Closed loop; per-step trace + Table III aggregation; ego-centric top-down video matching the paper's; the 200 benchmark scenarios plus a generator for fresh ones. |

### Scripts `IDEAM_sim/scripts/`

| Script | Purpose |
|---|---|
| `run_scenario.py` | One scenario, with options for starting lane, bend radius, emergency mode, no-probing, video and JSON export |
| `run_benchmark.py` | All 200 scenarios in parallel — **resumable** (skips finished scenarios) and failure-isolated (one bad scenario can't abort the sweep) |
| `report_table3.py` | Aggregate a results directory into Table III beside the published values |
| `validate_port.py` | Clone upstream, patch it for Linux, run both implementations, diff step by step |
| `run_both_sweeps.sh` | Queue the No-Probing sweep behind the main one instead of running both at once |

### Data and outputs committed

| Path | Contents |
|---|---|
| `scenarios/ideam200.npz` | The authors' 200 benchmark scenarios — initial states and desired speeds for 15 HDVs, lifted from their `file_save/*_400` pickles. 100 KB instead of several hundred MB. |
| `results/benchmark/` | 200 JSONs — the full 40 s sweep |
| `results/window20/` | 200 JSONs — the 20 s sweep needed for two-window averaging |
| `results/no_probing/` | 75 JSONs — the abandoned ablation sweep, kept as evidence (see §5) |
| `results/scenarios/` | 6 extension runs with video: three starting lanes, three bend radii |
| `results/ideam_demo.*`, `ideam_emergency.*` | The paper's demo scenario and the Fig. 11 emergency scenario, JSON + mp4 |
| `docs/IDEAM_Reconstruction.pdf` | 14-page reconstruction report including the mathematics and constraints section |

---

## 3. The fidelity argument

This is the part that distinguishes the work from a re-implementation.

`validate_port.py` replays a scenario through both this package and a minimally-patched copy of
the upstream repository, and diffs them step by step:

```
compared 120 steps against the reference implementation

numeric fields (max |port - reference|):
  vx 0.000e+00   vy 0.000e+00   x 0.000e+00   y 0.000e+00
  psi 0.000e+00  a  0.000e+00   d 0.000e+00   s_obs 0.000e+00

categorical fields (mismatching steps):
  group 0    C 0    add 0    lane 0

constraint states exercised:
  No Probe 52 steps · Probe 32 steps · constraint 36 steps

RESULT: MATCH
```

Bit-identical — including every discrete decision (target group, lane-change intent, constraint
state, occupied lane).

**The coverage block is the point.** A match only proves what the window actually entered, and the
three constraint states are three different optimisation problems. Scenario 2 exercises all three;
scenario 144 (the default) never enters lane-probing at all. Validate against a scenario with
non-zero `probe_steps` before trusting that branch.

**The solver matters, and this is why Python 3.12 is pinned.** The paper uses ECOS. This is a
closed loop, so a 1e-9 difference in one QP solution compounds over 400 steps and can flip a
lane-change decision, which changes the whole trajectory. CLARABEL is mathematically equivalent
per step but does *not* guarantee the same 200-run averages. ECOS ships no wheel past CPython 3.12.

| Layer | Status |
|---|---|
| Dynamics, traffic, LSGM, MPC | **Bit-identical** to the authors' code, proven per step |
| The 200 scenarios | **Identical** — initial conditions lifted from their pickles |
| Metric *definitions* | **Reconstructed** — their aggregation script is not in the repository |

---

## 4. Results

Full 200-scenario sweep:

| Metric | This run | Paper | Δ |
|---|---|---|---|
| 20 s progress (m) | **265.08** | 265.08 | −0.00 |
| 40 s progress (m) | **525.19** | 525.19 | −0.00 |
| Max progress (m) | **619.61** | 619.60 | +0.01 |
| Avg velocity (m/s) | **12.45** | 12.44 | +0.01 |
| Max velocity (m/s) | **21.32** | 21.32 | +0.00 |
| Avg min S_o (m) | **2.27** | 2.26 | +0.01 |
| Min S_o (m) | **1.14** | 1.14 | +0.00 |
| Max acceleration (m/s²) | **3.00** | 3.00 | +0.00 |
| Avg jerk | **0.22** | 0.22 | −0.00 |
| Avg lane changes | **4.72** | 4.74 | −0.02 |
| Avg acceleration (m/s²) | 1.18 | 1.40 | −0.22 |

Both progress columns match to the precision the paper prints (265.0781 and 525.1896). All four
extrema land on their published values — and extrema are single-scenario quantities, so matching
them to two decimals means *individual trajectories* reproduce, not merely their averages.

### The four aggregation inferences

The reference repo stores raw per-run traces but not the code that turned them into Table III, so
how each column was computed had to be inferred from Section VI-B. Four inferences were non-obvious
and each moves a column by more than the simulation ever does:

- **Two-window averaging.** Avg Vel, Avg Min S_o and Avg Acc average the statistic over the 20 s
  *and* 40 s windows, not the 40 s run alone — matching how the authors' `metrics_save()` writes a
  separate `_200` and `_400` file per run. Established by *prediction, not fitting*: 40 s-only gives
  12.64 and 2.02 against published 12.44 and 2.26, which requires 20 s values of ≈12.24 and ≈2.50;
  measured 20 s values over all 200 scenarios came out at **12.255** and **2.520**. Progress and the
  extrema use the 40 s file only.
- **Avg Min S_o** is the mean across tracks of *each track's minimum* clearance, not the mean of
  per-step clearances. The latter gives ≈6 m against a published 2.26 m, and only the former puts
  points in the 1.5–2.5 m band Fig. 8 plots.
- **Max Vel / Max Prog** are maxima *across* tracks, not averages of per-track maxima.
- **Jerk** is the mean step-to-step change in acceleration, not a true derivative — dividing by
  dt = 0.1 s would put every Table III method 10× above its published 0.08–0.22 band.

Consequence worth carrying forward: **a column that lands off the paper is far more likely to be an
aggregation convention than a divergence in the simulation**, because the simulation itself is
checked directly by `validate_port.py` and is exact.

### The one column that doesn't reproduce

`Avg. Acc.` — 1.18 against a published 1.40. Every natural definition was measured over 40
evenly-spread scenarios and none lands on it; the published value falls *between* them:

| Candidate (two-window) | Value |
|---|---|
| mean \|a\| commanded | 1.168 |
| mean \|a\| realised longitudinal | 1.146 |
| **paper** | **1.40** |
| RMS a commanded | 1.576 |
| mean \|a\| realised total, √(a_lon² + a_lat²) | 1.623 |
| RMS a realised total | 2.358 |

A blend could be contrived to hit 1.40, but that is curve-fitting rather than reproduction, so the
column is left as computed and flagged. Note `Max. Acc.` matches exactly at 3.00, so whatever the
authors used agrees with this one on the peak.

---

## 5. Baselines — one included, one refuted, three left out

**Included: No-Probing IDEAM** (Table III row 5). Not a re-implementation — the authors ship it as
`iMPC_solve_OneStep_for_noadapt` / `inquire_C_state_for_noadapt`. Diffed against the full system it
is exactly two one-line changes: (1) an unearned lane change reports the lane-keeping constraint
state instead of the probing one, and (2) the leader's longitudinal DCBF is gated on constraint
state rather than lane-change intent. Both sit behind `--no-probing`.

**It does not reproduce Table III row 5, and the published code cannot have produced it.** Over the
first 75 scenarios, **67 collide** (`min_so` = 0.00, median 0.00) against a published average
clearance of 2.33 m and worst case 0.84 m; lane changes run at 7.7 against a published 3.75. The
sweep was abandoned at 75/200 rather than spend a further ~11 hours on it, and those runs are kept
in `results/no_probing/` as evidence.

This is not a porting error — `validate_port.py --no-probing` is bit-identical to upstream over the
95 steps it survives, including 63 steps in exactly the suppressed-probing state that causes the
collisions. The mechanism is visible in the code: with probing suppressed the constraint state
becomes `"No Probe"` while lane intent stays `"L"`/`"R"`, so the ego commits to the target lane's
reference path while the `"No Probe"` branch builds **no ellipse constraint against the target
lane's follower** — it merges into a gap with nothing protecting the vehicle behind it.

A sensible ablation is easy to write (suppress the probe but keep the lane change gated on the
follower constraint, so the ego *declines* instead of merging blind) — deliberately not implemented,
because that would be this reconstruction's design decision, not the authors'.

**Not included: SO-DM (LAS 30/60), DRB-FSM, MOBIL.** Genuinely absent from the repository — only the
stripped planner they plug into is present. Re-implementing them from their source papers would
produce numbers that are ours, not the authors'.

---

## 6. Extension studies (marked as extensions, not reproductions)

| Study | Command | Note |
|---|---|---|
| Starting lane | `--start-lane 0\|1\|2` | **Reproduction-safe** — runs on the published track, traffic unchanged, so lane is the only variable |
| Road curvature | `--bend-radius 40 \| 300` | **Not a reproduction.** No published figure applies to a track the paper never used, and `rebuild(105.25)` does not restore the published track bit-for-bit. Any Table III run must use a process that never calls `rebuild()`. |

On curvature: the 200 m straights are held fixed, so the bend always begins at s = 200 m and the
radius sets how long and hard the arc is (63 m of arc at R=40, 471 m at R=300). Over a fixed 40 s
window the comparison is therefore *brief sharp corner* vs *sustained gentle curve*, not the same
corner at different tightness.

---

## 7. Deviations from upstream

**Behaviour-preserving** (all verified bit-identical): Windows absolute paths and `sys.path.append`
hacks replaced with package-relative paths; `requirements.txt` rewritten as UTF-8 and cut to what is
actually imported (dropped `casadi`, `openpyxl`, `GPy`, `opencv-python`, `PyQt5` and the Windows/MKL
pins); ragged `np.array(...)` fixed for NumPy 2.x; dead code removed (RK6 integrator, `Model` base
class, `MPC_solve` and its helpers, the never-enabled Gaussian-process residual path, plotting
helpers with no callers, the `*_for_comparison` variants belonging to absent baselines);
visualisation restored (commented out upstream, and its video writer had hard-coded Windows paths).

**Preserved deliberately, though they look like bugs** — changing any of them would move the
published numbers:

- Left-lane polylines are sampled to 1400 m although that lane is 1439 m round, so the last 39 m of
  the loop has no sample points (`track.py`).
- `_lateral_dmin` keeps the *last* vehicle in the ROI list rather than the tightest bound (`groups.py`).
- The target gap's follower is propagated with the *ego* follower's speed profile (`groups.py`).
- Only a lower bound is imposed on the steering rate, not an upper one (`planner.py`).

**Genuinely new:** if all three feasibility relaxations fail, the simulator coasts on the previous
plan and counts the step (`infeasible_steps`) instead of crashing as the original does — report a
non-zero count as a caveat on that run. Timing is split into `solve_ms` (solver only, comparable to
Fig. 7's 5.97 ms mean) and `step_ms` (solver plus problem construction).

---

## 8. Running it

Python **3.12** required (see §3 for why).

```bash
cd IDEAM_sim && python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
# or, without root:  uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt

# ECOS must be present or the numbers will not match:
.venv/bin/python -c "import cvxpy; print('ECOS' in cvxpy.installed_solvers())"
```

```bash
# one scenario with video — 400 steps at dt=0.1s = 40s simulated, ~7 min
.venv/bin/python scripts/run_scenario.py --scenario 144 --video results/ideam.mp4

# Fig. 11 emergency (left-lane leader sheds speed from t=7s)
.venv/bin/python scripts/run_scenario.py --scenario 120 --emergency --video results/emergency.mp4

# Table III — budget several hours; resumable
.venv/bin/python scripts/run_benchmark.py --workers 7
.venv/bin/python scripts/run_benchmark.py --workers 7 --steps 200 --out-dir results/window20
.venv/bin/python scripts/report_table3.py --window20 results/window20

# prove the port
.venv/bin/python scripts/validate_port.py
```

Solver timings in the committed results are hardware- and load-dependent — the paper used a
2.70 GHz i7-12700H, and running 8 benchmark workers on 8 cores inflates them several-fold.

---

## 9. Relationship to the rest of the project

Nothing in `IDEAM_sim` runs on the QCar. It shares vocabulary with the hardware stack — LSGM,
lane-change decisions, overtaking — but the QCar's overtaking is a plain rule-based state machine
(`notes.md` §1), not a CBF-constrained MPC over gap groups. The two are separate lines of work.

The overlap that could matter later is `IDEAM_sim/ideam/lsgm.py`: the V2X branch's
surrounding-vehicle state pipeline was written to feed an LSGM-shaped decision layer, so this is the
reference implementation of what that layer is supposed to do. Any attempt to port it to hardware
starts by confronting the gap between a 15-vehicle highway at 12 m/s and a single-obstacle indoor
track at 0.6 m/s.
