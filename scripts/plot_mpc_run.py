#!/usr/bin/env python3
"""Plot a full MPC run with ALL controller values.

    python3 scripts/plot_mpc_run.py [run_log.npz] [track_waypoints.npy]

Panels:
  (1) XY map: centreline, driven path, obstacles, predicted-horizon snapshots
  (2) cross-track error vs time (lane-keeping quality)
  (3) speed: commanded v, target v, actual v
  (4) steering omega (commanded) + heading
  (5) mode timeline (LANE_KEEP / OVERTAKE / etc.)
"""
import sys
import pathlib

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ws = pathlib.Path.home() / "rosbot_ws"
log_path = sys.argv[1] if len(sys.argv) > 1 else str(ws / "mpc_run_log.npz")
wp_path = sys.argv[2] if len(sys.argv) > 2 else str(ws / "track_waypoints.npy")
out_png = str(ws / "mpc_run_plot.png")

d = np.load(log_path, allow_pickle=True)
actual = d["actual"]                       # (M,5) t,x,y,v,yaw
ref_snaps = list(d["ref_snaps"])
obstacles = d["obstacles"] if "obstacles" in d else np.zeros((0, 4))
cmd = d["cmd"] if "cmd" in d else np.zeros((0, 3))          # t, cmd_v, cmd_omega
tgt = d["tgt_speed"] if "tgt_speed" in d else np.zeros((0, 2))
modes = list(d["modes"]) if "modes" in d else []
wp = np.load(wp_path)                       # (2,N)

t = actual[:, 0] if actual.size else np.array([])

# cross-track error of actual path vs centreline (nearest-point, signed)
cx, cy = wp[0], wp[1]
xt = []
for _, x, y, *_ in actual:
    j = int(np.argmin((cx - x) ** 2 + (cy - y) ** 2))
    j2 = min(j + 1, cx.size - 1)
    th = np.arctan2(cy[j2] - cy[j], cx[j2] - cx[j])
    xt.append(-np.sin(th) * (x - cx[j]) + np.cos(th) * (y - cy[j]))
xt = np.array(xt)

# ---- load the controller tuning values to display on the plot ----
def _load_params():
    import yaml
    cfg_dir = ws / "src/qcar2_autonomous_lanes/qcar2_autonomy/config"
    lines = []
    try:
        m = yaml.safe_load(open(cfg_dir / "mpc.yaml"))
        c = m["controller"]; v = m["model"]["vehicle"]; w = c["weights"]
        lines += [
            "MPC TUNING (mpc.yaml)",
            f"  horizon = {c['prediction']['horizon_time']}s @ dt={c['prediction']['timestep']}s",
            f"  Q state  = {w['state_cost']}",
            f"            [along, cross, speed, head]",
            f"  R input  = {w['input_cost']}   Rd rate = {w['input_rate_cost']}",
            f"  max_steer={v['max_steer']} rad  max_acc={v['max_acc']}",
            f"  max_speed={v['max_speed']}  wheelbase={v['wheelbase']}",
            f"  obstacle: margin={c['obstacle']['safety_margin']} slack={c['obstacle']['slack_penalty']}",
        ]
        n = yaml.safe_load(open(cfg_dir / "mpc_nodes.yaml"))["mpc_reference_planner_node"]["ros__parameters"]
        lines += [
            "",
            "PLANNER (mpc_nodes.yaml)",
            f"  speeds: normal={n['normal_speed']} change={n['lane_change_speed']} return={n['return_speed']}",
            f"  ramp_in/out={n['ramp_in']}/{n['ramp_out']}  clearance={n['clearance_margin']}",
            f"  trigger_dist={n['trigger_distance']}  lane_half_w={n['current_lane_half_width_m']}",
            f"  prefer_side={n['prefer_side']}  hold_pad={n['hold_pad']}",
        ]
    except Exception as e:  # noqa
        lines = [f"(params unavailable: {e})"]
    return "\n".join(lines)

PARAMS_TXT = _load_params()

fig = plt.figure(figsize=(19, 12))
gs = fig.add_gridspec(4, 2, hspace=0.55, wspace=0.25,
                      height_ratios=[2, 1, 1, 0.9])
axm = fig.add_subplot(gs[:3, 0])     # big XY map on the left
ax_params = fig.add_subplot(gs[3, 0])  # params text under the map
ax_params.axis("off")
ax_params.text(0.0, 1.0, PARAMS_TXT, transform=ax_params.transAxes,
               fontsize=8.5, family="monospace", va="top",
               bbox=dict(facecolor="#f5f5dc", alpha=0.9, boxstyle="round"))
ax_xt = fig.add_subplot(gs[0, 1])
ax_v = fig.add_subplot(gs[1, 1])
ax_w = fig.add_subplot(gs[2, 1])

# ---- (1) XY map ----
axm.plot(cx, cy, "--", color="0.7", lw=2, label="recorded centreline")
if actual.size:
    axm.plot(actual[:, 1], actual[:, 2], "b-", lw=1.6, label="actual driven")
    axm.plot(actual[0, 1], actual[0, 2], "go", ms=11, label="start")
    axm.plot(actual[-1, 1], actual[-1, 2], "rs", ms=10, label="end")
step = max(1, len(ref_snaps) // 15)
for i in range(0, len(ref_snaps), step):
    s = ref_snaps[i]
    axm.plot(s[:, 0], s[:, 1], "-", color="orange", lw=0.7, alpha=0.6)
if ref_snaps:
    axm.plot([], [], "-", color="orange", label="predicted horizons")
# unique obstacle positions
if obstacles.size:
    seen = set()
    for row in obstacles:
        ox, oy, r = row[1], row[2], row[3]
        key = (round(ox, 1), round(oy, 1))
        if key in seen:
            continue
        seen.add(key)
        axm.add_patch(plt.Circle((ox, oy), r, color="red", alpha=0.45))
    axm.plot([], [], "o", color="red", alpha=0.5, label="obstacle (radius)")
axm.set_aspect("equal")
axm.set_xlabel("x [m]"); axm.set_ylabel("y [m]")
axm.set_title("Path tracking (centreline vs driven vs predicted)")
axm.legend(loc="best", fontsize=9); axm.grid(alpha=0.3)

# ---- (2) cross-track ----
if xt.size:
    ax_xt.plot(t, xt * 100, "b-", lw=1.3)
    ax_xt.axhspan(-10, 10, alpha=0.07, color="green")
    ax_xt.axhline(0, color="0.6", lw=0.8)
    rms = float(np.sqrt(np.mean(xt ** 2))) * 100
    mx = float(np.max(np.abs(xt))) * 100
    ax_xt.set_title(f"Cross-track error  (RMS={rms:.0f} cm, max={mx:.0f} cm)")
ax_xt.set_ylabel("cross-track [cm]"); ax_xt.grid(alpha=0.3)

# ---- (3) speed: commanded / target / actual ----
if cmd.size:
    ax_v.plot(cmd[:, 0], cmd[:, 1], "g-", lw=1.2, label="commanded v")
if tgt.size:
    ax_v.plot(tgt[:, 0], tgt[:, 1], "m--", lw=1.2, label="target v")
if actual.size:
    ax_v.plot(actual[:, 0], actual[:, 3], "b-", lw=1.0, alpha=0.6, label="actual v")
ax_v.set_ylabel("speed [m/s]"); ax_v.legend(fontsize=8, loc="upper right")
ax_v.grid(alpha=0.3); ax_v.set_title("Speed (commanded / target / actual)")

# ---- (4) steering + heading ----
if cmd.size:
    ax_w.plot(cmd[:, 0], cmd[:, 2], "r-", lw=1.2, label="commanded omega [rad/s]")
if actual.size:
    ax_w.plot(actual[:, 0], actual[:, 4], "c-", lw=0.9, alpha=0.6, label="heading [rad]")
# shade mode regions
colors = {"LANE_KEEP_RIGHT": None, "LANE_CHANGE_LEFT": "orange",
          "LANE_CHANGE_RIGHT": "yellow", "PASS_OBSTACLE": "red",
          "RETURN_RIGHT": "green"}
if modes and t.size:
    tmax = t[-1]
    for i, (mt, mname) in enumerate(modes):
        mt = float(mt)
        mend = float(modes[i + 1][0]) if i + 1 < len(modes) else tmax
        c = colors.get(mname)
        if c:
            ax_w.axvspan(mt, mend, alpha=0.12, color=c)
ax_w.axhline(0, color="0.6", lw=0.8)
ax_w.set_xlabel("time [s]"); ax_w.set_ylabel("omega / heading")
ax_w.legend(fontsize=8, loc="upper right"); ax_w.grid(alpha=0.3)
mode_str = " ".join(f"{float(mt):.0f}s:{mn}" for mt, mn in modes[:6])
ax_w.set_title(f"Steering + heading  (modes: {mode_str})", fontsize=9)

plt.suptitle("QCar2 MPC Run — full controller view", fontsize=14, fontweight="bold")
plt.savefig(out_png, dpi=115, bbox_inches="tight")
print(f"saved {out_png}")
if xt.size:
    print(f"cross-track RMS={np.sqrt(np.mean(xt**2))*100:.1f} cm  max={np.max(np.abs(xt))*100:.1f} cm")
if modes:
    print("modes:", [(round(float(m),0), n) for m, n in modes])
