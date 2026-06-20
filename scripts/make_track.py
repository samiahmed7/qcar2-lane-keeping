#!/usr/bin/env python3
"""Track generator for the university course.

Builds lane centrelines from simple 'turtle' moves (straight + arc), in DRIVING
order, saves them as (2, N) waypoint files the MPC stack reads, and renders a
preview of the driven route. Edit the sizes / moves, re-run, look at the PNGs.

    straight(length)            go forward
    arc(radius, angle_deg)      angle>0 = LEFT turn, angle<0 = RIGHT turn
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

STEP = 0.05          # metres between path points
SAVE_SPACING = 0.10  # metres between saved waypoints


def straight(pose, length):
    x, y, h = pose
    n = max(2, int(length / STEP))
    s = np.linspace(0.0, length, n)
    return np.stack([x + s * np.cos(h), y + s * np.sin(h)]), (
        x + length * np.cos(h), y + length * np.sin(h), h)


def arc(pose, radius, angle_deg):
    x, y, h = pose
    angle = np.radians(angle_deg)
    sign = 1.0 if angle_deg >= 0 else -1.0
    cx = x - sign * radius * np.sin(h)
    cy = y + sign * radius * np.cos(h)
    n = max(2, int(abs(radius * angle) / STEP))
    t = np.linspace(0.0, angle, n)
    xs = cx + (x - cx) * np.cos(t) - (y - cy) * np.sin(t)
    ys = cy + (x - cx) * np.sin(t) + (y - cy) * np.cos(t)
    return np.stack([xs, ys]), (xs[-1], ys[-1], h + angle)


def build(pose, moves):
    pts, marks = [], {}
    for kind, *args in moves:
        if kind == "S":
            seg, pose = straight(pose, *args)
        elif kind == "A":
            seg, pose = arc(pose, *args)
        elif kind == "mark":
            marks[args[0]] = (pose[0], pose[1]); continue
        pts.append(seg)
    return np.concatenate(pts, axis=1), pose, marks


def decimate(xy, spacing=SAVE_SPACING):
    k = max(1, int(round(spacing / STEP)))
    return xy[:, ::k]


# --- sizes (metres) ---
L = 4.0   # straight length
R = 1.0   # corner / roundabout radius

# IMPORTANT: /model/qcar2/odometry is SPAWN-RELATIVE (reads 0,0,0 at spawn), so
# the path the car tracks MUST start at (0,0) heading east. We therefore build
# the whole track with START at the origin: branch/route start at (0,0); the
# loop is shifted up so its intersection sits at (2,4). Spawn the car at
# x:=0 y:=0 yaw:=0.

# --- MAIN LOOP, driven counter-clockwise, starting at the intersection (2,4)
# heading east. Closes cleanly so it can be looped forever. ---
loop, _, marks = build((2.0, 4.0, 0.0), [
    ("mark", "intersection"),
    ("S", 2.0),
    ("A", R, 90),                 # bottom-right corner
    ("S", L),
    ("mark", "roundabout"),       # top-right corner
    ("A", R, 90),                 # the roundabout (treated as a left turn)
    ("S", L),
    ("A", R, 90),                 # top-left corner
    ("S", L),
    ("A", R, 90),                 # bottom-left corner
    ("S", 2.0),                   # back to the intersection
])

# --- ENTRY BRANCH: from START up to the intersection (driven once at the
# start). START -> bit straight -> curve -> up into the intersection. ---
branch, _, _ = build((0.0, 0.0, 0.0), [
    ("S", 1.0),
    ("A", R, 90),
    ("S", 3.0),
])
start_pt = (0.0, 0.0)
ix, iy = marks["intersection"]

# --- save the waypoint files the stack reads: shape (2, N) ---
loop_xy = decimate(loop)
branch_xy = decimate(branch)
np.save("university_loop.npy", loop_xy)
np.save("branch_in.npy", branch_xy)

# --- FULL DRIVEN ROUTE the car actually tracks: START -> up the branch ->
# 90 deg RIGHT turn at the intersection -> once around the loop. One continuous
# path; drive it with loop:=false. The TURN_R=0.9 corner stays on the junction
# road, gives a gentle centre route, and leaves the optional right-lane offset
# route inside the QCar2 steering envelope.
TURN_R = 0.9
route, _, _ = build((0.0, 0.0, 0.0), [
    ("S", 1.0), ("A", R, 90), ("S", 3.0 - TURN_R),   # START -> up the branch to the turn
    ("A", TURN_R, -90),                      # 90 deg RIGHT turn onto the loop
    ("S", 2.0 - TURN_R),                     # onto the loop bottom straight
    ("A", R, 90), ("S", L),                  # bottom-right corner + right side
    ("A", R, 90), ("S", L),                  # roundabout corner + top side
    ("A", R, 90), ("S", L),                  # top-left corner + left side
    ("A", R, 90), ("S", 2.0 + TURN_R),       # bottom-left corner + back to join
])
route_xy = decimate(route)
np.save("university_route.npy", route_xy)

# --- draw the driven route ---
fig, ax = plt.subplots(figsize=(8, 8))
ax.plot(loop[0], loop[1], "-", lw=6, color="#9ecae1", solid_capstyle="round",
        label="main loop (drive CCW, repeats)")
ax.plot(branch[0], branch[1], "--", lw=4, color="#74c476",
        label="entry branch (driven once)")

# the actual driven route (branch -> right turn -> loop) + direction arrows
ax.plot(route[0], route[1], "-", lw=2.4, color="magenta", label="driven route")
ri = np.linspace(0, route.shape[1] - 12, 14).astype(int)
ax.quiver(route[0, ri], route[1, ri],
          route[0, ri + 10] - route[0, ri], route[1, ri + 10] - route[1, ri],
          color="magenta", angles="xy", scale_units="xy", scale=1.8, width=0.004)

ax.plot(*start_pt, "ko", ms=10)
ax.annotate("START", start_pt, textcoords="offset points", xytext=(8, 8),
            fontweight="bold")
ax.plot(ix, iy, "rP", ms=16)
ax.annotate("intersection\n(turn onto loop)", (ix, iy),
            textcoords="offset points", xytext=(10, 8), color="red")
rcx, rcy = marks["roundabout"][0] - R, marks["roundabout"][1]
ax.add_patch(plt.Circle((rcx, rcy), 0.55, color="#fdae6b", alpha=0.7))
ax.annotate("roundabout", (rcx, rcy), textcoords="offset points",
            xytext=(10, 10), color="#e6550d")

ax.set_aspect("equal"); ax.grid(True, alpha=0.3); ax.legend(loc="lower right")
ax.set_title("Driven route — START → right turn at intersection → around loop")
ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
fig.tight_layout()
fig.savefig("route_preview.png", dpi=110)

print(f"saved university_loop.npy {loop_xy.shape}, branch_in.npy {branch_xy.shape}")
print(f"saved university_route.npy {route_xy.shape}  (full driven path)")
print(f"saved route_preview.png; START ({start_pt[0]:.1f},{start_pt[1]:.1f}) "
      f"intersection ({ix:.1f},{iy:.1f})")
