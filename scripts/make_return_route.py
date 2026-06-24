#!/usr/bin/env python3
"""Append an 'other-lane' return-down-the-branch to the recorded loop route.

my_route_clean.npy drives: branch UP (right lane) -> loop -> back to the
T-junction. This script appends a RETURN segment so that after the loop the car
turns down the branch and drives back to the start in the OPPOSITE lane (the
correct lane for the downward direction), instead of stopping at the junction.

Why not a tight U-turn at the T? The two branch lanes are only ~0.5 m apart, so
an in-place U-turn needs ~0.25 m radius; the QCar2's minimum is ~0.47 m. So we
reuse the junction turn (~1 m radius, feasible): finish the loop heading along
the bottom straight, turn onto the branch heading south, descend the branch in
the other lane, and arrive back near the start.

The descent is the branch ASCENT reflected to the other lane and reversed, so it
exactly mirrors the recorded geometry. Output: my_route_return.npy
"""
import numpy as np

LANE_SHIFT = 0.50          # distance between the two lane centres (m)
JUNCTION_END_Y = 3.90      # odom Y above which the ascent is "in the junction"

src = np.load("my_route_clean.npy")            # (2, N), odom frame (spawn 0,-0.25)
xs, ys = src[0], src[1]

# --- 1. isolate the branch ASCENT (start -> just below the junction) ---
# The ascent is the leading run before the path first reaches the junction Y.
asc_end = int(np.argmax(ys > JUNCTION_END_Y))   # first index above the junction
if asc_end < 5:
    asc_end = 52                                 # fallback (measured)
ascent = src[:, :asc_end]

# --- 2. branch centre & right-normal along the ascent (to reflect to other lane) ---
ax, ay = ascent[0], ascent[1]
dx = np.gradient(ax); dy = np.gradient(ay)
n = np.hypot(dx, dy); n[n < 1e-9] = 1e-9
# right normal of travel direction (going UP): (dy, -dx)/|.|
rx, ry = dy / n, -dx / n
# other lane = shift the right-lane ascent to the LEFT by one lane width
other_x = ax - LANE_SHIFT * rx
other_y = ay - LANE_SHIFT * ry
descent = np.vstack([other_x, other_y])[:, ::-1]   # reverse: now descending

# --- 3. connector from loop end (last point of src) to descent start ---
loop_end = src[:, -1]
desc_start = descent[:, 0]
seg_len = np.hypot(*(desc_start - loop_end))
n_con = max(2, int(seg_len / 0.10))
con = np.vstack([
    np.linspace(loop_end[0], desc_start[0], n_con),
    np.linspace(loop_end[1], desc_start[1], n_con),
])

# --- 4. stitch: full loop route + connector + descent ---
route = np.hstack([src, con[:, 1:], descent])

np.save("my_route_return.npy", route)
print(f"ascent end idx={asc_end}, ascent pts={ascent.shape[1]}")
print(f"src loop pts={src.shape[1]}, descent pts={descent.shape[1]}, "
      f"connector pts={con.shape[1]-1}")
print(f"saved my_route_return.npy  ({route.shape[1]} pts)")
print(f"  start=({route[0,0]:.2f},{route[1,0]:.2f})  end=({route[0,-1]:.2f},{route[1,-1]:.2f})")

# --- preview ---
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(7, 9))
ax.plot(src[0], src[1], "-", color="0.6", lw=2, label="loop (up + around)")
ax.plot(descent[0], descent[1], "-", color="crimson", lw=2, label="return (other lane)")
ax.plot(con[0], con[1], "--", color="orange", lw=1.5, label="connector")
ax.plot(*route[:, 0], "gs", ms=12, label="start")
ax.plot(*route[:, -1], "r^", ms=12, label="end")
ax.set_aspect("equal"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
ax.set_title("my_route_return.npy — loop then return down branch (other lane)")
fig.tight_layout(); fig.savefig("my_route_return.png", dpi=130)
print("saved my_route_return.png")
