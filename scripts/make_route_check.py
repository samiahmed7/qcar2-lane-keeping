#!/usr/bin/env python3
"""Show where the generated route sits on the road: centre lane vs right lane.

Draws the road (grey ribbon + white edge lines + dashed centre) from the loop +
branch centrelines, then overlays the road-centre route and a right-lane route
(centre offset to the right of travel by a quarter road width). Saves the
right-lane route as university_route_rlane.npy.
"""
import pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

WS = pathlib.Path.home() / "rosbot_ws"
HALF = 0.5            # half road width (m)  -> road is 1.0 m wide
LANE = 0.25          # right-lane centre = quarter width right of road centre


def _unit_tangent(xy):
    tx, ty = np.gradient(xy[0]), np.gradient(xy[1])
    n = np.hypot(tx, ty)
    n[n == 0] = 1.0
    return tx / n, ty / n


def offset(xy, d):
    """+d = LEFT of travel, -d = RIGHT of travel (left normal = (-ty, tx))."""
    tx, ty = _unit_tangent(xy)
    return np.vstack([xy[0] - d * ty, xy[1] + d * tx])


fig, ax = plt.subplots(figsize=(8, 9))
ax.set_facecolor("#bfe3bf")
for fname in ("university_loop.npy", "branch_in.npy"):
    c = np.load(WS / fname)
    left, right = offset(c, HALF), offset(c, -HALF)
    ax.fill(np.concatenate([left[0], right[0][::-1]]),
            np.concatenate([left[1], right[1][::-1]]), color="#555555", zorder=1)
    ax.plot(left[0], left[1], "-", color="white", lw=1.5, zorder=2)
    ax.plot(right[0], right[1], "-", color="white", lw=1.5, zorder=2)
    ax.plot(c[0], c[1], "--", color="white", lw=1.0, dashes=(6, 6), zorder=2)

route = np.load(WS / "university_route.npy")
rlane = offset(route, -LANE)                       # right of travel
np.save(WS / "university_route_rlane.npy", rlane)

ax.plot(route[0], route[1], "-", color="orange", lw=2.0, zorder=3,
        label="road-centre route (current)")
ax.plot(rlane[0], rlane[1], "-", color="magenta", lw=2.2, zorder=3,
        label="RIGHT-lane route")
ax.plot(route[0, 0], route[1, 0], "ko", ms=8, zorder=4)
ax.annotate("START", (route[0, 0], route[1, 0]), xytext=(8, 8),
            textcoords="offset points")
ax.set_aspect("equal")
ax.grid(True, alpha=0.3)
ax.legend(loc="upper left")
ax.set_title("route on the road — centre vs right lane")
ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
fig.tight_layout()
fig.savefig(WS / "route_check.png", dpi=110)
print("saved route_check.png + university_route_rlane.npy")
