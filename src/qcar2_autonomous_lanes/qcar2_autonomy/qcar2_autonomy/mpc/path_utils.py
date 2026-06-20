"""World-frame reference-path helpers for the vendored MPC.

The MPC in cvxpy_mpc.py expects the reference as a WORLD-frame array
[x, y, v, heading] of shape (4, horizon+1). (The upstream get_ref_trajectory is
vehicle-frame and does NOT match this MPC, so we provide a matching world-frame
sampler here.) compute_path_from_wp is reused from the upstream repo.
"""
import numpy as np
from scipy.interpolate import interp1d


def compute_path_from_wp(start_xp, start_yp, step=0.1, heading_window_m=0.35):
    """Dense reference path [x, y, theta] (3, N) from sparse waypoints."""
    final_xp = []
    final_yp = []
    delta = step
    for idx in range(len(start_xp) - 1):
        section_len = np.sum(
            np.sqrt(
                np.power(np.diff(start_xp[idx : idx + 2]), 2)
                + np.power(np.diff(start_yp[idx : idx + 2]), 2)
            )
        )
        n = max(2, int(np.floor(section_len / delta)))
        interp_range = np.linspace(0, 1, n)
        fx = interp1d(np.linspace(0, 1, 2), start_xp[idx : idx + 2], kind="linear")
        fy = interp1d(np.linspace(0, 1, 2), start_yp[idx : idx + 2], kind="linear")
        final_xp = np.append(final_xp, fx(interp_range)[1:])
        final_yp = np.append(final_yp, fy(interp_range)[1:])
    theta = _local_tangent_heading(final_xp, final_yp, step, heading_window_m)
    return np.vstack((final_xp, final_yp, theta))


def _local_tangent_heading(x, y, step, window_m):
    """Heading from a local chord instead of adjacent samples.

    The waypoints stay exactly where they are; only the heading reference is
    smoothed. This removes small stair-step heading changes that make the MPC
    hunt left/right on otherwise smooth curves.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 2:
        return np.zeros_like(x)
    half_window = max(1, int(round(float(window_m) / max(float(step), 1e-6))))
    theta = np.zeros_like(x)
    for i in range(x.size):
        i0 = max(0, i - half_window)
        i1 = min(x.size - 1, i + half_window)
        if i1 == i0:
            i0 = max(0, i - 1)
            i1 = min(x.size - 1, i + 1)
        dx = x[i1] - x[i0]
        dy = y[i1] - y[i0]
        if abs(dx) < 1e-9 and abs(dy) < 1e-9:
            if i > 0:
                theta[i] = theta[i - 1]
            else:
                theta[i] = 0.0
        else:
            theta[i] = np.arctan2(dy, dx)
    return np.unwrap(theta)


def get_nn_idx(state, path):
    """Index of the path waypoint closest to the vehicle (state = [x,y,v,yaw])."""
    dx = state[0] - path[0, :]
    dy = state[1] - path[1, :]
    return int(np.argmin(np.hypot(dx, dy)))


def get_ref_trajectory(state, path, target_v, horizon, dt, dl=0.05, loop=True):
    """WORLD-frame reference [x, y, v, theta] of shape (4, horizon+1).

    Samples the path ahead of the nearest point by arc length (target_v*dt per
    step). For a closed-loop track, indices wrap around the path.
    """
    n_steps = int(horizon / dt) + 1
    xref = np.zeros((4, n_steps))
    ncourse = path.shape[1]
    ind = get_nn_idx(state, path)

    def _sample(i):
        j = (i % ncourse) if loop else min(i, ncourse - 1)
        return path[0, j], path[1, j], path[2, j]

    x0, y0, th0 = _sample(ind)
    xref[0, 0], xref[1, 0], xref[2, 0], xref[3, 0] = x0, y0, target_v, th0

    travel = 0.0
    for i in range(1, n_steps):
        travel += abs(target_v) * dt
        dind = int(round(travel / dl))
        x, y, th = _sample(ind + dind)
        xref[0, i], xref[1, i], xref[2, i], xref[3, i] = x, y, target_v, th

    xref[3, :] = np.unwrap(xref[3, :])
    while xref[3, 0] - state[3] > np.pi:
        xref[3, :] -= 2 * np.pi
    while xref[3, 0] - state[3] < -np.pi:
        xref[3, :] += 2 * np.pi
    return xref
