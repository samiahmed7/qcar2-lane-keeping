"""Overtake reference planner (Frenet lateral-offset) for the MPC tracker.

Problem: follow a recorded centreline, but when an obstacle sits ON the path,
ride a smooth lateral offset around it and come back. We encode the overtake in
the *reference* the MPC tracks, instead of a half-plane constraint (which is
degenerate when the obstacle lies on the path -> the avoidance normal is ~0).

Math
----
Path P(s) = (x_p, y_p, psi_p) parameterised by arc length s.
Obstacle at world (ox, oy), radius r. Project onto the path:
    i*    = argmin_i ||(x_i,y_i)-(ox,oy)||
    s_obs = s[i*]
    e_obs = -sin(psi_i*)(ox-x_i*) + cos(psi_i*)(oy-y_i*)   (+left)

Clearance d = r + w_car/2 + margin. Pass on side sigma in {+1 left,-1 right}:
    e* = e_obs + sigma*d

Lateral offset profile centred on s_obs (raised-cosine ramps, flat hold):
    ds = s - s_obs
      ds < -(Lh/2+Lin)          : 0
      ramp up over Lin          : e* * 0.5(1-cos(pi*(ds+Lh/2+Lin)/Lin))
      |ds| <= Lh/2              : e*
      ramp down over Lout       : e* * 0.5(1+cos(pi*(ds-Lh/2)/Lout))
      ds >  Lh/2+Lout           : 0
Lh ("hold length") ~ obstacle length along the path so the full offset spans the
whole box. Heading of the shifted path: psi_ref = psi_p + atan(de/ds).

Shifted (world) reference, left-normal n = (-sin psi_p, cos psi_p):
    x_ref = x_p - e*sin psi_p
    y_ref = y_p + e*cos psi_p
"""
import numpy as np


class OvertakePlanner:
    def __init__(self, path, vehicle_width=0.20, clearance_margin=0.18,
                 ramp_in=0.9, ramp_out=0.9, hold_pad=0.25):
        # path: (3, M) = [x; y; psi] dense, ordered by arc length.
        self.xp = np.asarray(path[0], dtype=float)
        self.yp = np.asarray(path[1], dtype=float)
        self.psip = np.asarray(path[2], dtype=float)
        self.M = self.xp.size
        # cumulative arc length
        seg = np.hypot(np.diff(self.xp), np.diff(self.yp))
        self.s = np.concatenate([[0.0], np.cumsum(seg)])
        self.total_len = float(self.s[-1])
        self.dl = float(np.median(seg)) if seg.size else 0.05

        self.w_car = float(vehicle_width)
        self.margin = float(clearance_margin)
        self.ramp_in = float(ramp_in)
        self.ramp_out = float(ramp_out)
        self.hold_pad = float(hold_pad)

    # ---- geometry helpers ----
    def nearest_index(self, x, y):
        return int(np.argmin(np.hypot(self.xp - x, self.yp - y)))

    def nearest_index_near(self, x, y, center_idx, back_m=0.5, ahead_m=2.0, loop=True):
        """Nearest path sample inside a progress window around ``center_idx``."""
        if center_idx is None:
            return self.nearest_index(x, y)

        center_idx = int(center_idx)
        if loop:
            center_idx %= self.M
        else:
            center_idx = int(np.clip(center_idx, 0, self.M - 1))

        dl = max(self.dl, 1e-6)
        back_steps = max(1, int(np.ceil(float(back_m) / dl)))
        ahead_steps = max(1, int(np.ceil(float(ahead_m) / dl)))
        offsets = np.arange(-back_steps, ahead_steps + 1, dtype=int)
        if loop:
            candidates = (center_idx + offsets) % self.M
        else:
            candidates = np.unique(np.clip(center_idx + offsets, 0, self.M - 1))

        d = np.hypot(self.xp[candidates] - x, self.yp[candidates] - y)
        return int(candidates[int(np.argmin(d))])

    def project_obstacle(self, ox, oy):
        """Return (s_obs, e_obs, idx) of the obstacle projected on the path."""
        i = self.nearest_index(ox, oy)
        c, sn = np.cos(self.psip[i]), np.sin(self.psip[i])
        e_obs = -sn * (ox - self.xp[i]) + c * (oy - self.yp[i])
        return self.s[i], float(e_obs), i

    def _idx_at_s(self, s_query, loop):
        if loop and self.total_len > 0:
            s_query = s_query % self.total_len
        return int(np.clip(np.searchsorted(self.s, s_query), 0, self.M - 1))

    # ---- offset profile ----
    def offset_at(self, ds, e_star, half_hold):
        """Lateral offset e(ds) and slope de/ds for one or many ds values."""
        ds = np.asarray(ds, dtype=float)
        e = np.zeros_like(ds)
        de = np.zeros_like(ds)
        Lin, Lout = self.ramp_in, self.ramp_out
        up0 = -(half_hold + Lin)
        up1 = -half_hold
        dn0 = half_hold
        dn1 = half_hold + Lout
        # ramp up
        m = (ds >= up0) & (ds < up1)
        t = (ds[m] - up0) / max(Lin, 1e-6)
        e[m] = e_star * 0.5 * (1 - np.cos(np.pi * t))
        de[m] = e_star * 0.5 * np.pi / max(Lin, 1e-6) * np.sin(np.pi * t)
        # hold
        m = (ds >= up1) & (ds <= dn0)
        e[m] = e_star
        de[m] = 0.0
        # ramp down
        m = (ds > dn0) & (ds <= dn1)
        t = (ds[m] - dn0) / max(Lout, 1e-6)
        e[m] = e_star * 0.5 * (1 + np.cos(np.pi * t))
        de[m] = -e_star * 0.5 * np.pi / max(Lout, 1e-6) * np.sin(np.pi * t)
        return e, de

    def passed_end_s(self, s_obs, obstacle_radius):
        """Arc length after which the overtake is complete (offset back to 0)."""
        half_hold = obstacle_radius + self.hold_pad
        return s_obs + half_hold + self.ramp_out

    def clearance_for(self, obstacle_radius):
        return obstacle_radius + 0.5 * self.w_car + self.margin

    # ---- main: build the MPC reference ----
    def reference(self, state, target_v, horizon_steps, dt, loop=True,
                  overtake=None, start_index=None):
        """World-frame reference [x, y, v, psi] of shape (4, horizon_steps+1).

        overtake: None for plain path-following, else dict with
            s_obs, e_obs, radius, side (+1 left / -1 right).
        """
        n = horizon_steps + 1
        ref = np.zeros((4, n))
        if start_index is None:
            i0 = self.nearest_index(state[0], state[1])
        elif loop:
            i0 = int(start_index) % self.M
        else:
            i0 = int(np.clip(start_index, 0, self.M - 1))
        s0 = self.s[i0]

        if overtake is not None:
            d = self.clearance_for(overtake["radius"])
            e_star = overtake["e_obs"] + overtake["side"] * d
            half_hold = overtake["radius"] + self.hold_pad
            s_obs = overtake["s_obs"]

        travel = 0.0
        for k in range(n):
            s_k = s0 + travel
            idx = self._idx_at_s(s_k, loop)
            xp, yp, psip = self.xp[idx], self.yp[idx], self.psip[idx]
            if overtake is not None:
                ds = s_k - s_obs
                if loop and self.total_len > 0:
                    # shortest signed arc distance to the obstacle
                    ds = (ds + self.total_len / 2) % self.total_len - self.total_len / 2
                e, de = self.offset_at(np.array([ds]), e_star, half_hold)
                e, de = float(e[0]), float(de[0])
                xr = xp - e * np.sin(psip)
                yr = yp + e * np.cos(psip)
                psir = psip + np.arctan2(de, 1.0)
            else:
                xr, yr, psir = xp, yp, psip
            ref[0, k], ref[1, k], ref[2, k], ref[3, k] = xr, yr, target_v, psir
            travel += abs(target_v) * dt

        ref[3, :] = np.unwrap(ref[3, :])
        while ref[3, 0] - state[3] > np.pi:
            ref[3, :] -= 2 * np.pi
        while ref[3, 0] - state[3] < -np.pi:
            ref[3, :] += 2 * np.pi
        return ref
