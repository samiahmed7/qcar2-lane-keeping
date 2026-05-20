#!/usr/bin/env python3
"""
Extended Kalman Filter for the Clothoid lane model.

State vector (in vehicle frame, units in metres / radians):
    x = [c0, c1, c2, c3]^T

Lane centreline (relative to vehicle heading at the origin):
    y(s) = c0 + c1*s + (c2/2)*s² + (c3/6)*s³

Where:
    c0 — lateral offset of lane centre from vehicle (m, +left)
    c1 — lane heading at vehicle (rad, +left)
    c2 — lane curvature (1/m)
    c3 — lane curvature rate (1/m²) — the "clothoid" coefficient

Process model:
    As the vehicle moves forward by Δs = v·dt and rotates by yaw rate ω,
    the lane appears to translate/rotate relative to the vehicle. Using
    Taylor expansion of the cubic at s = Δs:
        c0' = c0 + c1·Δs + (c2/2)·Δs² + (c3/6)·Δs³
        c1' = c1 + c2·Δs + (c3/2)·Δs²  −  ω·dt
        c2' = c2 + c3·Δs
        c3' = c3

Measurement model:
    A detected lane-centreline point (s_i, y_i) in vehicle metres:
        y_i = c0 + c1·s_i + (c2/2)·s_i² + (c3/6)·s_i³ + noise
    so the row of H for that measurement is:
        H_i = [1, s_i, s_i²/2, s_i³/6]

This module has no ROS dependencies — it can be unit-tested standalone.
"""
import numpy as np


class ClothoidEKF:
    DIM = 4

    def __init__(
        self,
        q_diag=(1e-3, 1e-3, 1e-4, 1e-5),
        p_init=(1.0, 0.5, 0.1, 0.01),
        r_meas=0.01,
    ):
        """
        q_diag : process noise covariance diagonal (per second, scaled by dt)
        p_init : initial state covariance diagonal
        r_meas : per-measurement variance (m²) of a lane-point observation
        """
        self.x = np.zeros(self.DIM)
        self.P = np.diag(p_init).astype(np.float64)
        self.Q_per_sec = np.diag(q_diag).astype(np.float64)
        self.R_meas = float(r_meas)
        self.initialized = False
        self.last_update_time = None    # for staleness tracking

    # ────────────────────────────────────────────────────────────
    def reset(self):
        self.x = np.zeros(self.DIM)
        self.P = np.diag(np.diag(self.P) * 0 + 1.0)  # large again
        self.initialized = False
        self.last_update_time = None

    # ────────────────────────────────────────────────────────────
    def predict(self, v: float, omega: float, dt: float):
        """Roll the state forward by dt given vehicle motion."""
        if dt <= 0:
            return
        ds = v * dt
        c0, c1, c2, c3 = self.x

        # Nonlinear state update (cubic Taylor expansion + yaw rotation)
        self.x = np.array([
            c0 + c1*ds + 0.5*c2*ds**2 + (1.0/6.0)*c3*ds**3,
            c1 + c2*ds + 0.5*c3*ds**2 - omega*dt,
            c2 + c3*ds,
            c3,
        ])

        # Jacobian F = ∂f/∂x  (state update is linear in x for fixed ds)
        F = np.array([
            [1.0, ds,  0.5*ds**2, (1.0/6.0)*ds**3],
            [0.0, 1.0, ds,         0.5*ds**2      ],
            [0.0, 0.0, 1.0,        ds              ],
            [0.0, 0.0, 0.0,        1.0             ],
        ])

        # Process noise scaled by dt
        Q = self.Q_per_sec * dt
        self.P = F @ self.P @ F.T + Q

    # ────────────────────────────────────────────────────────────
    def update(self, measurements):
        """
        Apply a batch of measurements.

        measurements : iterable of (s_i, y_i) tuples in vehicle metres.
                       s_i is forward distance, y_i is lateral position
                       (+left) of the lane centreline at that distance.
        """
        if len(measurements) == 0:
            return

        if not self.initialized:
            self._init_from_measurements(measurements)
            return

        # Apply each measurement as a 1-D scalar update (numerically stable
        # and lets us drop NaNs/outliers without rebuilding the whole H)
        I = np.eye(self.DIM)
        for s_i, y_i in measurements:
            H = np.array([1.0, s_i, 0.5*s_i**2, (1.0/6.0)*s_i**3]).reshape(1, self.DIM)
            y_pred = float(H @ self.x)
            innovation = float(y_i) - y_pred
            S = float(H @ self.P @ H.T) + self.R_meas
            # χ² gate at 3σ to reject gross outliers
            if (innovation * innovation) / S > 9.0:
                continue
            K = (self.P @ H.T) / S          # (4,1)
            self.x = self.x + K.flatten() * innovation
            self.P = (I - K @ H) @ self.P

    # ────────────────────────────────────────────────────────────
    def _init_from_measurements(self, measurements):
        """Bootstrap the filter by least-squares cubic fit on first batch."""
        if len(measurements) < self.DIM:
            return
        ss = np.array([m[0] for m in measurements], dtype=np.float64)
        ys = np.array([m[1] for m in measurements], dtype=np.float64)
        # np.polyfit returns highest-order-first: [a3, a2, a1, a0]
        # for y = a3*s³ + a2*s² + a1*s + a0
        a3, a2, a1, a0 = np.polyfit(ss, ys, 3)
        # Map to our [c0, c1, c2, c3]: a2 = c2/2, a3 = c3/6
        self.x = np.array([a0, a1, 2.0 * a2, 6.0 * a3])
        self.initialized = True

    # ────────────────────────────────────────────────────────────
    def evaluate(self, s):
        """Lateral position y(s) of lane centre at forward distance s (m)."""
        c0, c1, c2, c3 = self.x
        return c0 + c1*s + 0.5*c2*s*s + (1.0/6.0)*c3*s*s*s

    def evaluate_heading(self, s):
        """Tangent angle at forward distance s (rad, +left)."""
        c0, c1, c2, c3 = self.x
        return c1 + c2*s + 0.5*c3*s*s

    def evaluate_curvature(self, s):
        """Local curvature κ(s) (1/m)."""
        c0, c1, c2, c3 = self.x
        return c2 + c3*s

    def lateral_offset(self):
        """Lateral offset of lane centre at vehicle (s=0). Equal to c0."""
        return float(self.x[0])

    def heading_error(self):
        """Lane heading at vehicle (s=0). Equal to c1."""
        return float(self.x[1])

    # ────────────────────────────────────────────────────────────
    def covariance_diag(self):
        return np.diag(self.P).copy()

    def state_tuple(self):
        return tuple(float(v) for v in self.x)
