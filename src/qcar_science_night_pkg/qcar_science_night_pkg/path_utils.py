import math
import numpy as np


class PathUtils:

    @staticmethod
    def wrap_angle(a):
        return math.atan2(math.sin(a), math.cos(a))

    @staticmethod
    def load_trajectory(path_file):
        data = np.load(path_file)

        if data.ndim != 2 or data.shape[1] < 2:
            raise RuntimeError("Trajectory must be Nx2, Nx3, or Nx4")

        if data.shape[1] == 2:
            data = PathUtils.add_yaw_and_curvature(data)

        elif data.shape[1] == 3:
            xy = data[:, 0:2]
            curvature = PathUtils.compute_curvature(xy)

            data = np.column_stack([
                data[:, 0],
                data[:, 1],
                data[:, 2],
                curvature
            ])

        else:
            data = data[:, 0:4]

        data[:, 2] = np.unwrap(data[:, 2])

        return data.astype(float)

    @staticmethod
    def add_yaw_and_curvature(xy):
        yaw = []

        for i in range(len(xy)):

            if i < len(xy) - 1:
                dx = xy[i + 1, 0] - xy[i, 0]
                dy = xy[i + 1, 1] - xy[i, 1]

            else:
                dx = xy[i, 0] - xy[i - 1, 0]
                dy = xy[i, 1] - xy[i - 1, 1]

            yaw.append(math.atan2(dy, dx))

        yaw = np.unwrap(np.array(yaw))

        curvature = PathUtils.compute_curvature(xy)

        return np.column_stack([
            xy[:, 0],
            xy[:, 1],
            yaw,
            curvature
        ])

    @staticmethod
    def compute_curvature(xy):
        x = xy[:, 0]
        y = xy[:, 1]

        dx = np.gradient(x)
        dy = np.gradient(y)

        ddx = np.gradient(dx)
        ddy = np.gradient(dy)

        denom = np.power(
            dx * dx + dy * dy,
            1.5
        )

        denom[denom < 1e-6] = 1e-6

        curvature = np.abs(
            dx * ddy - dy * ddx
        ) / denom

        return np.nan_to_num(
            curvature,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

    @staticmethod
    def closest_point(pose, trajectory, previous_idx, global_search=False):
        # global_search=True searches the WHOLE trajectory and drops the
        # forward-only clamp below. Needed after a localization jump
        # (cartographer relocalising into the loaded pbstream), where the
        # true index can be anywhere -- including behind previous_idx --
        # and the windowed search structurally cannot reach it.
        best = previous_idx
        best_score = float("inf")

        if global_search:
            start = 0
            end = len(trajectory)
        else:
            start = max(
                0,
                previous_idx - 10
            )

            end = min(
                len(trajectory),
                previous_idx + 120
            )

        for i in range(start, end):
            p = trajectory[i]

            dist = math.hypot(
                pose[0] - p[0],
                pose[1] - p[1]
            )

            yaw_err = abs(
                PathUtils.wrap_angle(
                    pose[2] - p[2]
                )
            )

            score = dist + 0.6 * yaw_err

            if score < best_score:
                best_score = score
                best = i

        if global_search:
            return best

        return max(best, previous_idx)

    @staticmethod
    def step_indices_for_speed(speed, dt, spacing):
        """
        Convert a target speed [m/s] into an equivalent number of
        path-array indices to advance per MPC stage, given the
        waypoint spacing [m] the path was resampled at and the
        controller timestep dt [s].

        speed is expected to already be a magnitude (use abs() at
        call site for reverse mode, since negative/zero speed must
        still advance forward through the index array).
        """
        speed = max(abs(speed), 1e-3)
        spacing = max(spacing, 1e-6)

        return max(1, int(round((speed * dt) / spacing)))

    @staticmethod
    def build_reference(trajectory, idx, horizon, dt, spacing, target_v):
        """
        Build the (x_ref, y_ref, yaw_ref) horizon reference, spacing
        stages by predicted arc length (target_v * dt) rather than by
        raw path-array index, so stage k corresponds to where the
        vehicle should be after k*dt seconds, not k*spacing meters.
        """
        ref = []
        last_idx = len(trajectory) - 1

        step = PathUtils.step_indices_for_speed(target_v, dt, spacing)

        for k in range(horizon):
            j = min(idx + k * step, last_idx)
            p = trajectory[j]

            ref.extend([
                p[0],
                p[1],
                p[2]
            ])

        return np.array(
            ref,
            dtype=float
        )

    @staticmethod
    def target_speed(
            trajectory,
            idx,
            horizon,
            vmax,
            vmin,
            dt=None,
            spacing=None,
            lookahead_speed=None,
            max_decel=None,
            brake_lookahead_m=None,
            curve_kappa_threshold=0.25):
        """
        Look ahead over the horizon for the tightest curvature and
        slow down accordingly. If dt/spacing/lookahead_speed are
        provided, the lookahead is spaced by predicted arc length
        (consistent with build_reference) instead of raw index;
        otherwise falls back to the original index-per-stage behavior.

        If max_decel and brake_lookahead_m (and spacing) are provided,
        this also scans further ahead than `horizon` for the next
        meaningfully curved section and caps the returned speed so the
        vehicle can actually decelerate down to that section's safe
        speed in time -- this is what prevents accelerating hard on a
        short straight that's too brief to brake back down from.
        """
        last_idx = len(trajectory) - 1

        if dt is not None and spacing is not None:
            ref_speed = lookahead_speed if lookahead_speed is not None else vmax
            step = PathUtils.step_indices_for_speed(ref_speed, dt, spacing)
        else:
            step = 1

        curvatures = []

        for k in range(horizon):
            j = min(idx + k * step, last_idx)
            curvatures.append(
                abs(trajectory[j][3])
            )

        kappa = max(curvatures)

        v_curve = vmax / (
            1.0 + 1.0 * kappa
        )

        v_curve = float(
            np.clip(
                v_curve,
                vmin,
                vmax
            )
        )

        if max_decel is None or brake_lookahead_m is None or spacing is None:
            return v_curve

        # Scan beyond the MPC horizon for the next point where curvature
        # exceeds curve_kappa_threshold, up to brake_lookahead_m meters
        # ahead (in real arc length, via spacing). If found, compute the
        # max speed that allows braking from here to there in time:
        #   v_now^2 <= v_target^2 + 2 * max_decel * distance
        max_lookahead_idx = int(brake_lookahead_m / max(spacing, 1e-6))

        next_curve_kappa = None
        dist_to_curve = None

        for n in range(max_lookahead_idx):
            j = min(idx + n, last_idx)
            k_here = abs(trajectory[j][3])

            if k_here >= curve_kappa_threshold:
                next_curve_kappa = k_here
                dist_to_curve = n * spacing
                break

        if next_curve_kappa is None:
            return v_curve

        v_at_curve = vmax / (1.0 + 1.0 * next_curve_kappa)
        v_at_curve = float(np.clip(v_at_curve, vmin, vmax))

        v_brake_limited = math.sqrt(
            max(
                v_at_curve ** 2 + 2.0 * max_decel * dist_to_curve,
                0.0,
            )
        )

        v_final = min(v_curve, v_brake_limited)

        return float(
            np.clip(
                v_final,
                vmin,
                vmax,
            )
        )

    @staticmethod
    def apply_offset(ref, horizon, offset):
        shifted = ref.copy()

        max_k = min(
            horizon,
            len(ref) // 3
        )

        for k in range(max_k):
            yaw = shifted[3 * k + 2]

            nx = -math.sin(yaw)
            ny = math.cos(yaw)

            shifted[3 * k] += offset * nx
            shifted[3 * k + 1] += offset * ny

        return shifted