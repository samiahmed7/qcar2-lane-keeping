"""Iterative MPC for an Ackermann vehicle.

Vendored from github.com/mcarfagno/mpc_python (cvxpy_mpc/cvxpy_mpc.py), MIT
licensed. State = [x, y, v, heading]; control = [acceleration, steering_angle].
Tracks a world-frame reference path (along/cross-track error) with a soft
obstacle half-plane constraint, solved as a QP via CLARABEL with iterative
re-linearization (iMPC).
"""
from __future__ import annotations

import pathlib

import cvxpy as opt
import numpy as np
import numpy.typing as npt
import yaml


class MPC:
    def __init__(
        self,
        config: str | pathlib.Path | dict,
        horizon_time: float | None = None,
        timestep: float | None = None,
        state_cost: list[float] | None = None,
        final_state_cost: list[float] | None = None,
        input_cost: list[float] | None = None,
        input_rate_cost: list[float] | None = None,
        slack_penalty: float | None = None,
    ) -> None:
        if isinstance(config, (str, pathlib.Path)):
            with open(pathlib.Path(config)) as f:
                config_data = yaml.safe_load(f)
        else:
            config_data = config

        vehicle_config = config_data["model"]["vehicle"]
        obstacle_config = config_data["controller"]["obstacle"]
        prediction_config = config_data["controller"]["prediction"]
        weights_config = config_data["controller"]["weights"]

        self._state_dim: int = 4
        self._control_dim: int = 2

        self.wheelbase: float = vehicle_config["wheelbase"]
        self.width: float = vehicle_config["width"]
        self.max_speed: float = vehicle_config["max_speed"]
        self.max_acc: float = vehicle_config["max_acc"]
        self.max_d_acc: float = vehicle_config["max_d_acc"]
        self.max_steer: float = vehicle_config["max_steer"]
        self.max_d_steer: float = vehicle_config["max_d_steer"]

        self.dt: float = (
            timestep if timestep is not None else prediction_config["timestep"]
        )
        horizon = (
            horizon_time
            if horizon_time is not None
            else prediction_config["horizon_time"]
        )
        self.control_horizon: int = int(horizon / self.dt)

        state_cost_weights = (
            state_cost if state_cost is not None else weights_config["state_cost"]
        )
        terminal_cost_weights = (
            final_state_cost
            if final_state_cost is not None
            else weights_config["final_state_cost"]
        )
        input_cost_weights = (
            input_cost if input_cost is not None else weights_config["input_cost"]
        )
        input_rate_cost_weights = (
            input_rate_cost
            if input_rate_cost is not None
            else weights_config["input_rate_cost"]
        )

        self.q_matrix = np.diag(state_cost_weights)
        self.qf_matrix = np.diag(terminal_cost_weights)
        self.r_matrix = np.diag(input_cost_weights)
        self.rr_matrix = np.diag(input_rate_cost_weights)

        self._safety_margin: float = obstacle_config["safety_margin"]
        self._slack_penalty: float = (
            slack_penalty
            if slack_penalty is not None
            else obstacle_config["slack_penalty"]
        )
        self._vehicle_buffer: float = self.width / 2.0 + self._safety_margin

        self._states = opt.Variable(
            (self._state_dim, self.control_horizon + 1), name="states"
        )
        self._controls = opt.Variable(
            (self._control_dim, self.control_horizon), name="actions"
        )

        self._initial_state = opt.Parameter(self._state_dim, name="x0")
        self._last_command = opt.Parameter(self._control_dim, name="last_applied_command")

        self._A_params = [
            opt.Parameter((self._state_dim, self._state_dim), name=f"A_{k}")
            for k in range(self.control_horizon)
        ]
        self._B_params = [
            opt.Parameter((self._state_dim, self._control_dim), name=f"B_{k}")
            for k in range(self.control_horizon)
        ]
        self._C_params = [
            opt.Parameter(self._state_dim, name=f"C_{k}")
            for k in range(self.control_horizon)
        ]

        self._cos_reference = opt.Parameter(self.control_horizon + 1)
        self._sin_reference = opt.Parameter(self.control_horizon + 1)
        self._along_reference = opt.Parameter(self.control_horizon + 1)
        self._cross_reference = opt.Parameter(self.control_horizon + 1)
        self._velocity_reference = opt.Parameter(self.control_horizon + 1)
        self._heading_reference = opt.Parameter(self.control_horizon + 1)

        self._obstacle_normal_x = opt.Parameter(self.control_horizon, name="obs_nx")
        self._obstacle_normal_y = opt.Parameter(self.control_horizon, name="obs_ny")
        self._obstacle_safe_distance = opt.Parameter(self.control_horizon, name="obs_dist")
        self._obstacle_slack = opt.Variable(
            self.control_horizon, nonneg=True, name="obstacle_slacks"
        )

        self._previous_command = None
        self._previous_trajectory = None

        self._problem = self._make_mpc_problem()

    def _compute_linear_model_matrices(self, x_bar, u_bar):
        v = x_bar[2]
        theta = x_bar[3]
        a = u_bar[0]
        delta = u_bar[1]

        ct = np.cos(theta)
        st = np.sin(theta)
        cd = np.cos(delta)
        td = np.tan(delta)

        A = np.zeros((self._state_dim, self._state_dim))
        A[0, 2] = ct
        A[0, 3] = -v * st
        A[1, 2] = st
        A[1, 3] = v * ct
        # d(psi_dot)/dv = tan(delta)/L. (Upstream had v*tan(delta)/L, an extra
        # factor of v; verified wrong by finite-difference. C still made the
        # operating-point prediction exact, but the velocity->yaw sensitivity the
        # optimizer used was ~v-times too small. This is the correct Jacobian.)
        A[3, 2] = td / self.wheelbase
        A_lin = np.eye(self._state_dim) + self.dt * A

        B = np.zeros((self._state_dim, self._control_dim))
        B[2, 0] = 1
        B[3, 1] = v / (self.wheelbase * cd**2)
        B_lin = self.dt * B

        f_xu = np.array([v * ct, v * st, a, v * td / self.wheelbase]).reshape(
            self._state_dim, 1
        )
        C_lin = (
            self.dt
            * (
                f_xu
                - np.dot(A, x_bar.reshape(self._state_dim, 1))
                - np.dot(B, u_bar.reshape(self._control_dim, 1))
            ).flatten()
        )
        return A_lin, B_lin, C_lin

    def _make_mpc_problem(self):
        cost = 0
        constraints = []

        for k in range(self.control_horizon):
            constraints += [
                self._states[:, k + 1]
                == self._A_params[k] @ self._states[:, k]
                + self._B_params[k] @ self._controls[:, k]
                + self._C_params[k]
            ]

            along_track_error = (
                self._cos_reference[k] * self._states[0, k]
                + self._sin_reference[k] * self._states[1, k]
                - self._along_reference[k]
            )
            cross_track_error = (
                -self._sin_reference[k] * self._states[0, k]
                + self._cos_reference[k] * self._states[1, k]
                - self._cross_reference[k]
            )
            error = opt.vstack(
                [
                    along_track_error,
                    cross_track_error,
                    self._states[2, k] - self._velocity_reference[k],
                    self._states[3, k] - self._heading_reference[k],
                ]
            )
            cost += opt.quad_form(error, self.q_matrix)

            constraints += [
                self._obstacle_normal_x[k] * self._states[0, k + 1]
                + self._obstacle_normal_y[k] * self._states[1, k + 1]
                >= self._obstacle_safe_distance[k] - self._obstacle_slack[k]
            ]
            cost += self._slack_penalty * self._obstacle_slack[k]

            cost += opt.quad_form(self._controls[:, k], self.r_matrix)

            if k == 0:
                cost += opt.quad_form(
                    self._controls[:, 0] - self._last_command, self.rr_matrix
                )
            else:
                cost += opt.quad_form(
                    self._controls[:, k] - self._controls[:, k - 1], self.rr_matrix
                )

        terminal_along = (
            self._cos_reference[-1] * self._states[0, -1]
            + self._sin_reference[-1] * self._states[1, -1]
            - self._along_reference[-1]
        )
        terminal_cross = (
            -self._sin_reference[-1] * self._states[0, -1]
            + self._cos_reference[-1] * self._states[1, -1]
            - self._cross_reference[-1]
        )
        terminal_error = opt.vstack(
            [
                terminal_along,
                terminal_cross,
                self._states[2, -1] - self._velocity_reference[-1],
                self._states[3, -1] - self._heading_reference[-1],
            ]
        )
        cost += opt.quad_form(terminal_error, self.qf_matrix)

        constraints += [self._states[:, 0] == self._initial_state]
        constraints += [opt.abs(self._states[2, :]) <= self.max_speed]
        constraints += [opt.abs(self._controls[0, :]) <= self.max_acc]
        constraints += [opt.abs(self._controls[1, :]) <= self.max_steer]
        constraints += [
            opt.abs(self._controls[0, 0] - self._last_command[0]) / self.dt
            <= self.max_d_acc
        ]
        constraints += [
            opt.abs(self._controls[1, 0] - self._last_command[1]) / self.dt
            <= self.max_d_steer
        ]
        for k in range(1, self.control_horizon):
            constraints += [
                opt.abs(self._controls[0, k] - self._controls[0, k - 1]) / self.dt
                <= self.max_d_acc
            ]
            constraints += [
                opt.abs(self._controls[1, k] - self._controls[1, k - 1]) / self.dt
                <= self.max_d_steer
            ]

        return opt.Problem(opt.Minimize(cost), constraints)

    def solve(self, initial_state, target, verbose=False, max_iter=3,
              tolerance=1e-2, obstacle=None):
        assert len(initial_state) == self._state_dim
        assert target.shape == (self._state_dim, self.control_horizon + 1)

        (obstacle_x, obstacle_y, obstacle_radius,
         obstacle_velocity_x, obstacle_velocity_y) = (
            obstacle if obstacle is not None else (None, None, None, None, None)
        )

        self._initial_state.value = np.array(initial_state)
        self._last_command.value = (
            self._previous_command[:, 0]
            if self._previous_command is not None
            else np.zeros(self._control_dim)
        )

        x_ref, y_ref = target[0, :], target[1, :]
        v_ref, theta_ref = target[2, :], target[3, :]

        cos_values = np.cos(theta_ref)
        sin_values = np.sin(theta_ref)
        self._cos_reference.value = cos_values
        self._sin_reference.value = sin_values
        self._along_reference.value = cos_values * x_ref + sin_values * y_ref
        self._cross_reference.value = -sin_values * x_ref + cos_values * y_ref
        self._velocity_reference.value = v_ref
        self._heading_reference.value = theta_ref

        if self._previous_trajectory is not None and self._previous_command is not None:
            x_guess = np.roll(self._previous_trajectory, -1, axis=1)
            x_guess[:, -1] = self._previous_trajectory[:, -1]
            u_guess = np.roll(self._previous_command, -1, axis=1)
            u_guess[:, -1] = self._previous_command[:, -1]
        else:
            x_guess = target
            u_guess = np.zeros((self._control_dim, self.control_horizon))

        for _ in range(max_iter):
            for k in range(self.control_horizon):
                A_k, B_k, C_k = self._compute_linear_model_matrices(
                    x_guess[:, k], u_guess[:, k]
                )
                self._A_params[k].value = A_k
                self._B_params[k].value = B_k
                self._C_params[k].value = C_k

            normals_x = np.zeros(self.control_horizon)
            normals_y = np.zeros(self.control_horizon)
            distances = np.zeros(self.control_horizon)
            for k in range(self.control_horizon):
                if obstacle is None:
                    normals_x[k] = 1.0
                    normals_y[k] = 0.0
                    distances[k] = -1000.0
                else:
                    dx = x_ref[k + 1] - obstacle_x - obstacle_velocity_x * k * self.dt
                    dy = y_ref[k + 1] - obstacle_y - obstacle_velocity_y * k * self.dt
                    dist = np.hypot(dx, dy)
                    dist = dist if dist > 1e-5 else 1e-5
                    nx = dx / dist
                    ny = dy / dist
                    normals_x[k] = nx
                    normals_y[k] = ny
                    distances[k] = nx * obstacle_x + ny * obstacle_y + obstacle_radius

            self._obstacle_normal_x.value = normals_x
            self._obstacle_normal_y.value = normals_y
            self._obstacle_safe_distance.value = distances + self._vehicle_buffer

            # NOTE: the reference rotation (cos/sin parameters * state variables)
            # in the cost makes this problem NOT DPP, so we do not enforce_dpp.
            # cvxpy re-canonicalizes each solve -- fine at our slow control rate.
            self._problem.solve(
                solver=opt.CLARABEL, warm_start=True, verbose=verbose
            )

            if self._states.value is None:
                emergency = np.zeros((self._control_dim, self.control_horizon))
                v = initial_state[2]
                for k in range(self.control_horizon):
                    a = -self.max_acc if v > 0 else 0.0
                    emergency[0, k] = a
                    v = max(0.0, v + a * self.dt)
                self._previous_command = np.copy(emergency)
                return None, self._previous_command

            new_x = np.array(self._states.value)
            new_u = np.array(self._controls.value)
            if np.max(np.abs(new_x - x_guess)) < tolerance:
                break
            x_guess = new_x
            u_guess = new_u

        self._previous_trajectory = np.copy(new_x)
        self._previous_command = np.copy(new_u)
        return self._previous_trajectory, self._previous_command
