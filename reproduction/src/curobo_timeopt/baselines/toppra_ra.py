"""TOPP-RA literature baseline for the classical second-order TOPP problem.

The adapter deliberately supports only C2={velocity, acceleration, torque}.
Jerk and torque-rate constraints are third-order constraints and must be
benchmarked in a separate comparison gate.

TOPPRA's reachability algorithm is used unchanged. The robot-space torque
coefficients are prepared with this repository's inverse-dynamics model so the
baseline uses exactly the same cuRobo dynamics as the native methods.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from torch import Tensor

from curobo_timeopt.baselines.constant_acceleration_audit import (
    audit_constant_acceleration,
)
from curobo_timeopt.retiming.gpu_dense_validation import (
    DenseFeasibilityResult,
)
from curobo_timeopt.retiming.problem import RetimingProblem
from curobo_timeopt.types import ConstraintMode


class ToppraUnavailableError(ImportError):
    """Raised when the optional TOPPRA dependency is not installed."""


@dataclass(frozen=True)
class ToppraRAConfig:
    """Numerical settings for the external TOPP-RA baseline."""

    grid_samples: int = 1001
    solver_wrapper: str = "seidel"
    interpolation_constraints: bool = True
    safety_scale: float = 0.995

    def __post_init__(self) -> None:
        if self.grid_samples < 3:
            raise ValueError("grid_samples must be at least 3")
        if not self.solver_wrapper:
            raise ValueError("solver_wrapper must be non-empty")
        if not 0.0 < self.safety_scale <= 1.0:
            raise ValueError("safety_scale must be in (0, 1]")


@dataclass(frozen=True)
class ToppraRAResult:
    """TOPP-RA output in the repository's x(s)=s_dot**2 convention."""

    success: bool
    s: Tensor
    x: Tensor
    path_acceleration: Tensor
    duration: float
    total_ms: float
    preparation_ms: float
    solve_ms: float
    solver_wrapper: str
    return_code: str
    diagnostics: dict[str, Any] = field(default_factory=dict)


def _require_toppra():
    try:
        import toppra
        import toppra.constraint as constraint
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ToppraUnavailableError(
            "TOPP-RA requires the optional baseline dependency; "
            "install with python -m pip install -e '.[baselines]'"
        ) from exc
    return toppra, constraint


def _require_c2(problem: RetimingProblem) -> None:
    if problem.mode != ConstraintMode.second_order():
        raise ValueError(
            "TOPP-RA is a C2 baseline and requires exactly "
            "ConstraintMode.second_order()={velocity, acceleration, torque}"
        )
    if problem.dynamics is None:
        raise ValueError("TOPP-RA torque constraints require inverse dynamics")


@torch.no_grad()
def _precompute_torque_terms(
    problem: RetimingProblem,
    grid: Tensor,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return affine TOPP torque terms m, c, g on the supplied path grid."""
    if problem.dynamics is None:
        raise ValueError("torque coefficient extraction requires inverse dynamics")
    q = problem.path.evaluate(grid, 0)
    q_s = problem.path.evaluate(grid, 1)
    q_ss = problem.path.evaluate(grid, 2)
    if q.shape != q_s.shape or q.shape != q_ss.shape:
        raise ValueError("path evaluations must have identical shapes")

    zeros = torch.zeros_like(q)
    probe_q = torch.cat((q, q, q), dim=0)
    probe_qd = torch.cat((zeros, zeros, q_s), dim=0)
    probe_qdd = torch.cat((zeros, q_s, q_ss), dim=0)
    tau = problem.dynamics.inverse_dynamics(probe_q, probe_qd, probe_qdd)
    gravity, unit_acceleration, unit_speed = tau.chunk(3, dim=0)
    m = unit_acceleration - gravity
    c = unit_speed - gravity
    return tuple(value.detach().cpu().double().numpy() for value in (m, c, gravity))


def _profile_duration(grid: np.ndarray, sd: np.ndarray) -> float:
    denominator = sd[:-1] + sd[1:]
    if np.any(~np.isfinite(denominator)) or np.any(denominator <= 0.0):
        return float("inf")
    return float(np.sum(2.0 * np.diff(grid) / denominator))


def solve_toppra_ra(
    problem: RetimingProblem,
    config: ToppraRAConfig | None = None,
) -> ToppraRAResult:
    """Solve the matched C2 retiming problem with TOPP-RA.

    total_ms includes path/constraint preparation, batched inverse-dynamics
    coefficient extraction, TOPPRA problem construction, and the reachability
    solve. Import time and the repository's independent dense audit are excluded.
    """
    _require_c2(problem)
    cfg = config or ToppraRAConfig()
    toppra, constraint = _require_toppra()

    grid = torch.linspace(
        0.0, 1.0, cfg.grid_samples, device=problem.device, dtype=problem.dtype
    )
    grid_np = grid.detach().cpu().double().numpy()

    class TorchGeometricPath(toppra.interpolator.AbstractGeometricPath):
        @property
        def dof(self) -> int:
            return problem.path.dof

        @property
        def path_interval(self) -> np.ndarray:
            return np.array([0.0, 1.0], dtype=float)

        def __call__(self, path_positions, order: int = 0) -> np.ndarray:
            positions = np.asarray(path_positions)
            scalar = positions.ndim == 0
            query = torch.as_tensor(
                np.atleast_1d(positions),
                device=problem.device,
                dtype=problem.dtype,
            )
            values = problem.path.evaluate(query, order)
            result = values.detach().cpu().double().numpy()
            return result[0] if scalar else result

    started = time.perf_counter()
    path = TorchGeometricPath()
    m, c, gravity = _precompute_torque_terms(problem, grid)

    velocity = (
        cfg.safety_scale * problem.limits.velocity.detach().cpu().double().numpy()
    )
    acceleration = (
        cfg.safety_scale * problem.limits.acceleration.detach().cpu().double().numpy()
    )
    torque = cfg.safety_scale * problem.limits.torque.detach().cpu().double().numpy()
    velocity_limits = np.column_stack((-velocity, velocity))
    acceleration_limits = np.column_stack((-acceleration, acceleration))

    discretization = (
        constraint.DiscretizationType.Interpolation
        if cfg.interpolation_constraints
        else constraint.DiscretizationType.Collocation
    )
    velocity_constraint = constraint.JointVelocityConstraint(velocity_limits)
    acceleration_constraint = constraint.JointAccelerationConstraint(
        acceleration_limits,
        discretization_scheme=discretization,
    )

    class PrecomputedTorqueConstraint(constraint.LinearConstraint):
        def __init__(self) -> None:
            super().__init__()
            self.dof = problem.path.dof
            self.identical = True
            self.set_discretization_type(discretization)
            eye = np.eye(self.dof)
            self._f = np.vstack((eye, -eye))
            self._g = np.concatenate((torque, torque))

        def compute_constraint_params(self, path_arg, gridpoints, *args, **kwargs):
            del args, kwargs
            if path_arg.dof != self.dof:
                raise ValueError("TOPPRA torque constraint path dof mismatch")
            if gridpoints.shape != grid_np.shape or not np.allclose(
                gridpoints, grid_np, atol=1e-12, rtol=0.0
            ):
                raise ValueError(
                    "precomputed torque coefficients require the configured TOPP-RA grid"
                )
            if self.discretization_type == constraint.DiscretizationType.Collocation:
                return m, c, gravity, self._f, self._g, None, None
            return constraint.canlinear_colloc_to_interpolate(
                m, c, gravity, self._f, self._g, None, None, gridpoints, identical=True
            )

    torque_constraint = PrecomputedTorqueConstraint()
    algorithm = toppra.algorithm.TOPPRA(
        [velocity_constraint, acceleration_constraint, torque_constraint],
        path,
        gridpoints=grid_np,
        solver_wrapper=cfg.solver_wrapper,
    )
    preparation_ms = (time.perf_counter() - started) * 1e3

    solve_started = time.perf_counter()
    sdd, sd, _ = algorithm.compute_parameterization(
        problem.initial_speed, problem.final_speed
    )
    solve_ms = (time.perf_counter() - solve_started) * 1e3
    total_ms = (time.perf_counter() - started) * 1e3

    return_code = str(getattr(algorithm.problem_data, "return_code", "unknown"))
    if sd is None:
        return ToppraRAResult(
            success=False,
            s=grid,
            x=torch.full_like(grid, torch.nan),
            path_acceleration=torch.full_like(grid[:-1], torch.nan),
            duration=float("inf"),
            total_ms=total_ms,
            preparation_ms=preparation_ms,
            solve_ms=solve_ms,
            solver_wrapper=cfg.solver_wrapper,
            return_code=return_code,
            diagnostics={
                "reason": "compute_parameterization returned no speed profile"
            },
        )

    sd_np = np.asarray(sd, dtype=np.float64)
    sdd_np = np.asarray(sdd, dtype=np.float64) if sdd is not None else np.empty(0)
    finite = (
        sd_np.shape == grid_np.shape
        and sdd_np.shape == (grid_np.size - 1,)
        and np.all(np.isfinite(sd_np))
        and np.all(np.isfinite(sdd_np))
        and np.all(sd_np >= 0.0)
    )
    x = torch.as_tensor(sd_np * sd_np, device=problem.device, dtype=problem.dtype)
    # TOPPRA clips/reduces its forward speed states for controllability after
    # computing the raw LP controls. ParametrizeConstAccel executes the slope
    # of those returned states, not the raw sdd controls. Match that native
    # interpolant so duration, endpoints and physical audit refer to one path.
    executed_acceleration = np.diff(sd_np * sd_np) / (2.0 * np.diff(grid_np))
    path_acceleration = torch.as_tensor(
        executed_acceleration, device=problem.device, dtype=problem.dtype
    )
    duration = _profile_duration(grid_np, sd_np)
    success = bool(finite and np.isfinite(duration))
    return ToppraRAResult(
        success=success,
        s=grid,
        x=x,
        path_acceleration=path_acceleration,
        duration=duration,
        total_ms=total_ms,
        preparation_ms=preparation_ms,
        solve_ms=solve_ms,
        solver_wrapper=cfg.solver_wrapper,
        return_code=return_code,
        diagnostics={
            "grid_samples": cfg.grid_samples,
            "interpolation_constraints": cfg.interpolation_constraints,
            "path_acceleration_samples": (
                0 if sdd is None else int(np.asarray(sdd).size)
            ),
            "safety_scale": cfg.safety_scale,
            "execution_interpolant": "toppra.ParametrizeConstAccel",
            "raw_control_execution_max_difference": (
                float(np.max(np.abs(sdd_np - executed_acceleration)))
                if finite
                else None
            ),
        },
    )


@torch.no_grad()
def validate_toppra_ra_profile_gpu(
    problem: RetimingProblem,
    result: ToppraRAResult,
    *,
    samples: int = 4097,
    tolerance: float = 5e-4,
) -> DenseFeasibilityResult:
    """Audit TOPP-RA using its native constant-acceleration segments."""
    _require_c2(problem)
    return audit_constant_acceleration(
        problem, result, samples=samples, tolerance=tolerance
    )


__all__ = [
    "ToppraRAConfig",
    "ToppraRAResult",
    "ToppraUnavailableError",
    "solve_toppra_ra",
    "validate_toppra_ra_profile_gpu",
]
