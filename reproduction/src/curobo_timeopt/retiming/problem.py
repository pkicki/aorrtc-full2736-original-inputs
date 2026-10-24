"""Definition and validation of a fixed-path retiming problem."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import torch
from torch import Tensor

from curobo_timeopt.types import ConstraintMode, RobotLimits


class GeometricPath(Protocol):
    @property
    def dof(self) -> int: ...

    def evaluate(self, s: Tensor, derivative_order: int = 0) -> Tensor: ...


class InverseDynamics(Protocol):
    @property
    def dof(self) -> int: ...

    def inverse_dynamics(self, q: Tensor, qd: Tensor, qdd: Tensor) -> Tensor: ...


@dataclass(frozen=True)
class RetimingProblem:
    path: GeometricPath
    limits: RobotLimits
    s: Tensor
    dynamics: InverseDynamics | None = None
    mode: ConstraintMode = ConstraintMode.VELOCITY | ConstraintMode.ACCELERATION
    initial_speed: float = 0.0
    final_speed: float = 0.0
    feasibility_tolerance: float = 1e-8
    x_floor: float = 1e-12
    initial_acceleration: float | None = None
    final_acceleration: float | None = None

    def __post_init__(self) -> None:
        if self.s.ndim != 1 or self.s.numel() < 3:
            raise ValueError("s must be a one-dimensional grid with at least 3 nodes")
        if not torch.all(self.s[1:] > self.s[:-1]):
            raise ValueError("s must be strictly increasing")
        if abs(float(self.s[0])) > 1e-10 or abs(float(self.s[-1]) - 1.0) > 1e-10:
            raise ValueError("path grid must span [0, 1]")
        if not self.s.is_floating_point() or not torch.all(torch.isfinite(self.s)):
            raise ValueError("s must contain finite floating-point values")
        if not math.isfinite(self.initial_speed) or self.initial_speed < 0:
            raise ValueError("initial_speed must be finite and nonnegative")
        if not math.isfinite(self.final_speed) or self.final_speed < 0:
            raise ValueError("final_speed must be finite and nonnegative")
        if self.initial_acceleration is not None and not math.isfinite(self.initial_acceleration):
            raise ValueError("initial_acceleration must be finite when prescribed")
        if self.final_acceleration is not None and not math.isfinite(self.final_acceleration):
            raise ValueError("final_acceleration must be finite when prescribed")
        if not math.isfinite(self.feasibility_tolerance) or self.feasibility_tolerance < 0:
            raise ValueError("feasibility_tolerance must be finite and nonnegative")
        if not math.isfinite(self.x_floor) or self.x_floor <= 0:
            raise ValueError("x_floor must be finite and positive")
        self.limits.validate(self.path.dof)
        tensors = [
            self.limits.velocity,
            self.limits.acceleration,
            self.limits.torque,
            self.limits.jerk,
            self.limits.torque_rate,
        ]
        if any(v.device != self.s.device or v.dtype != self.s.dtype for v in tensors):
            raise ValueError("grid and limit tensors must share device and dtype")
        if (
            self.mode & (ConstraintMode.TORQUE | ConstraintMode.TORQUE_RATE)
            and self.dynamics is None
        ):
            raise ValueError("torque constraints require a dynamics model")
        if self.dynamics is not None and self.dynamics.dof != self.path.dof:
            raise ValueError("path and dynamics degrees of freedom must agree")

    @classmethod
    def _with_path_prevalidated(
        cls,
        template: "RetimingProblem",
        path: GeometricPath,
    ) -> "RetimingProblem":
        """Clone a validated fixed-path problem with a new trusted path.

        Internal hot paths use this when only geometric control points changed.
        The template grid, limits, dynamics, modes, and endpoint conditions have
        already been validated, so repeating CUDA reduction checks here would
        add synchronization without adding benchmark-relevant safety.
        """
        if path.dof != template.path.dof:
            raise ValueError("replacement path must preserve degrees of freedom")
        problem = object.__new__(cls)
        object.__setattr__(problem, "path", path)
        object.__setattr__(problem, "limits", template.limits)
        object.__setattr__(problem, "s", template.s)
        object.__setattr__(problem, "dynamics", template.dynamics)
        object.__setattr__(problem, "mode", template.mode)
        object.__setattr__(problem, "initial_speed", template.initial_speed)
        object.__setattr__(problem, "final_speed", template.final_speed)
        object.__setattr__(
            problem, "initial_acceleration", template.initial_acceleration
        )
        object.__setattr__(
            problem, "final_acceleration", template.final_acceleration
        )
        object.__setattr__(
            problem,
            "feasibility_tolerance",
            template.feasibility_tolerance,
        )
        object.__setattr__(problem, "x_floor", template.x_floor)
        return problem

    @property
    def device(self) -> torch.device:
        return self.s.device

    @property
    def dtype(self) -> torch.dtype:
        return self.s.dtype
