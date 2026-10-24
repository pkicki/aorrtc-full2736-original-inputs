"""Stable public data structures shared by retimers and adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Flag, auto
from typing import Any

import torch
from torch import Tensor


class ConstraintMode(Flag):
    """Independently composable physical constraint families."""

    VELOCITY = auto()
    ACCELERATION = auto()
    TORQUE = auto()
    JERK = auto()
    TORQUE_RATE = auto()

    @classmethod
    def second_order(cls) -> ConstraintMode:
        return cls.VELOCITY | cls.ACCELERATION | cls.TORQUE

    @classmethod
    def all(cls) -> ConstraintMode:
        return cls.second_order() | cls.JERK | cls.TORQUE_RATE


@dataclass(frozen=True)
class RobotLimits:
    """Symmetric per-joint limits; infinity explicitly disables a component."""

    velocity: Tensor
    acceleration: Tensor
    torque: Tensor
    jerk: Tensor
    torque_rate: Tensor

    @classmethod
    def from_optional(
        cls,
        dof: int,
        *,
        velocity: Tensor | float | None = None,
        acceleration: Tensor | float | None = None,
        torque: Tensor | float | None = None,
        jerk: Tensor | float | None = None,
        torque_rate: Tensor | float | None = None,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float64,
    ) -> RobotLimits:
        def value(x: Tensor | float | None) -> Tensor:
            if x is None:
                return torch.full((dof,), torch.inf, device=device, dtype=dtype)
            out = torch.as_tensor(x, device=device, dtype=dtype)
            return out.expand(dof).clone() if out.ndim == 0 else out.clone()

        result = cls(*(value(v) for v in (velocity, acceleration, torque, jerk, torque_rate)))
        result.validate(dof)
        return result

    def validate(self, dof: int | None = None) -> None:
        for name, value in self.__dict__.items():
            if value.ndim != 1 or (dof is not None and value.numel() != dof):
                raise ValueError(f"{name} must have shape ({dof if dof else 'dof'},)")
            if not value.is_floating_point():
                raise TypeError(f"{name} limits must use a floating-point dtype")
            if torch.any(torch.isnan(value)):
                raise ValueError(f"{name} limits must not contain NaN")
            if torch.any(value <= 0):
                raise ValueError(f"{name} limits must be positive")


@dataclass
class RetimingResult:
    success: bool
    x: Tensor
    s: Tensor
    t: Tensor
    q: Tensor
    qd: Tensor
    qdd: Tensor
    duration: float
    iterations: int = 0
    solve_time_ms: float = 0.0
    qddd: Tensor | None = None
    tau: Tensor | None = None
    tau_rate: Tensor | None = None
    coefficients: Tensor | None = None
    max_constraint_violation: dict[str, float] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)
