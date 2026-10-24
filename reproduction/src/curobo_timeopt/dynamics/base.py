"""Backend-independent inverse-dynamics interfaces and small reference models."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch
from torch import Tensor


@runtime_checkable
class InverseDynamicsModel(Protocol):
    """Minimal interface consumed by the retiming implementation.

    Implementations must preserve leading batch dimensions and return one torque
    per joint. Inputs have shape ``(..., dof)`` and reside on a common device.
    """

    @property
    def dof(self) -> int:
        """Number of actuated joints."""
        ...

    def inverse_dynamics(self, q: Tensor, qd: Tensor, qdd: Tensor) -> Tensor:
        """Evaluate RNEA for joint position, velocity, and acceleration."""
        ...


@dataclass
class CallableDynamics:
    """Adapt a differentiable torch callable to :class:`InverseDynamicsModel`."""

    dof: int
    function: Callable[[Tensor, Tensor, Tensor], Tensor]

    def inverse_dynamics(self, q: Tensor, qd: Tensor, qdd: Tensor) -> Tensor:
        _validate_state(q, qd, qdd, self.dof)
        tau = self.function(q, qd, qdd)
        if tau.shape != q.shape:
            raise ValueError(f"inverse dynamics returned {tau.shape}; expected {q.shape}")
        return tau


@dataclass
class DiagonalDynamics:
    """Deterministic analytical dynamics for tests and dependency-free examples.

    The model is a collection of independent rotary joints,

    ``tau = inertia*qdd + velocity_quadratic*qd*abs(qd) + gravity*sin(q)``.

    It is deliberately simple, but has the same quadratic-velocity structure as
    rigid-body dynamics and therefore provides an exact TOPP affine oracle.
    """

    inertia: Tensor
    velocity_quadratic: Tensor
    gravity: Tensor

    def __post_init__(self) -> None:
        tensors = (self.inertia, self.velocity_quadratic, self.gravity)
        if any(value.ndim != 1 for value in tensors):
            raise ValueError("diagonal dynamics parameters must be one-dimensional")
        if len({value.numel() for value in tensors}) != 1:
            raise ValueError("diagonal dynamics parameters must have equal lengths")
        if any(not torch.all(torch.isfinite(value)) for value in tensors):
            raise ValueError("diagonal dynamics parameters must be finite")
        if torch.any(self.inertia <= 0):
            raise ValueError("inertia must be strictly positive")

    @property
    def dof(self) -> int:
        return self.inertia.numel()

    def inverse_dynamics(self, q: Tensor, qd: Tensor, qdd: Tensor) -> Tensor:
        _validate_state(q, qd, qdd, self.dof)
        parameters = (self.inertia, self.velocity_quadratic, self.gravity)
        if any(value.device != q.device or value.dtype != q.dtype for value in parameters):
            raise ValueError("model parameters and state must share dtype and device")
        return self.inertia * qdd + self.velocity_quadratic * qd * qd.abs() + self.gravity * q.sin()


def _validate_state(q: Tensor, qd: Tensor, qdd: Tensor, dof: int) -> None:
    if q.shape != qd.shape or q.shape != qdd.shape:
        raise ValueError("q, qd, and qdd must have identical shapes")
    if q.ndim < 1 or q.shape[-1] != dof:
        raise ValueError(f"state tensors must end in the dof dimension ({dof})")
    if q.device != qd.device or q.device != qdd.device:
        raise ValueError("q, qd, and qdd must be on one device")
    if q.dtype != qd.dtype or q.dtype != qdd.dtype:
        raise ValueError("q, qd, and qdd must have one dtype")
