"""Fixed-path inverse-dynamics and affine torque coefficient construction."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .base import InverseDynamicsModel


@dataclass(frozen=True)
class PathDynamicsTerms:
    """TOPP dynamics terms satisfying ``tau = m*s_ddot + c*s_dot**2 + g``."""

    m: Tensor
    c: Tensor
    g: Tensor

    def __post_init__(self) -> None:
        if self.m.shape != self.c.shape or self.m.shape != self.g.shape:
            raise ValueError("m, c, and g must have identical (..., samples, dof) shapes")
        if self.m.ndim < 2:
            raise ValueError("dynamics terms require sample and joint dimensions")


@dataclass(frozen=True)
class AffineTorqueCoefficients:
    """Coefficients for left-segment torques ``A*x[k] + B*x[k+1] + bias``."""

    A: Tensor
    B: Tensor
    bias: Tensor

    def __post_init__(self) -> None:
        if self.A.shape != self.B.shape or self.A.shape != self.bias.shape:
            raise ValueError("A, B, and bias must have identical shapes")
        if self.A.ndim < 2:
            raise ValueError("torque coefficients require segment and joint dimensions")

    @property
    def num_segments(self) -> int:
        return self.A.shape[-2]

    def evaluate(self, x: Tensor) -> Tensor:
        """Evaluate torque without host transfers or Python loops."""
        if x.shape[-1] != self.num_segments + 1:
            raise ValueError(f"x must contain {self.num_segments + 1} path samples")
        return self.A * x[..., :-1, None] + self.B * x[..., 1:, None] + self.bias

    def difference_coefficients(self) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Return exact sparse coefficients of ``tau[k+1] - tau[k]``.

        The result ``(d0,d1,d2,dg)`` acts on ``(x[k],x[k+1],x[k+2],1)``.
        """
        return (
            -self.A[..., :-1, :],
            self.A[..., 1:, :] - self.B[..., :-1, :],
            self.B[..., 1:, :],
            self.bias[..., 1:, :] - self.bias[..., :-1, :],
        )


def extract_path_dynamics(
    model: InverseDynamicsModel,
    q: Tensor,
    q_s: Tensor,
    q_ss: Tensor,
) -> PathDynamicsTerms:
    """Extract exact rigid-body TOPP terms in one batched RNEA call.

    Three probe states are concatenated along the sample dimension. This avoids
    mass-matrix/Coriolis API dependencies and also accommodates cuRobo's reused
    output buffers safely. For standard RNEA,
    ``ID(q, 0, q_s)-ID(q,0,0)`` is ``m`` and
    ``ID(q,q_s,q_ss)-ID(q,0,0)`` is ``c``.
    """
    _validate_path_inputs(model, q, q_s, q_ss)
    specialized = getattr(model, "extract_path_dynamics_terms", None)
    if callable(specialized):
        terms = specialized(q, q_s, q_ss)
        if not isinstance(terms, PathDynamicsTerms):
            raise TypeError(
                "extract_path_dynamics_terms must return PathDynamicsTerms"
            )
        return terms

    zeros = torch.zeros_like(q)
    probe_q = torch.cat((q, q, q), dim=-2)
    probe_qd = torch.cat((zeros, zeros, q_s), dim=-2)
    probe_qdd = torch.cat((zeros, q_s, q_ss), dim=-2)
    g, unit_acceleration, unit_speed = model.inverse_dynamics(probe_q, probe_qd, probe_qdd).chunk(
        3, dim=-2
    )
    m = unit_acceleration - g
    c = unit_speed - g
    return PathDynamicsTerms(m=m, c=c, g=g)


def discretize_affine_torque(terms: PathDynamicsTerms, s: Tensor) -> AffineTorqueCoefficients:
    """Convert path terms to affine torque under forward ``x`` differences.

    At left sample ``k``, ``s_ddot=(x[k+1]-x[k])/(2*ds[k])``.
    The final point has no forward segment and is intentionally not represented.
    """
    if s.ndim != 1 or s.numel() != terms.m.shape[-2]:
        raise ValueError("s must be 1-D and match the path sample count")
    ds = torch.diff(s)
    if torch.any(ds <= 0):
        raise ValueError("path grid s must be strictly increasing")
    scale = (0.5 / ds)[..., None]
    m = terms.m[..., :-1, :]
    c = terms.c[..., :-1, :]
    return AffineTorqueCoefficients(
        A=c - m * scale,
        B=m * scale,
        bias=terms.g[..., :-1, :],
    )


def reconstruct_left_states(
    q: Tensor,
    q_s: Tensor,
    q_ss: Tensor,
    s: Tensor,
    x: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Reconstruct ``q, qdot, qddot`` at segment-left samples."""
    if x.shape[-1] != s.numel():
        raise ValueError("x and s must have the same sample count")
    if not torch.all(torch.isfinite(x)):
        raise ValueError("x must contain only finite values")
    if torch.any(x < 0):
        raise ValueError("x = s_dot**2 must be non-negative")
    ds = torch.diff(s)
    s_ddot = torch.diff(x, dim=-1) / (2.0 * ds)
    speed = torch.sqrt(x[..., :-1].clamp_min(0.0))
    q_left = q[..., :-1, :]
    qd = q_s[..., :-1, :] * speed[..., None]
    qdd = q_ss[..., :-1, :] * x[..., :-1, None] + q_s[..., :-1, :] * s_ddot[..., None]
    broadcast = torch.broadcast_tensors(q_left, qd, qdd)
    return broadcast[0], broadcast[1], broadcast[2]


def validate_affine_torque(
    model: InverseDynamicsModel,
    coefficients: AffineTorqueCoefficients,
    q: Tensor,
    q_s: Tensor,
    q_ss: Tensor,
    s: Tensor,
    x: Tensor,
) -> Tensor:
    """Return elementwise affine-minus-direct-RNEA error for independent checks."""
    q_left, qd, qdd = reconstruct_left_states(q, q_s, q_ss, s, x)
    return coefficients.evaluate(x) - model.inverse_dynamics(q_left, qd, qdd)


def _validate_path_inputs(
    model: InverseDynamicsModel, q: Tensor, q_s: Tensor, q_ss: Tensor
) -> None:
    if q.shape != q_s.shape or q.shape != q_ss.shape:
        raise ValueError("q and path derivatives must have identical shapes")
    if q.ndim < 2 or q.shape[-1] != model.dof:
        raise ValueError("path tensors must have shape (..., samples, model.dof)")
