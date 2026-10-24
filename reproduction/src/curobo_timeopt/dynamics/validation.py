"""Independent direct-RNEA and autodiff checks for torque and torque rate."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from .base import InverseDynamicsModel
from .coefficients import AffineTorqueCoefficients, reconstruct_left_states


@dataclass(frozen=True)
class TorqueRateValidation:
    """Direct and affine values on the optimization grid.

    Torques live on segment-left samples and rates between adjacent torques.
    ``h`` is Lee's average of the two neighboring segment traversal times.
    """

    direct_torque: Tensor
    affine_torque: Tensor
    direct_rate: Tensor
    affine_rate: Tensor
    h: Tensor

    @property
    def torque_error(self) -> Tensor:
        return self.affine_torque - self.direct_torque

    @property
    def numerator_error(self) -> Tensor:
        return torch.diff(self.affine_torque, dim=-2) - torch.diff(self.direct_torque, dim=-2)

    @property
    def rate_error(self) -> Tensor:
        return self.affine_rate - self.direct_rate


@dataclass(frozen=True)
class InverseDynamicsJVP:
    """Inverse-dynamics value and one Jacobian-vector product.

    Both tensors preserve all leading state dimensions.  Keeping the generic
    directional derivative separate from the time-derivative convenience
    wrapper makes this useful for gradient checks beyond torque rate.
    """

    torque: Tensor
    tangent: Tensor


@dataclass(frozen=True)
class TorqueRateJVPValidation:
    """Comparison of torque secants with autodiff time derivatives.

    ``sampled_rate`` is the torque difference on each time interval.
    ``reference_rate`` is the trapezoidal average of the analytic/JVP rates at
    its endpoints.  Their difference converges to zero for a smooth trajectory
    as the validation grid is refined; it is not a continuous-time certificate.
    """

    torque: Tensor
    analytic_rate: Tensor
    sampled_rate: Tensor
    reference_rate: Tensor
    segment_dt: Tensor

    @property
    def sampled_error(self) -> Tensor:
        return self.sampled_rate - self.reference_rate


def inverse_dynamics_jvp(
    model: InverseDynamicsModel,
    q: Tensor,
    qd: Tensor,
    qdd: Tensor,
    q_tangent: Tensor,
    qd_tangent: Tensor,
    qdd_tangent: Tensor,
    *,
    create_graph: bool = False,
    strict: bool = False,
) -> InverseDynamicsJVP:
    """Evaluate inverse dynamics and its JVP in a supplied state direction.

    This intentionally differentiates through the model's RNEA implementation
    rather than expanding derivatives of mass, Coriolis, or gravity terms.
    ``strict=False`` permits physically ordinary models that do not depend on
    every state input (for example, gravity-only reference models).
    """

    states = (q, qd, qdd)
    tangents = (q_tangent, qd_tangent, qdd_tangent)
    _validate_jvp_inputs(model, states, tangents)

    specialized_pair = getattr(
        model, "inverse_dynamics_with_directional_derivative", None
    )
    if callable(specialized_pair):
        if create_graph:
            raise RuntimeError(
                "backend-specific inverse-dynamics JVP does not support "
                "higher-order graph construction"
            )
        torque, tangent = specialized_pair(q, qd, qdd, *tangents)
        if torque.shape != q.shape or tangent.shape != q.shape:
            raise ValueError(
                "backend-specific inverse-dynamics JVP returned an invalid shape"
            )
        if not torch.all(torch.isfinite(torque)) or not torch.all(
            torch.isfinite(tangent)
        ):
            raise ValueError(
                "inverse dynamics or its backend-specific JVP is non-finite"
            )
        return InverseDynamicsJVP(torque=torque, tangent=tangent)

    specialized = getattr(model, "inverse_dynamics_directional_derivative", None)
    if callable(specialized):
        if create_graph:
            raise RuntimeError(
                "backend-specific inverse-dynamics derivative does not support "
                "higher-order graph construction"
            )
        torque = model.inverse_dynamics(q, qd, qdd)
        tangent = specialized(q, qd, qdd, *tangents)
        if torque.shape != q.shape or tangent.shape != q.shape:
            raise ValueError(
                "backend-specific inverse-dynamics derivative returned an invalid shape"
            )
        if not torch.all(torch.isfinite(torque)) or not torch.all(torch.isfinite(tangent)):
            raise ValueError(
                "inverse dynamics or its backend-specific derivative is non-finite"
            )
        return InverseDynamicsJVP(torque=torque, tangent=tangent)

    def evaluate(position: Tensor, velocity: Tensor, acceleration: Tensor) -> Tensor:
        torque = model.inverse_dynamics(position, velocity, acceleration)
        if torque.shape != position.shape:
            raise ValueError(f"inverse dynamics returned {torque.shape}; expected {position.shape}")
        if torch.is_grad_enabled() and not torque.requires_grad:
            raise RuntimeError(
                "inverse dynamics detached its output; torque-rate JVP requires an "
                "autograd-compatible model"
            )
        return torque

    torque, tangent = torch.autograd.functional.jvp(
        evaluate,
        states,
        tangents,
        create_graph=create_graph,
        strict=strict,
    )
    if not torch.all(torch.isfinite(torque)) or not torch.all(torch.isfinite(tangent)):
        raise ValueError("inverse dynamics or its JVP produced non-finite values")
    return InverseDynamicsJVP(torque=torque, tangent=tangent)


def torque_rate_jvp(
    model: InverseDynamicsModel,
    q: Tensor,
    qd: Tensor,
    qdd: Tensor,
    qddd: Tensor,
    *,
    create_graph: bool = False,
    strict: bool = False,
) -> InverseDynamicsJVP:
    """Evaluate ``tau`` and the analytic time derivative ``d tau / dt``.

    Along a trajectory the tangent of the inverse-dynamics state
    ``(q, qdot, qddot)`` is exactly ``(qdot, qddot, qdddot)``.  One JVP through
    differentiable RNEA therefore yields torque rate without symbolic rigid-body
    dynamics derivatives.
    """

    return inverse_dynamics_jvp(
        model,
        q,
        qd,
        qdd,
        qd,
        qdd,
        qddd,
        create_graph=create_graph,
        strict=strict,
    )


def inverse_dynamics_reverse_mode(
    model: InverseDynamicsModel,
    q: Tensor,
    qd: Tensor,
    qdd: Tensor,
    q_tangent: Tensor,
    qd_tangent: Tensor,
    qdd_tangent: Tensor,
    *,
    chunk_size: int = 25,
) -> InverseDynamicsJVP:
    """Compute an inverse-dynamics directional derivative by reverse mode.

    For each output joint, a VJP of the batch sum recovers the per-state
    Jacobian row when RNEA samples are independent across the batch dimension.
    The result is contracted with ``(qd, qdd, qddd)``. This intentionally uses
    ordinary first-order ``autograd.grad`` and does not depend on forward-mode
    or double-backward support.

    ``chunk_size`` bounds native workspaces and avoids backend-specific batch
    launch regimes. cuRobo certification currently uses 25-state chunks; this
    conservative default is a reference path, not an optimized validator.
    """
    _validate_jvp_inputs(
        model,
        (q, qd, qdd),
        (q_tangent, qd_tangent, qdd_tangent),
    )
    if int(chunk_size) <= 0:
        raise ValueError("chunk_size must be positive")
    original_shape = q.shape
    flat_states = tuple(
        value.reshape(-1, model.dof)
        for value in (q, qd, qdd, q_tangent, qd_tangent, qdd_tangent)
    )
    torque_parts: list[Tensor] = []
    rate_parts: list[Tensor] = []
    with torch.enable_grad():
        for start in range(0, flat_states[0].shape[0], int(chunk_size)):
            stop = min(start + int(chunk_size), flat_states[0].shape[0])
            count = stop - start
            chunks = []
            for value in flat_states:
                chunk = value[start:stop]
                if count < int(chunk_size):
                    padding = chunk[-1:].expand(int(chunk_size) - count, -1)
                    chunk = torch.cat((chunk, padding), dim=0)
                chunks.append(chunk)
            q_leaf, qd_leaf, qdd_leaf = tuple(
                value.detach().clone().requires_grad_(True)
                for value in chunks[:3]
            )
            q_direction, qd_direction, qdd_direction = (
                value for value in chunks[3:]
            )
            tau = model.inverse_dynamics(q_leaf, qd_leaf, qdd_leaf).clone()
            if tau.shape != q_leaf.shape:
                raise ValueError(
                    f"inverse dynamics returned {tau.shape}; expected {q_leaf.shape}"
                )
            if not torch.all(torch.isfinite(tau)):
                raise ValueError("inverse dynamics produced non-finite torque")
            jacobian_rows: list[list[Tensor]] = [[], [], []]
            for joint in range(model.dof):
                grads = torch.autograd.grad(
                    tau[:, joint].sum(),
                    (q_leaf, qd_leaf, qdd_leaf),
                    retain_graph=joint + 1 < model.dof,
                    allow_unused=True,
                )
                for input_index, grad in enumerate(grads):
                    if grad is None:
                        raise RuntimeError(
                            "inverse-dynamics output is independent of an input "
                            f"required by reverse torque-rate authority (joint={joint}, "
                            f"input={input_index})"
                        )
                    copied = grad.detach().clone()
                    if not torch.all(torch.isfinite(copied)):
                        raise ValueError("inverse-dynamics gradient is non-finite")
                    jacobian_rows[input_index].append(copied)
            contractions = [
                torch.einsum("bji,bi->bj", torch.stack(rows, dim=1), direction)
                for rows, direction in zip(
                    jacobian_rows,
                    (q_direction, qd_direction, qdd_direction),
                    strict=True,
                )
            ]
            torque_parts.append(tau[:count].detach())
            rate_parts.append(sum(contractions)[:count])
    torque = torch.cat(torque_parts, dim=0).reshape(original_shape)
    tangent = torch.cat(rate_parts, dim=0).reshape(original_shape)
    if not torch.all(torch.isfinite(tangent)):
        raise ValueError("reverse-mode torque rate produced non-finite values")
    return InverseDynamicsJVP(torque=torque, tangent=tangent)


def torque_rate_reverse_mode(
    model: InverseDynamicsModel,
    q: Tensor,
    qd: Tensor,
    qdd: Tensor,
    qddd: Tensor,
    *,
    chunk_size: int = 25,
) -> InverseDynamicsJVP:
    """Compute analytic torque rate by ordinary reverse-mode contractions."""
    return inverse_dynamics_reverse_mode(
        model,
        q,
        qd,
        qdd,
        qd,
        qdd,
        qddd,
        chunk_size=chunk_size,
    )


def torque_rate_reverse_hybrid(
    model: InverseDynamicsModel,
    q: Tensor,
    qd: Tensor,
    qdd: Tensor,
    qddd: Tensor,
    *,
    chunk_size: int = 25,
) -> InverseDynamicsJVP:
    """Reverse-mode q/qd contribution plus exact RNEA mass action.

    Uses ``J_q qd + J_qd qdd`` from ordinary reverse-mode gradients and
    ``ID(q,0,qddd)-ID(q,0,0)`` for ``J_qdd qddd``. Both inertial states are
    evaluated in one batched call to preserve ownership under reusable native
    RNEA buffers.
    """
    _validate_jvp_inputs(model, (q, qd, qdd), (qd, qdd, qddd))
    if int(chunk_size) <= 0:
        raise ValueError("chunk_size must be positive")
    original_shape = q.shape
    flat_states = tuple(value.reshape(-1, model.dof) for value in (q, qd, qdd, qddd))
    torque_parts: list[Tensor] = []
    rate_parts: list[Tensor] = []
    with torch.enable_grad():
        for start in range(0, flat_states[0].shape[0], int(chunk_size)):
            stop = min(start + int(chunk_size), flat_states[0].shape[0])
            count = stop - start
            chunks = []
            for value in flat_states:
                chunk = value[start:stop]
                if count < int(chunk_size):
                    chunk = torch.cat(
                        (chunk, chunk[-1:].expand(int(chunk_size) - count, -1)),
                        dim=0,
                    )
                chunks.append(chunk)
            q_chunk, qd_chunk, qdd_chunk, qddd_chunk = (
                value for value in chunks
            )
            q_leaf = q_chunk.detach().clone().requires_grad_(True)
            qd_leaf = qd_chunk.detach().clone().requires_grad_(True)
            qdd_leaf = qdd_chunk.detach().clone().requires_grad_(True)
            tau = model.inverse_dynamics(q_leaf, qd_leaf, qdd_leaf).clone()
            if tau.shape != q_leaf.shape:
                raise ValueError(
                    f"inverse dynamics returned {tau.shape}; expected {q_leaf.shape}"
                )
            q_rows: list[Tensor] = []
            qd_rows: list[Tensor] = []
            for joint in range(model.dof):
                grads = torch.autograd.grad(
                    tau[:, joint].sum(),
                    (q_leaf, qd_leaf),
                    retain_graph=joint + 1 < model.dof,
                    allow_unused=True,
                )
                if grads[0] is None or grads[1] is None:
                    raise RuntimeError(
                        "inverse-dynamics output is independent of q or qd in "
                        f"reverse-hybrid torque-rate authority (joint={joint})"
                    )
                q_rows.append(grads[0].detach().clone())
                qd_rows.append(grads[1].detach().clone())
            q_contribution = torch.einsum(
                "bji,bi->bj", torch.stack(q_rows, dim=1), qd_chunk
            )
            qd_contribution = torch.einsum(
                "bji,bi->bj", torch.stack(qd_rows, dim=1), qdd_chunk
            )
            zeros = torch.zeros_like(q_chunk)
            inertial_tau = model.inverse_dynamics(
                torch.cat((q_chunk, q_chunk), dim=0),
                torch.cat((zeros, zeros), dim=0),
                torch.cat((qddd_chunk, zeros), dim=0),
            ).clone()
            mass_action = inertial_tau[: int(chunk_size)] - inertial_tau[
                int(chunk_size) :
            ]
            torque_parts.append(tau[:count].detach())
            rate_parts.append(
                (q_contribution + qd_contribution + mass_action)[:count]
            )
    torque = torch.cat(torque_parts, dim=0).reshape(original_shape)
    tangent = torch.cat(rate_parts, dim=0).reshape(original_shape)
    if not torch.all(torch.isfinite(torque)) or not torch.all(torch.isfinite(tangent)):
        raise ValueError("reverse-hybrid torque-rate produced non-finite values")
    return InverseDynamicsJVP(torque=torque, tangent=tangent)


def torque_rate_rnea_fd5(
    model: InverseDynamicsModel,
    q: Tensor,
    qd: Tensor,
    qdd: Tensor,
    qddd: Tensor,
    *,
    step: float = 3.0e-3,
) -> Tensor:
    """Estimate continuous torque rate with a centered RNEA directional FD5.

    This is intended for independent audits of dynamics backends whose JVP
    implementation is unavailable or untrustworthy.  It differentiates RNEA
    along ``(q, qd, qdd)`` in direction ``(qd, qdd, qddd)``.  The step is in
    seconds; ``3e-3`` is a practical float32 default validated against nearby
    steps on the cuRoboV2 Panda model.
    """
    if not math.isfinite(float(step)) or float(step) <= 0:
        raise ValueError("step must be finite and positive")
    h = float(step)
    values = {}
    for multiple in (-2.0, -1.0, 1.0, 2.0):
        delta = multiple * h
        values[multiple] = model.inverse_dynamics(
            q + delta * qd,
            qd + delta * qdd,
            qdd + delta * qddd,
        )
    return (values[-2.0] - 8.0 * values[-1.0] + 8.0 * values[1.0] - values[2.0]) / (12.0 * h)


def validate_sampled_against_jvp(
    model: InverseDynamicsModel,
    q: Tensor,
    qd: Tensor,
    qdd: Tensor,
    qddd: Tensor,
    segment_dt: Tensor,
) -> TorqueRateJVPValidation:
    """Compare sampled RNEA torque differences with analytic/JVP torque rate.

    State tensors have shape ``(..., samples, dof)`` and ``segment_dt`` has
    shape ``(..., samples - 1)``.  Leading dimensions may broadcast.  The
    comparison uses an endpoint-average reference, which is second-order for a
    smooth torque trajectory on a uniform time grid.
    """

    result = torque_rate_jvp(model, q, qd, qdd, qddd)
    if segment_dt.ndim < 1 or segment_dt.shape[-1] != q.shape[-2] - 1:
        raise ValueError("segment_dt must contain one value per sample interval")
    if segment_dt.device != q.device or segment_dt.dtype != q.dtype:
        raise ValueError("segment_dt and state tensors must share device and dtype")
    if not torch.all(torch.isfinite(segment_dt)) or torch.any(segment_dt <= 0):
        raise ValueError("segment_dt must contain finite, strictly positive values")
    try:
        torch.broadcast_shapes(segment_dt.shape[:-1], q.shape[:-2])
    except RuntimeError as exc:
        raise ValueError("segment_dt batch dimensions are not broadcastable with state") from exc

    sampled_rate = torch.diff(result.torque, dim=-2) / segment_dt[..., :, None]
    reference_rate = 0.5 * (result.tangent[..., :-1, :] + result.tangent[..., 1:, :])
    return TorqueRateJVPValidation(
        torque=result.torque,
        analytic_rate=result.tangent,
        sampled_rate=sampled_rate,
        reference_rate=reference_rate,
        segment_dt=segment_dt,
    )


def validate_sampled_torque_rate(
    model: InverseDynamicsModel,
    coefficients: AffineTorqueCoefficients,
    q: Tensor,
    q_s: Tensor,
    q_ss: Tensor,
    s: Tensor,
    x: Tensor,
) -> TorqueRateValidation:
    """Compare affine torque rates to a separate direct-RNEA reconstruction.

    This validation deliberately does not reuse path dynamics terms. It rebuilds
    ``qdot`` and ``qddot``, calls RNEA, and finite-differences the resulting
    torques with the same discrete Lee denominator used by rate constraints.
    """
    q_left, qd, qdd = reconstruct_left_states(q, q_s, q_ss, s, x)
    direct_torque = model.inverse_dynamics(q_left, qd, qdd)
    affine_torque = coefficients.evaluate(x)
    roots = torch.sqrt(x.clamp_min(0.0))
    segment_dt = 2.0 * torch.diff(s) / (roots[..., :-1] + roots[..., 1:])
    if not torch.all(torch.isfinite(segment_dt)) or torch.any(segment_dt <= 0):
        raise ValueError("x must give finite, strictly positive segment traversal times")
    h = 0.5 * (segment_dt[..., :-1] + segment_dt[..., 1:])
    direct_rate = torch.diff(direct_torque, dim=-2) / h[..., :, None]
    affine_rate = torch.diff(affine_torque, dim=-2) / h[..., :, None]
    return TorqueRateValidation(direct_torque, affine_torque, direct_rate, affine_rate, h)


def _validate_jvp_inputs(
    model: InverseDynamicsModel,
    states: tuple[Tensor, Tensor, Tensor],
    tangents: tuple[Tensor, Tensor, Tensor],
) -> None:
    q = states[0]
    if q.ndim < 1 or q.shape[-1] != model.dof:
        raise ValueError(f"state tensors must end in the dof dimension ({model.dof})")
    for value in (*states[1:], *tangents):
        if value.shape != q.shape:
            raise ValueError("states and JVP tangents must have identical shapes")
        if value.device != q.device or value.dtype != q.dtype:
            raise ValueError("states and JVP tangents must share device and dtype")
    if not q.is_floating_point():
        raise TypeError("inverse-dynamics JVP inputs must use a floating dtype")
    if any(not torch.all(torch.isfinite(value)) for value in (*states, *tangents)):
        raise ValueError("inverse-dynamics JVP states and tangents must be finite")


__all__ = [
    "InverseDynamicsJVP",
    "TorqueRateJVPValidation",
    "TorqueRateValidation",
    "inverse_dynamics_jvp",
    "torque_rate_jvp",
    "torque_rate_reverse_mode",
    "torque_rate_reverse_hybrid",
    "validate_sampled_against_jvp",
    "validate_sampled_torque_rate",
]
