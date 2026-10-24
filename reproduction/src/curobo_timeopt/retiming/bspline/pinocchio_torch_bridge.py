"""Torch-device bridge around the fixed-base Pinocchio CPU dynamics backend.

Cross-robot generalization experiments prioritize physical-model portability,
not preparation latency.  This adapter lets ordinary URDF/Pinocchio models feed
the existing GPU retiming search: path-dynamics probes are evaluated in
Pinocchio on CPU once per fixed path, then copied back to the input Torch device.

Do not use this bridge for headline online latency measurements.  It is a
quality/generalization preparation backend.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from curobo_timeopt.dynamics import PathDynamicsTerms

from .cpu_pinocchio import PinocchioCpuDynamics


@dataclass
class PinocchioTorchBridge:
    backend: PinocchioCpuDynamics

    @property
    def torque_rate_backend_name(self) -> str:
        """Name the independent non-autograd derivative used by physical audits."""
        return "pinocchio_rnea_fd5"

    @property
    def dof(self) -> int:
        return self.backend.dof

    @torch.no_grad()
    def extract_path_dynamics_terms(
        self,
        q: Tensor,
        q_s: Tensor,
        q_ss: Tensor,
    ) -> PathDynamicsTerms:
        device = q.device
        dtype = q.dtype
        # Preserve the caller precision all the way through the Pinocchio
        # path-dynamics probe. This matters for independent float64 validation
        # (e.g. Track B): requesting float32 here and widening afterwards would
        # silently quantize m/c/g and make the audit only nominally float64.
        shape = q.shape
        if q.shape != q_s.shape or q.shape != q_ss.shape:
            raise ValueError("q/q_s/q_ss shapes must match")
        if q.ndim < 2 or q.shape[-1] != self.dof:
            raise ValueError(f"expected path tensors (..., samples, {self.dof})")

        # PinocchioCpuDynamics intentionally exposes a 2-D [samples,dof]
        # native API. Flatten any leading batch dimensions here so this bridge
        # satisfies the generic InverseDynamicsModel contract used by the
        # batched SLP/SQP builders, then restore the original shape.
        flat_shape = (-1, self.dof)
        terms = self.backend.extract_path_dynamics_terms(
            q.detach().reshape(flat_shape).to(device="cpu", dtype=torch.float64),
            q_s.detach().reshape(flat_shape).to(device="cpu", dtype=torch.float64),
            q_ss.detach().reshape(flat_shape).to(device="cpu", dtype=torch.float64),
            dtype=dtype,
        )
        return PathDynamicsTerms(
            m=terms.m.to(device=device, dtype=dtype).reshape(shape),
            c=terms.c.to(device=device, dtype=dtype).reshape(shape),
            g=terms.g.to(device=device, dtype=dtype).reshape(shape),
        )

    @torch.no_grad()
    def inverse_dynamics(
        self,
        q: Tensor,
        qd: Tensor,
        qdd: Tensor,
    ) -> Tensor:
        shape = q.shape
        if q.shape != qd.shape or q.shape != qdd.shape:
            raise ValueError("q/qd/qdd shapes must match")
        if q.shape[-1] != self.dof:
            raise ValueError(f"expected dof={self.dof}")
        flat_q = q.detach().reshape(-1, self.dof).to("cpu", torch.float64)
        flat_qd = qd.detach().reshape(-1, self.dof).to("cpu", torch.float64)
        flat_qdd = qdd.detach().reshape(-1, self.dof).to("cpu", torch.float64)
        tau = self.backend.inverse_dynamics_numpy(
            flat_q.numpy(),
            flat_qd.numpy(),
            flat_qdd.numpy(),
        )
        return (
            torch.from_numpy(tau)
            .to(
                device=q.device,
                dtype=q.dtype,
            )
            .reshape(shape)
        )

    @torch.no_grad()
    def inverse_dynamics_with_directional_derivative(
        self,
        q: Tensor,
        qd: Tensor,
        qdd: Tensor,
        q_tangent: Tensor,
        qd_tangent: Tensor,
        qdd_tangent: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Return Pinocchio torque and an independent centered RNEA FD5.

        Pinocchio calls cross a NumPy boundary and cannot participate in Torch
        autograd. This derivative is for independent physical audits only.
        Differentiable optimization uses the cuRobo native dynamics adapter.
        """
        states = (q, qd, qdd)
        tangents = (q_tangent, qd_tangent, qdd_tangent)
        if any(value.shape != q.shape for value in (*states, *tangents)):
            raise ValueError("Pinocchio state and tangent tensors must match")
        if any(
            value.device != q.device or value.dtype != q.dtype
            for value in (*states, *tangents)
        ):
            raise ValueError("Pinocchio states and tangents must share device/dtype")
        h = 3.0e-3
        values = {}
        for multiple in (-2.0, -1.0, 0.0, 1.0, 2.0):
            delta = multiple * h
            values[multiple] = self.inverse_dynamics(
                q + delta * q_tangent,
                qd + delta * qd_tangent,
                qdd + delta * qdd_tangent,
            )
        tangent = (
            values[-2.0] - 8.0 * values[-1.0] + 8.0 * values[1.0] - values[2.0]
        ) / (12.0 * h)
        return values[0.0], tangent


__all__ = ["PinocchioTorchBridge"]
