"""Optional CPU coefficient-only analytical Pinocchio precomputation.

The frozen Kaserer_fixed implementation evaluates three full RNEA Jacobian
probes *plus* RNEA torques per path checkpoint through a generic bridge.
This specialized routine computes the same m', c', g' with two full
RNEA-derivative probes and one dedicated gravity derivative; it does not
evaluate unused torque outputs or construct new full robot states per probe.

Only select through an explicit research opt-in. This is not a change to
the independent dense certificate or the original Kaserer baseline.
"""
from __future__ import annotations

import time

import numpy as np
import torch
from torch import Tensor


@torch.no_grad()
def precompute_kaserer_analytic_coefficients(
    backend: object,
    q: Tensor,
    q_s: Tensor,
    q_ss: Tensor,
    q_sss: Tensor,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute analytical (m',c',g') along a path in float64 on CPU.

    Uses only Pinocchio derivatives. The path and model must be fixed-base,
    tangent-coordinate-compatible and backed by the source-qualified model.
    No finite-difference fallback is permitted.
    """
    import pinocchio as pin

    vectors = (q, q_s, q_ss, q_sss)
    dof = int(backend.dof)
    if (
        q.ndim != 2
        or q.shape[-1] != dof
        or any(x.shape != q.shape or x.device.type != "cpu" for x in vectors)
        or any(x.dtype != torch.float64 for x in vectors)
    ):
        raise ValueError("Kaserer analytic precompute requires CPU float64 [samples,dof]")
    if any(not bool(torch.isfinite(x).all()) for x in vectors):
        raise ValueError("nonfinite analytical precompute input")
    model = backend.model
    qidx = np.asarray(backend.q_indices, dtype=np.intp)
    vidx = np.asarray(backend.v_indices, dtype=np.intp)
    if (
        dof < 1
        or len(qidx) != dof
        or len(vidx) != dof
        or qidx.min() < 0
        or qidx.max() >= model.nq
        or vidx.min() < 0
        or vidx.max() >= model.nv
        or len(np.unique(qidx)) != dof
        or len(np.unique(vidx)) != dof
    ):
        raise ValueError("invalid Pinocchio joint index mapping")

    positions = q.contiguous().numpy()
    qs = q_s.contiguous().numpy()
    qss = q_ss.contiguous().numpy()
    qsss = q_sss.contiguous().numpy()
    n = q.shape[0]
    mp = np.empty((n, dof), dtype=np.float64)
    cp = np.empty((n, dof), dtype=np.float64)
    gp = np.empty((n, dof), dtype=np.float64)
    indices = np.ix_(vidx, vidx)
    data = model.createData()
    full_q = np.asarray(pin.neutral(model), dtype=np.float64).copy()
    full_v = np.zeros(model.nv, dtype=np.float64)
    full_a = np.zeros(model.nv, dtype=np.float64)
    zeros = np.zeros(model.nv, dtype=np.float64)
    profile = getattr(backend, "kaserer_precompute_profile", None)
    if not isinstance(profile, dict):
        profile = None
    total_started = time.perf_counter_ns() if profile is not None else 0
    gravity_time_ns = 0
    rnea_time_ns = 0
    gravity_calls = 0
    rnea_calls = 0
    if model.nq != model.nv:
        # The existing source adapter and frozen tests target scalar joints.
        # Reject unsupported configuration manifolds instead of guessing the
        # mapping from geometric q_s to Pinocchio tangent coordinates.
        raise ValueError("non-Euclidean Pinocchio model is unsupported")

    for i in range(n):
        full_q[qidx] = positions[i]
        full_v[vidx] = qs[i]
        full_a[vidx] = qss[i]
        # g' uses dedicated gravity derivatives rather than a generic RNEA
        # derivative with zero velocity and acceleration.
        call_started = time.perf_counter_ns() if profile is not None else 0
        gravity_dq = pin.computeGeneralizedGravityDerivatives(model, data, full_q)
        if profile is not None:
            gravity_time_ns += time.perf_counter_ns() - call_started
            gravity_calls += 1
        gprime = gravity_dq[indices] @ qs[i]
        gp[i] = gprime

        # m' = d/ds [RNEA(q,0,q_s) - g(q)].
        # Evaluate m' before the second probe overwrites Data's Jacobians.
        call_started = time.perf_counter_ns() if profile is not None else 0
        dq_m, _, da_m = pin.computeRNEADerivatives(
            model, data, full_q, zeros, full_v
        )
        if profile is not None:
            rnea_time_ns += time.perf_counter_ns() - call_started
            rnea_calls += 1
        mp[i] = dq_m[indices] @ qs[i] + da_m[indices] @ qss[i] - gprime

        # c' = d/ds [RNEA(q,q_s,q_ss) - g(q)].
        call_started = time.perf_counter_ns() if profile is not None else 0
        dq_c, dv_c, da_c = pin.computeRNEADerivatives(
            model, data, full_q, full_v, full_a
        )
        if profile is not None:
            rnea_time_ns += time.perf_counter_ns() - call_started
            rnea_calls += 1
        cp[i] = (
            dq_c[indices] @ qs[i]
            + dv_c[indices] @ qss[i]
            + da_c[indices] @ qsss[i]
            - gprime
        )
    if any(not np.isfinite(arr).all() for arr in (mp, cp, gp)):
        raise ValueError("nonfinite analytical Pinocchio coefficient derivative")
    if isinstance(profile, dict):
        total_ns = time.perf_counter_ns() - total_started
        profile.update(
            total_ms=total_ns / 1.0e6,
            gravity_derivative_ms=gravity_time_ns / 1.0e6,
            rnea_derivative_ms=rnea_time_ns / 1.0e6,
            other_ms=max(0, total_ns - gravity_time_ns - rnea_time_ns) / 1.0e6,
            gravity_derivative_calls=gravity_calls,
            rnea_derivative_calls=rnea_calls,
        )
    return np.ascontiguousarray(mp), np.ascontiguousarray(cp), np.ascontiguousarray(gp)
