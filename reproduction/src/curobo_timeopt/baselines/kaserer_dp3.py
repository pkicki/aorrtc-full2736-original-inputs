"""Paper-faithful clean-room implementation of Kaserer et al. DP3.

Reference
---------
D. Kaserer, H. Gattringer, and A. Müller,
"Nearly Optimal Path Following With Jerk and Torque Rate Limits Using
Dynamic Programming," IEEE Transactions on Robotics, 35(2), 521-528, 2019.
DOI: 10.1109/TRO.2018.2880120.

This module follows Algorithms 1-3 and Sections III-IV of the paper:

* z = sdot**2 is discretized; z' is NOT discretized.
* The DP state is (s_i, z_i,j).
* The best outgoing interpolation at every DP state stores one cost-to-go,
  successor index, and induced slope z'_i,j.
* The final interval uses the 2-parameter linear-z profile.
* Interior backward edges use the 3-parameter profile in sqrt(z), matching
  z_i,j, z_i+1,l and the already-stored right slope z'_i+1,l.
* The first interval uses the 2-parameter profile and the second interval uses
  the 4-parameter profile to enforce slope continuity at s_2.
* Search ranges are obtained from node-level z' bounds due to acceleration
  and torque constraints (power is absent from the repository's common C3
  benchmark).
* Jerk and torque-rate constraints are checked along interpolation profiles.

The repository's generic dynamics interface does not expose analytic
derivatives of inverse dynamics.  Joint jerk is evaluated analytically from
z,z',z'', while torque rate is checked from adjacent interpolation check
points in physical time.  Final feasibility is independently audited on a
dense grid with the repository validator.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Literal

import numpy as np
import torch
from torch import Tensor

from curobo_timeopt.baselines.kaserer_dp3_fast import get_kaserer_dp3_fast_extension
from curobo_timeopt.baselines.toppra_ra import _precompute_torque_terms
from curobo_timeopt.retiming.gpu_dense_validation import (
    DenseFeasibilityResult,
    validate_dense_path_profile_gpu,
)
from curobo_timeopt.retiming.problem import RetimingProblem
from curobo_timeopt.types import ConstraintMode

ProfileKind = Literal["c2", "c3", "c4"]


@dataclass(frozen=True)
class KasererDP3Config:
    """Numerical settings for the paper-faithful DP3 reconstruction."""

    grid_samples: int = 40
    z_samples: int = 128
    check_points: int = 17
    quadrature_order: int = 16
    safety_scale: float = 0.995
    feasibility_tolerance: float = 1.0e-10
    use_native: bool = True
    torque_rate_mode: Literal["finite_difference", "analytic"] = "finite_difference"
    path_segment_check_points: int = 0
    endpoint_path_segment_check_points: int = 0
    endpoint_z_enrichment: bool = False
    endpoint_z_enrichment_check_points: int = 0
    endpoint_z_enrichment_count: int = 1
    # Diagnostic-only K2 search: densely sample z2 in the continuous
    # Algorithm-3 start interval while keeping the published downstream DP
    # states/slopes fixed. Zero preserves the literal solver behavior.
    algorithm3_local_refinement_points: int = 0
    # Diagnostic-only oracle: scan node-feasible stage-3 slopes for the C4
    # start bridge while keeping z2/z3 on the existing reachable state grids.
    algorithm3_free_slope_scan_points: int = 0
    # Diagnostic K3a: retain up to M distinct feasible outgoing slopes at
    # stage 3 only. M=1 preserves the paper's single-label DP behavior.
    algorithm3_stage3_labels: int = 1
    # Diagnostic K3: retain multiple slope/cost continuations at every
    # Algorithm-2 state. One preserves the paper's single-label DP.
    dp_labels_per_state: int = 1
    # Retention for diagnostic multi-label DP. "cost" preserves the
    # cheapest distinct slopes; "slope_diverse" preserves the cheapest
    # label plus alternatives spread across reachable slope space.
    dp_label_retention: str = "cost"

    def __post_init__(self) -> None:
        if self.grid_samples < 5:
            raise ValueError("grid_samples must be at least 5")
        if self.z_samples < 4:
            raise ValueError("z_samples must be at least 4")
        if self.check_points < 3:
            raise ValueError("check_points must be at least 3")
        if self.quadrature_order < 4:
            raise ValueError("quadrature_order must be at least 4")
        if not 0.0 < self.safety_scale <= 1.0:
            raise ValueError("safety_scale must be in (0,1]")
        if self.torque_rate_mode not in ("finite_difference", "analytic"):
            raise ValueError("invalid Kaserer torque_rate_mode")
        if self.feasibility_tolerance < 0.0:
            raise ValueError("feasibility_tolerance must be nonnegative")
        if self.path_segment_check_points not in (0,) and self.path_segment_check_points < 3:
            raise ValueError("path_segment_check_points must be 0 or at least 3")
        if (
            self.endpoint_path_segment_check_points not in (0,)
            and self.endpoint_path_segment_check_points < 3
        ):
            raise ValueError(
                "endpoint_path_segment_check_points must be 0 or at least 3"
            )
        if (
            self.endpoint_z_enrichment_check_points not in (0,)
            and self.endpoint_z_enrichment_check_points < 3
        ):
            raise ValueError(
                "endpoint_z_enrichment_check_points must be 0 or at least 3"
            )
        if self.endpoint_z_enrichment_count < 1:
            raise ValueError("endpoint_z_enrichment_count must be at least 1")
        if (
            self.algorithm3_local_refinement_points not in (0,)
            and self.algorithm3_local_refinement_points < 3
        ):
            raise ValueError(
                "algorithm3_local_refinement_points must be 0 or at least 3"
            )
        if (
            self.algorithm3_free_slope_scan_points not in (0,)
            and self.algorithm3_free_slope_scan_points < 3
        ):
            raise ValueError(
                "algorithm3_free_slope_scan_points must be 0 or at least 3"
            )
        if self.algorithm3_stage3_labels < 1:
            raise ValueError(
                "algorithm3_stage3_labels must be at least 1"
            )
        if self.dp_labels_per_state < 1:
            raise ValueError("dp_labels_per_state must be at least 1")
        if self.dp_label_retention not in ("cost", "slope_diverse"):
            raise ValueError(
                "dp_label_retention must be cost or slope_diverse"
            )


@dataclass(frozen=True)
class KasererDP3Result:
    success: bool
    s: Tensor
    z: Tensor
    z_prime: Tensor
    duration: float
    total_ms: float
    preparation_ms: float
    search_ms: float
    status: str
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass
class _Stage:
    z_grid: np.ndarray
    cost: np.ndarray
    successor: np.ndarray
    slope: np.ndarray


@dataclass
class _LabelStage:
    z_grid: np.ndarray
    cost: np.ndarray
    successor: np.ndarray
    successor_label: np.ndarray
    slope: np.ndarray


@dataclass(frozen=True)
class _EdgeGeometry:
    eta: np.ndarray
    q_s: np.ndarray
    q_ss: np.ndarray
    q_sss: np.ndarray
    m: np.ndarray
    c: np.ndarray
    gravity: np.ndarray
    m_prime: np.ndarray | None = None
    c_prime: np.ndarray | None = None
    gravity_prime: np.ndarray | None = None


def _require_problem(problem: RetimingProblem) -> None:
    if problem.mode != ConstraintMode.all():
        raise ValueError(
            "paper DP3 common benchmark requires full C3="
            "{velocity, acceleration, torque, jerk, torque_rate}"
        )
    if problem.dynamics is None:
        raise ValueError("paper DP3 requires inverse dynamics")


def _limit(value: Tensor, scale: float) -> np.ndarray:
    out = value.detach().cpu().double().numpy()
    if out.ndim != 1 or np.any(~np.isfinite(out)) or np.any(out <= 0.0):
        raise ValueError("DP3 requires finite positive symmetric limits")
    return np.ascontiguousarray(scale * out, dtype=np.float64)


def _linear_abs_interval(
    coefficient: float,
    constant: float,
    limit: float,
    lower: float,
    upper: float,
    eps: float = 1.0e-14,
) -> tuple[float, float] | None:
    if abs(coefficient) <= eps:
        if abs(constant) <= limit:
            return lower, upper
        return None
    a = (-limit - constant) / coefficient
    b = (limit - constant) / coefficient
    lo, hi = (a, b) if a <= b else (b, a)
    lo = max(lo, lower)
    hi = min(hi, upper)
    if lo > hi:
        return None
    return float(lo), float(hi)


def _velocity_z_cap(q_s: np.ndarray, velocity: np.ndarray) -> float:
    cap = float("inf")
    for derivative, limit in zip(q_s, velocity):
        a = abs(float(derivative))
        if a > 1.0e-14:
            cap = min(cap, (float(limit) / a) ** 2)
    return float(cap)


def _zprime_interval(
    index: int,
    z: float,
    *,
    q_s: np.ndarray,
    q_ss: np.ndarray,
    m: np.ndarray,
    c: np.ndarray,
    gravity: np.ndarray,
    acceleration: np.ndarray,
    torque: np.ndarray,
) -> tuple[float, float] | None:
    """Equation-(21/22)-style node bounds on z'=d(sdot^2)/ds."""

    lo = -float("inf")
    hi = float("inf")
    for j in range(q_s.shape[1]):
        interval = _linear_abs_interval(
            0.5 * float(q_s[index, j]),
            float(q_ss[index, j]) * z,
            float(acceleration[j]),
            lo,
            hi,
        )
        if interval is None:
            return None
        lo, hi = interval

        interval = _linear_abs_interval(
            0.5 * float(m[index, j]),
            float(c[index, j]) * z + float(gravity[index, j]),
            float(torque[j]),
            lo,
            hi,
        )
        if interval is None:
            return None
        lo, hi = interval

    if lo > hi:
        return None
    return float(lo), float(hi)


def _node_z_cap(
    index: int,
    *,
    q_s: np.ndarray,
    q_ss: np.ndarray,
    m: np.ndarray,
    c: np.ndarray,
    gravity: np.ndarray,
    velocity: np.ndarray,
    acceleration: np.ndarray,
    torque: np.ndarray,
) -> float:
    """Maximum node-feasible z under velocity, acceleration and torque."""

    velocity_cap = _velocity_z_cap(q_s[index], velocity)
    if not math.isfinite(velocity_cap) or velocity_cap <= 0.0:
        return 0.0

    def feasible(z: float) -> bool:
        return (
            _zprime_interval(
                index,
                z,
                q_s=q_s,
                q_ss=q_ss,
                m=m,
                c=c,
                gravity=gravity,
                acceleration=acceleration,
                torque=torque,
            )
            is not None
        )

    if feasible(velocity_cap):
        return float(velocity_cap)

    # The projection of the local linear inequalities in (z,z') is convex.
    # Scan from the velocity cap downwards to robustly locate its upper edge,
    # then refine the boundary.
    probe = np.linspace(velocity_cap, 0.0, 257, dtype=np.float64)
    lower = None
    upper = None
    previous = float(probe[0])
    for raw in probe[1:]:
        value = float(raw)
        if feasible(value):
            lower = value
            upper = previous
            break
        previous = value
    if lower is None or upper is None:
        return 0.0

    lo = lower
    hi = upper
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if feasible(mid):
            lo = mid
        else:
            hi = mid
    return float(lo)


def _c2(z0: float, z1: float) -> np.ndarray:
    # z(eta) = c1 eta + c0, eta in [-1,0].
    return np.asarray([z1, z1 - z0], dtype=np.float64)


def _c3(z0: float, z1: float, z1_prime_s: float, ds: float) -> np.ndarray | None:
    # sqrt(z) = a eta^2 + b eta + c. Match z0,z1 and z'_s at eta=0.
    if z0 <= 0.0 or z1 <= 0.0 or ds <= 0.0:
        return None
    v0 = math.sqrt(z0)
    v1 = math.sqrt(z1)
    b = z1_prime_s * ds / (2.0 * v1)
    a = v0 + b - v1
    coeff = np.asarray([v1, b, a], dtype=np.float64)
    return coeff if _positive_speed_polynomial(coeff) else None


def _c4(
    z0: float,
    z1: float,
    z0_prime_s: float,
    z1_prime_s: float,
    ds: float,
) -> np.ndarray | None:
    # sqrt(z) = a eta^3 + b eta^2 + c eta + d.
    if z0 <= 0.0 or z1 <= 0.0 or ds <= 0.0:
        return None
    v0 = math.sqrt(z0)
    v1 = math.sqrt(z1)
    c1 = z1_prime_s * ds / (2.0 * v1)
    r0 = z0_prime_s * ds / (2.0 * v0)
    A = v0 + c1 - v1
    a = r0 + 2.0 * A - c1
    b = r0 + 3.0 * A - c1
    coeff = np.asarray([v1, c1, b, a], dtype=np.float64)
    return coeff if _positive_speed_polynomial(coeff) else None


def _poly_value(coeff: np.ndarray, eta: np.ndarray | float) -> np.ndarray:
    # coeff is ascending order.
    return np.polynomial.polynomial.polyval(eta, coeff)


def _positive_speed_polynomial(coeff: np.ndarray) -> bool:
    """Guarantee sqrt(z)>0 throughout eta in [-1,0].

    C3/C4 profiles have degree at most three, so avoid np.roots in the
    transition hot loop and solve derivative extrema analytically.
    """

    if coeff.ndim != 1 or coeff.size < 1 or coeff.size > 4:
        raise ValueError("speed polynomial must have degree <= 3")
    if not np.all(np.isfinite(coeff)):
        return False

    values = [
        float(_poly_value(coeff, -1.0)),
        float(_poly_value(coeff, 0.0)),
    ]

    if coeff.size == 3:
        # v(eta)=c0+c1*eta+c2*eta^2. Only a convex interior vertex can
        # lower the polynomial below both endpoints.
        c1 = float(coeff[1])
        c2 = float(coeff[2])
        if c2 > 0.0:
            root = -c1 / (2.0 * c2)
            if -1.0 < root < 0.0:
                values.append(float(_poly_value(coeff, root)))
    elif coeff.size == 4:
        # v'(eta)=c1+2*c2*eta+3*c3*eta^2.
        c1 = float(coeff[1])
        c2 = float(coeff[2])
        c3 = float(coeff[3])
        a = 3.0 * c3
        b = 2.0 * c2
        c = c1
        if abs(a) <= 1.0e-14:
            if abs(b) > 1.0e-14:
                root = -c / b
                if -1.0 < root < 0.0:
                    values.append(float(_poly_value(coeff, root)))
        else:
            disc = b * b - 4.0 * a * c
            if disc >= 0.0:
                sqrt_disc = math.sqrt(disc)
                for root in (
                    (-b - sqrt_disc) / (2.0 * a),
                    (-b + sqrt_disc) / (2.0 * a),
                ):
                    if -1.0 < root < 0.0:
                        values.append(float(_poly_value(coeff, root)))

    return min(values) > 1.0e-14


def _evaluate_profile(
    kind: ProfileKind,
    coeff: np.ndarray,
    ds: float,
    eta: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return z, dz/ds, d2z/ds2 at eta."""

    if kind == "c2":
        z = coeff[0] + coeff[1] * eta
        zp = np.full_like(eta, coeff[1] / ds)
        zpp = np.zeros_like(eta)
        return z, zp, zpp

    v = _poly_value(coeff, eta)
    dcoeff = np.arange(1, coeff.size, dtype=np.float64) * coeff[1:]
    ddcoeff = (
        np.arange(2, coeff.size, dtype=np.float64)
        * np.arange(1, coeff.size - 1, dtype=np.float64)
        * coeff[2:]
    )
    v_eta = _poly_value(dcoeff, eta)
    v_etaeta = (
        _poly_value(ddcoeff, eta)
        if ddcoeff.size
        else np.zeros_like(eta, dtype=np.float64)
    )
    z = v * v
    zp = 2.0 * v * v_eta / ds
    zpp = 2.0 * (v_eta * v_eta + v * v_etaeta) / (ds * ds)
    return z, zp, zpp


def _left_slope(kind: ProfileKind, coeff: np.ndarray, ds: float) -> float:
    _, zp, _ = _evaluate_profile(
        kind,
        coeff,
        ds,
        np.asarray([-1.0], dtype=np.float64),
    )
    return float(zp[0])


@lru_cache(maxsize=None)
def _quadrature_rule(order: int) -> tuple[np.ndarray, np.ndarray]:
    nodes, weights = np.polynomial.legendre.leggauss(order)
    eta = np.ascontiguousarray(0.5 * (nodes - 1.0), dtype=np.float64)
    weight = np.ascontiguousarray(0.5 * weights, dtype=np.float64)
    eta.setflags(write=False)
    weight.setflags(write=False)
    return eta, weight


def _edge_time(
    kind: ProfileKind,
    coeff: np.ndarray,
    ds: float,
    quadrature_order: int,
) -> float:
    if kind == "c2":
        z0 = float(_poly_value(coeff, -1.0))
        z1 = float(_poly_value(coeff, 0.0))
        denom = math.sqrt(max(z0, 0.0)) + math.sqrt(max(z1, 0.0))
        return float("inf") if denom <= 0.0 else 2.0 * ds / denom

    eta, weight = _quadrature_rule(quadrature_order)
    speed = _poly_value(coeff, eta)
    if np.any(speed <= 0.0) or np.any(~np.isfinite(speed)):
        return float("inf")
    return float(ds * np.dot(weight, 1.0 / speed))


def _path_support_grid(problem: RetimingProblem) -> np.ndarray:
    """Return exposed geometric-path breakpoints in [0,1], if available."""

    raw = None
    if hasattr(problem.path, "s"):
        raw = getattr(problem.path, "s")
    elif hasattr(problem.path, "knots"):
        raw = getattr(problem.path, "knots")
    if raw is None:
        return np.empty(0, dtype=np.float64)
    if torch.is_tensor(raw):
        raw = raw.detach().cpu().double().numpy()
    values = np.unique(np.asarray(raw, dtype=np.float64).reshape(-1))
    return values[(values >= 0.0) & (values <= 1.0)]


def _edge_local_path_points(
    edge: int,
    edge_count: int,
    interior_points: int,
    endpoint_points: int,
) -> int:
    if endpoint_points > 0 and edge in (0, edge_count - 1):
        return endpoint_points
    return interior_points


def _edge_check_points(
    problem: RetimingProblem,
    grid: Tensor,
    check_points: int,
    path_segment_check_points: int,
    endpoint_path_segment_check_points: int = 0,
) -> list[Tensor]:
    """Build each DP edge's nominal + optional path-segment-aware checks.

    The nominal paper grid is always present.  When path-segment checks are
    requested, every exposed geometric path segment intersecting a DP edge is
    additionally sampled on its own local grid.  This targets narrow
    third-order features caused by highly nonuniform natural-cubic support
    without forcing thousands of uniform checks over the whole DP edge.
    """

    eta = torch.linspace(
        -1.0,
        0.0,
        check_points,
        device=problem.device,
        dtype=problem.dtype,
    )
    ds = grid[1:] - grid[:-1]
    base = grid[1:, None] + ds[:, None] * eta[None, :]
    if (
        path_segment_check_points == 0
        and endpoint_path_segment_check_points == 0
    ):
        return [base[k].contiguous() for k in range(base.shape[0])]

    support = _path_support_grid(problem)
    if support.size < 2:
        return [base[k].contiguous() for k in range(base.shape[0])]

    grid_cpu = grid.detach().cpu().double().numpy()
    result: list[Tensor] = []
    edge_count = grid.numel() - 1
    for k in range(edge_count):
        local_path_points = _edge_local_path_points(
            k,
            edge_count,
            path_segment_check_points,
            endpoint_path_segment_check_points,
        )
        if local_path_points == 0:
            result.append(base[k].contiguous())
            continue

        left = float(grid_cpu[k])
        right = float(grid_cpu[k + 1])
        cuts = support[(support > left) & (support < right)]
        boundaries = np.concatenate(
            (
                np.asarray([left], dtype=np.float64),
                cuts,
                np.asarray([right], dtype=np.float64),
            )
        )
        extras: list[Tensor] = []
        for a, b in zip(boundaries[:-1], boundaries[1:]):
            if not b > a:
                continue
            extras.append(
                torch.linspace(
                    float(a),
                    float(b),
                    local_path_points,
                    device=problem.device,
                    dtype=problem.dtype,
                )
            )
        if extras:
            points = torch.unique(
                torch.cat((base[k], *extras)),
                sorted=True,
            )
        else:
            points = base[k]
        result.append(points.contiguous())
    return result



@torch.no_grad()
def _analytic_torque_coefficients(
    problem: RetimingProblem, samples: Tensor
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Analytic path derivatives of m, c, g, computed once per path checkpoint.

    tau(s) = 0.5*m(s)*z'(s) + c(s)*z(s) + g(s).
    Uses three analytical RNEA-JVP probes and not finite time differences.
    """
    dynamics = problem.dynamics
    authority = getattr(dynamics, "torque_rate_backend_name", None)
    if authority != "pinocchio_computeRNEADerivatives_contraction":
        raise RuntimeError(
            "Kaserer_fixed requires analytic Pinocchio RNEA derivative authority, "
            f"not {authority!r}; generic PinocchioTorchBridge uses FD5"
        )
    hook = getattr(dynamics, "inverse_dynamics_directional_derivative", None)
    if not callable(hook):
        raise RuntimeError("Kaserer_fixed requires analytical RNEA JVP")
    precompute_started = time.perf_counter_ns()
    q = problem.path.evaluate(samples, 0)
    qs = problem.path.evaluate(samples, 1)
    qss = problem.path.evaluate(samples, 2)
    qsss = problem.path.evaluate(samples, 3)
    mode = getattr(dynamics, "kaserer_analytic_precompute_mode", "reference")
    if mode == "pinocchio_coefficients_fast_v1":
        specialized = getattr(dynamics, "kaserer_fast_analytic_coefficients", None)
        if not callable(specialized):
            raise RuntimeError("fast analytical coefficient backend unavailable")
        outputs = specialized(q, qs, qss, qsss)
        if (
            not isinstance(outputs, tuple) or len(outputs) != 3
            or any(
                not isinstance(value, np.ndarray)
                or value.shape != tuple(q.shape)
                or value.dtype != np.float64
                or not np.isfinite(value).all()
                for value in outputs
            )
        ):
            raise ValueError("invalid fast analytical Pinocchio coefficient output")
        if hasattr(dynamics, "kaserer_analytic_precompute_ms"):
            dynamics.kaserer_analytic_precompute_ms += (
                time.perf_counter_ns() - precompute_started
            ) / 1.0e6
        return tuple(np.ascontiguousarray(value) for value in outputs)
    if mode != "reference":
        raise ValueError(f"unsupported Kaserer analytic precompute mode: {mode!r}")
    zeros = torch.zeros_like(q)
    tangent = hook(
        torch.cat((q, q, q), dim=0),
        torch.cat((zeros, zeros, qs), dim=0),
        torch.cat((zeros, qs, qss), dim=0),
        torch.cat((qs, qs, qs), dim=0),
        torch.cat((zeros, zeros, qss), dim=0),
        torch.cat((zeros, qss, qsss), dim=0),
    )
    if tangent.shape != (3 * q.shape[0], q.shape[1]) or not torch.isfinite(tangent).all():
        raise ValueError("Kaserer_fixed analytical JVP is nonfinite or wrong shape")
    gp, mp_gp, cp_gp = tangent.chunk(3, dim=0)
    if hasattr(dynamics, "kaserer_analytic_precompute_ms"):
        dynamics.kaserer_analytic_precompute_ms += (
            time.perf_counter_ns() - precompute_started
        ) / 1.0e6
    return tuple(
        np.ascontiguousarray(value.detach().cpu().double().numpy())
        for value in (mp_gp - gp, cp_gp - gp, gp)
    )


def _native_rate_tensors(geometry: _EdgeGeometry) -> tuple[Tensor, Tensor, Tensor]:
    """Optional native analytical arrays; default baseline passes empty arrays."""
    arrays = (geometry.m_prime, geometry.c_prime, geometry.gravity_prime)
    if any(v is None for v in arrays):
        if not all(v is None for v in arrays):
            raise ValueError("partial analytic torque-rate geometry")
        return tuple(torch.empty((0, 0), dtype=torch.float64) for _ in range(3))
    return tuple(torch.from_numpy(np.ascontiguousarray(v)) for v in arrays)


def _precompute_geometry(
    problem: RetimingProblem,
    grid: Tensor,
    check_points: int,
    path_segment_check_points: int = 0,
    endpoint_path_segment_check_points: int = 0,
    analytic_rate: bool = False,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[_EdgeGeometry],
]:
    """Precompute node and all-edge geometry in batched path/RNEA calls."""

    q_s_t = problem.path.evaluate(grid, 1)
    q_ss_t = problem.path.evaluate(grid, 2)
    q_s = np.ascontiguousarray(q_s_t.detach().cpu().double().numpy())
    q_ss = np.ascontiguousarray(q_ss_t.detach().cpu().double().numpy())
    m, c, gravity = _precompute_torque_terms(problem, grid)
    m = np.ascontiguousarray(m, dtype=np.float64)
    c = np.ascontiguousarray(c, dtype=np.float64)
    gravity = np.ascontiguousarray(gravity, dtype=np.float64)

    edge_points = _edge_check_points(
        problem,
        grid,
        check_points,
        path_segment_check_points,
        endpoint_path_segment_check_points,
    )
    counts = [int(points.numel()) for points in edge_points]
    flat_points = torch.cat(edge_points)
    dof = problem.path.dof

    qs_flat = problem.path.evaluate(flat_points, 1)
    qss_flat = problem.path.evaluate(flat_points, 2)
    qsss_flat = problem.path.evaluate(flat_points, 3)
    em_flat, ec_flat, eg_flat = _precompute_torque_terms(
        problem, flat_points
    )

    qs_np = np.ascontiguousarray(
        qs_flat.detach().cpu().double().numpy()
    )
    qss_np = np.ascontiguousarray(
        qss_flat.detach().cpu().double().numpy()
    )
    qsss_np = np.ascontiguousarray(
        qsss_flat.detach().cpu().double().numpy()
    )
    rate_derivatives = (
        _analytic_torque_coefficients(problem, flat_points)
        if analytic_rate else (None, None, None)
    )
    em_np = np.ascontiguousarray(np.asarray(em_flat, dtype=np.float64))
    ec_np = np.ascontiguousarray(np.asarray(ec_flat, dtype=np.float64))
    eg_np = np.ascontiguousarray(np.asarray(eg_flat, dtype=np.float64))

    edge_geometry: list[_EdgeGeometry] = []
    offset = 0
    grid_cpu = grid.detach().cpu().double().numpy()
    for k, (points, count) in enumerate(zip(edge_points, counts)):
        sl = slice(offset, offset + count)
        ds_k = float(grid_cpu[k + 1] - grid_cpu[k])
        point_np = points.detach().cpu().double().numpy()
        local_path_points = _edge_local_path_points(
            k,
            grid.numel() - 1,
            path_segment_check_points,
            endpoint_path_segment_check_points,
        )
        if local_path_points == 0:
            eta_np = np.linspace(
                -1.0, 0.0, check_points, dtype=np.float64
            )
        else:
            eta_np = np.ascontiguousarray(
                (point_np - float(grid_cpu[k + 1])) / ds_k,
                dtype=np.float64,
            )
        edge_geometry.append(
            _EdgeGeometry(
                eta=eta_np,
                q_s=np.ascontiguousarray(qs_np[sl].reshape(count, dof)),
                q_ss=np.ascontiguousarray(qss_np[sl].reshape(count, dof)),
                q_sss=np.ascontiguousarray(qsss_np[sl].reshape(count, dof)),
                m=np.ascontiguousarray(em_np[sl].reshape(count, dof)),
                c=np.ascontiguousarray(ec_np[sl].reshape(count, dof)),
                gravity=np.ascontiguousarray(eg_np[sl].reshape(count, dof)),
                m_prime=(
                    np.ascontiguousarray(rate_derivatives[0][sl])
                    if analytic_rate else None
                ),
                c_prime=(
                    np.ascontiguousarray(rate_derivatives[1][sl])
                    if analytic_rate else None
                ),
                gravity_prime=(
                    np.ascontiguousarray(rate_derivatives[2][sl])
                    if analytic_rate else None
                ),
            )
        )
        offset += count

    return q_s, q_ss, m, c, gravity, edge_geometry


def _edge_feasible(
    edge: int,
    kind: ProfileKind,
    coeff: np.ndarray,
    ds: float,
    geometry: _EdgeGeometry,
    *,
    eta: np.ndarray,
    velocity: np.ndarray,
    acceleration: np.ndarray,
    torque: np.ndarray,
    jerk: np.ndarray,
    torque_rate: np.ndarray,
    tolerance: float,
) -> bool:
    del edge
    z, zp, zpp = _evaluate_profile(kind, coeff, ds, eta)
    if (
        np.any(~np.isfinite(z))
        or np.any(~np.isfinite(zp))
        or np.any(~np.isfinite(zpp))
        or np.min(z) < -tolerance
    ):
        return False

    speed = np.sqrt(np.maximum(z, 0.0))
    qd = geometry.q_s * speed[:, None]
    if np.any(np.abs(qd) > velocity[None, :] + tolerance):
        return False

    qdd = geometry.q_ss * z[:, None] + 0.5 * geometry.q_s * zp[:, None]
    if np.any(np.abs(qdd) > acceleration[None, :] + tolerance):
        return False

    qddd = (
        geometry.q_sss * (z * speed)[:, None]
        + 1.5 * geometry.q_ss * (speed * zp)[:, None]
        + 0.5 * geometry.q_s * (speed * zpp)[:, None]
    )
    if np.any(np.abs(qddd) > jerk[None, :] + tolerance):
        return False

    tau = (
        0.5 * geometry.m * zp[:, None]
        + geometry.c * z[:, None]
        + geometry.gravity
    )
    if np.any(np.abs(tau) > torque[None, :] + tolerance):
        return False

    if geometry.m_prime is not None:
        if geometry.c_prime is None or geometry.gravity_prime is None:
            raise ValueError("incomplete Kaserer_fixed analytical geometry")
        analytic_rate = speed[:, None] * (
            0.5 * geometry.m_prime * zp[:, None]
            + 0.5 * geometry.m * zpp[:, None]
            + geometry.c_prime * z[:, None]
            + geometry.c * zp[:, None]
            + geometry.gravity_prime
        )
        if (not np.isfinite(analytic_rate).all()
                or np.any(np.abs(analytic_rate) > torque_rate[None, :] + tolerance)):
            return False
    elif eta.size >= 2:
        delta_s = ds * np.diff(eta)
        denom = speed[:-1] + speed[1:]
        dt = np.full_like(denom, np.inf)
        positive = denom > 1.0e-14
        dt[positive] = 2.0 * delta_s[positive] / denom[positive]
        if np.any(~np.isfinite(dt)) or np.any(dt <= 0.0):
            return False
        tau_rate_value = np.diff(tau, axis=0) / dt[:, None]
        if np.any(np.abs(tau_rate_value) > torque_rate[None, :] + tolerance):
            return False

    return True


def _terminal_stage_native(
    current_z: np.ndarray,
    *,
    z_end: float,
    ds: float,
    eta_t: Tensor,
    geometry: _EdgeGeometry,
    velocity_t: Tensor,
    acceleration_t: Tensor,
    torque_t: Tensor,
    jerk_t: Tensor,
    torque_rate_t: Tensor,
    tolerance: float,
    rate_mode: str = "finite_difference",
) -> tuple[np.ndarray, np.ndarray, int, int]:
    ext = get_kaserer_dp3_fast_extension()
    cost_t, slope_t, checked_t, feasible_t = ext.terminal_stage(
        torch.from_numpy(np.ascontiguousarray(current_z)),
        float(z_end),
        float(ds),
        eta_t,
        torch.from_numpy(np.ascontiguousarray(geometry.q_s)),
        torch.from_numpy(np.ascontiguousarray(geometry.q_ss)),
        torch.from_numpy(np.ascontiguousarray(geometry.q_sss)),
        torch.from_numpy(np.ascontiguousarray(geometry.m)),
        torch.from_numpy(np.ascontiguousarray(geometry.c)),
        torch.from_numpy(np.ascontiguousarray(geometry.gravity)),
        velocity_t,
        acceleration_t,
        torque_t,
        jerk_t,
        torque_rate_t,
        float(tolerance),
        *_native_rate_tensors(geometry),
        rate_mode == "analytic",
    )
    return (
        cost_t.numpy(),
        slope_t.numpy(),
        int(checked_t.sum().item()),
        int(feasible_t.sum().item()),
    )


def _search_ranges_vectorized(
    current_z: np.ndarray,
    next_z: np.ndarray,
    *,
    ds: float,
    q_s: np.ndarray,
    q_ss: np.ndarray,
    m: np.ndarray,
    c: np.ndarray,
    gravity: np.ndarray,
    acceleration: np.ndarray,
    torque: np.ndarray,
    eps: float = 1.0e-14,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized paper Eq. (21/22) successor-index ranges.

    This is algebraically identical to applying `_zprime_interval` followed
    by `_search_indices` for each current z state, but removes the Python
    state-by-joint loop from every DP stage.
    """

    z = np.asarray(current_z, dtype=np.float64)
    lower = np.full(z.shape, -np.inf, dtype=np.float64)
    upper = np.full(z.shape, np.inf, dtype=np.float64)
    valid = np.isfinite(z)

    def apply(
        coefficient: float,
        constant: np.ndarray,
        limit: float,
    ) -> None:
        nonlocal lower, upper, valid
        if abs(coefficient) <= eps:
            valid &= np.abs(constant) <= limit
            return
        a = (-limit - constant) / coefficient
        b = (limit - constant) / coefficient
        lo = np.minimum(a, b)
        hi = np.maximum(a, b)
        lower = np.maximum(lower, lo)
        upper = np.minimum(upper, hi)
        valid &= lower <= upper

    for joint in range(q_s.size):
        apply(
            0.5 * float(q_s[joint]),
            float(q_ss[joint]) * z,
            float(acceleration[joint]),
        )
        apply(
            0.5 * float(m[joint]),
            float(c[joint]) * z + float(gravity[joint]),
            float(torque[joint]),
        )

    lower_z = z + lower * ds
    upper_z = z + upper * ds
    search_lower = np.minimum(lower_z, upper_z)
    search_upper = np.maximum(lower_z, upper_z)

    lo = np.searchsorted(next_z, search_lower, side="left").astype(np.int64)
    hi = (
        np.searchsorted(next_z, search_upper, side="right") - 1
    ).astype(np.int64)
    lo = np.maximum(lo, 0)
    hi = np.minimum(hi, next_z.size - 1)

    valid &= np.isfinite(search_lower) | np.isneginf(search_lower)
    valid &= np.isfinite(search_upper) | np.isposinf(search_upper)
    valid &= lo <= hi
    # z=0 is not a valid ordinary C3 left state; preserve the scalar loop.
    valid &= z > 0.0

    lo = np.where(valid, lo, -1).astype(np.int64, copy=False)
    hi = np.where(valid, hi, -1).astype(np.int64, copy=False)
    return np.ascontiguousarray(lo), np.ascontiguousarray(hi)


def _c3_stage_native(
    current_z: np.ndarray,
    next_stage: _Stage,
    *,
    ds: float,
    node_q_s: np.ndarray,
    node_q_ss: np.ndarray,
    node_m: np.ndarray,
    node_c: np.ndarray,
    node_gravity: np.ndarray,
    acceleration: np.ndarray,
    torque: np.ndarray,
    eta_t: Tensor,
    geometry: _EdgeGeometry,
    quad_eta_t: Tensor,
    quad_weight_t: Tensor,
    velocity_t: Tensor,
    acceleration_t: Tensor,
    torque_t: Tensor,
    jerk_t: Tensor,
    torque_rate_t: Tensor,
    tolerance: float,
    rate_mode: str = "finite_difference",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int, float, float]:
    """Compiled Algorithm-2 stage with identical C3 semantics."""

    ext = get_kaserer_dp3_fast_extension()
    range_started = time.perf_counter()
    search_lo_t, search_hi_t = ext.search_ranges(
        torch.from_numpy(np.ascontiguousarray(current_z)),
        torch.from_numpy(np.ascontiguousarray(next_stage.z_grid)),
        float(ds),
        torch.from_numpy(np.ascontiguousarray(node_q_s)),
        torch.from_numpy(np.ascontiguousarray(node_q_ss)),
        torch.from_numpy(np.ascontiguousarray(node_m)),
        torch.from_numpy(np.ascontiguousarray(node_c)),
        torch.from_numpy(np.ascontiguousarray(node_gravity)),
        torch.from_numpy(np.ascontiguousarray(acceleration)),
        torch.from_numpy(np.ascontiguousarray(torque)),
    )
    search_lo = search_lo_t.numpy()
    search_hi = search_hi_t.numpy()
    range_ms = (time.perf_counter() - range_started) * 1.0e3

    kernel_started = time.perf_counter()
    cost_t, successor_t, slope_t, checked_t, feasible_t = ext.c3_stage(
        torch.from_numpy(np.ascontiguousarray(current_z)),
        torch.from_numpy(np.ascontiguousarray(next_stage.z_grid)),
        torch.from_numpy(np.ascontiguousarray(next_stage.cost)),
        torch.from_numpy(np.ascontiguousarray(next_stage.slope)),
        torch.from_numpy(search_lo),
        torch.from_numpy(search_hi),
        float(ds),
        eta_t,
        torch.from_numpy(np.ascontiguousarray(geometry.q_s)),
        torch.from_numpy(np.ascontiguousarray(geometry.q_ss)),
        torch.from_numpy(np.ascontiguousarray(geometry.q_sss)),
        torch.from_numpy(np.ascontiguousarray(geometry.m)),
        torch.from_numpy(np.ascontiguousarray(geometry.c)),
        torch.from_numpy(np.ascontiguousarray(geometry.gravity)),
        quad_eta_t,
        quad_weight_t,
        velocity_t,
        acceleration_t,
        torque_t,
        jerk_t,
        torque_rate_t,
        float(tolerance),
        *_native_rate_tensors(geometry),
        rate_mode == "analytic",
    )
    kernel_ms = (time.perf_counter() - kernel_started) * 1.0e3
    return (
        cost_t.numpy(),
        successor_t.numpy(),
        slope_t.numpy(),
        int(checked_t.sum().item()),
        int(feasible_t.sum().item()),
        range_ms,
        kernel_ms,
    )


def _start_stage_native(
    node2: np.ndarray,
    stage3: _Stage,
    *,
    first_range: tuple[int, int] | None,
    z_start: float,
    ds0: float,
    ds1: float,
    node_q_s: np.ndarray,
    node_q_ss: np.ndarray,
    node_m: np.ndarray,
    node_c: np.ndarray,
    node_gravity: np.ndarray,
    acceleration: np.ndarray,
    torque: np.ndarray,
    eta0_t: Tensor,
    eta1_t: Tensor,
    edge0: _EdgeGeometry,
    edge1: _EdgeGeometry,
    quad_eta_t: Tensor,
    quad_weight_t: Tensor,
    velocity_t: Tensor,
    acceleration_t: Tensor,
    torque_t: Tensor,
    jerk_t: Tensor,
    torque_rate_t: Tensor,
    tolerance: float,
    rate_mode: str = "finite_difference",
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int
]:
    """Compiled Algorithm-3 C2/C4 search with exact Python ranges."""

    first_valid = np.zeros(node2.size, dtype=np.int64)
    second_lo = np.full(node2.size, -1, dtype=np.int64)
    second_hi = np.full(node2.size, -1, dtype=np.int64)
    if first_range is not None:
        first_valid[first_range[0] : first_range[1] + 1] = 1
        for j in range(first_range[0], first_range[1] + 1):
            z2 = float(node2[j])
            if z2 <= 0.0:
                continue
            lo = -float("inf")
            hi = float("inf")
            valid = True
            for joint in range(node_q_s.size):
                interval = _linear_abs_interval(
                    0.5 * float(node_q_s[joint]),
                    float(node_q_ss[joint]) * z2,
                    float(acceleration[joint]),
                    lo,
                    hi,
                )
                if interval is None:
                    valid = False
                    break
                lo, hi = interval
                interval = _linear_abs_interval(
                    0.5 * float(node_m[joint]),
                    float(node_c[joint]) * z2 + float(node_gravity[joint]),
                    float(torque[joint]),
                    lo,
                    hi,
                )
                if interval is None:
                    valid = False
                    break
                lo, hi = interval
            if not valid:
                continue
            search = _search_indices(
                stage3.z_grid,
                z2 + lo * ds1,
                z2 + hi * ds1,
            )
            if search is not None:
                second_lo[j], second_hi[j] = search

    ext = get_kaserer_dp3_fast_extension()
    (
        cost_t,
        successor_t,
        slope_t,
        first_cost_t,
        checked_t,
        feasible_t,
    ) = ext.start_stage(
        float(z_start),
        torch.from_numpy(np.ascontiguousarray(node2)),
        torch.from_numpy(np.ascontiguousarray(stage3.z_grid)),
        torch.from_numpy(np.ascontiguousarray(stage3.cost)),
        torch.from_numpy(np.ascontiguousarray(stage3.slope)),
        torch.from_numpy(first_valid),
        torch.from_numpy(second_lo),
        torch.from_numpy(second_hi),
        float(ds0),
        float(ds1),
        eta0_t,
        eta1_t,
        torch.from_numpy(np.ascontiguousarray(edge0.q_s)),
        torch.from_numpy(np.ascontiguousarray(edge0.q_ss)),
        torch.from_numpy(np.ascontiguousarray(edge0.q_sss)),
        torch.from_numpy(np.ascontiguousarray(edge0.m)),
        torch.from_numpy(np.ascontiguousarray(edge0.c)),
        torch.from_numpy(np.ascontiguousarray(edge0.gravity)),
        torch.from_numpy(np.ascontiguousarray(edge1.q_s)),
        torch.from_numpy(np.ascontiguousarray(edge1.q_ss)),
        torch.from_numpy(np.ascontiguousarray(edge1.q_sss)),
        torch.from_numpy(np.ascontiguousarray(edge1.m)),
        torch.from_numpy(np.ascontiguousarray(edge1.c)),
        torch.from_numpy(np.ascontiguousarray(edge1.gravity)),
        quad_eta_t,
        quad_weight_t,
        velocity_t,
        acceleration_t,
        torque_t,
        jerk_t,
        torque_rate_t,
        float(tolerance),
        *_native_rate_tensors(edge0),
        *_native_rate_tensors(edge1),
        rate_mode == "analytic",
    )
    return (
        cost_t.numpy(),
        successor_t.numpy(),
        slope_t.numpy(),
        first_cost_t.numpy(),
        int(checked_t.sum().item()),
        int(feasible_t.sum().item()),
    )


def _largest_feasible_c2_node_value(
    cap: float,
    *,
    left_fixed: float | None,
    right_fixed: float | None,
    ds: float,
    geometry: _EdgeGeometry,
    velocity: np.ndarray,
    acceleration: np.ndarray,
    torque: np.ndarray,
    jerk: np.ndarray,
    torque_rate: np.ndarray,
    tolerance: float,
    probes: int = 129,
) -> float:
    """Largest feasible free endpoint value for one C2 interval.

    Exactly one of left_fixed/right_fixed must be supplied.  A modest uniform
    scan finds the highest feasible interval, then bisection refines its upper
    boundary.  This is used only by the optional endpoint-grid enrichment; the
    paper-faithful equidistant grid remains unchanged by default.
    """

    if (left_fixed is None) == (right_fixed is None):
        raise ValueError("exactly one C2 endpoint must be fixed")
    if not (cap > 0.0) or not math.isfinite(cap):
        return 0.0

    def feasible(value: float) -> bool:
        if value <= 0.0 or not math.isfinite(value):
            return False
        z0 = float(left_fixed) if left_fixed is not None else value
        z1 = float(right_fixed) if right_fixed is not None else value
        coeff = _c2(z0, z1)
        return _edge_feasible(
            0,
            "c2",
            coeff,
            ds,
            geometry,
            eta=geometry.eta,
            velocity=velocity,
            acceleration=acceleration,
            torque=torque,
            jerk=jerk,
            torque_rate=torque_rate,
            tolerance=tolerance,
        )

    probe_count = max(33, int(probes))
    tiny = max(cap * 1.0e-12, 1.0e-16)
    probe_values = np.unique(
        np.concatenate(
            (
                np.geomspace(tiny, cap, 2 * probe_count),
                np.linspace(
                    0.0,
                    cap,
                    probe_count + 1,
                    dtype=np.float64,
                )[1:],
            )
        )
    )
    mask = np.asarray(
        [feasible(float(value)) for value in probe_values],
        dtype=bool,
    )
    if not np.any(mask):
        return 0.0

    index = int(np.flatnonzero(mask)[-1])
    lo = float(probe_values[index])
    if index == probe_values.size - 1:
        return float(np.nextafter(lo, 0.0))

    hi = float(probe_values[index + 1])
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if feasible(mid):
            lo = mid
        else:
            hi = mid
    return float(np.nextafter(lo, 0.0))


def _insert_sorted_unique_value(
    grid: np.ndarray,
    value: float,
) -> tuple[np.ndarray, bool]:
    if not (value > 0.0) or not math.isfinite(value):
        return grid, False
    pos = int(np.searchsorted(grid, value))
    scale = max(1.0, abs(value))
    atol = 32.0 * np.finfo(np.float64).eps * scale
    if pos < grid.size and abs(float(grid[pos]) - value) <= atol:
        return grid, False
    if pos > 0 and abs(float(grid[pos - 1]) - value) <= atol:
        return grid, False
    return np.insert(grid, pos, value).astype(np.float64, copy=False), True


def _endpoint_enrichment_values(
    maximum: float,
    count: int,
    *,
    left_fixed: float | None,
    right_fixed: float | None,
    ds: float,
    geometry: _EdgeGeometry,
    velocity: np.ndarray,
    acceleration: np.ndarray,
    torque: np.ndarray,
    jerk: np.ndarray,
    torque_rate: np.ndarray,
    tolerance: float,
) -> np.ndarray:
    """Return a small feasible C2 endpoint-state ladder up to maximum."""

    if count < 1:
        raise ValueError("endpoint enrichment count must be at least 1")
    if not (maximum > 0.0) or not math.isfinite(maximum):
        return np.empty(0, dtype=np.float64)
    if (left_fixed is None) == (right_fixed is None):
        raise ValueError("exactly one C2 endpoint must be fixed")

    candidate = maximum * (
        np.arange(1, count + 1, dtype=np.float64) / float(count)
    )
    feasible: list[float] = []
    for value in candidate:
        z0 = float(left_fixed) if left_fixed is not None else float(value)
        z1 = float(right_fixed) if right_fixed is not None else float(value)
        coeff = _c2(z0, z1)
        if _edge_feasible(
            0,
            "c2",
            coeff,
            ds,
            geometry,
            eta=geometry.eta,
            velocity=velocity,
            acceleration=acceleration,
            torque=torque,
            jerk=jerk,
            torque_rate=torque_rate,
            tolerance=tolerance,
        ):
            feasible.append(float(value))
    return np.asarray(feasible, dtype=np.float64)


def _insert_sorted_unique_values(
    grid: np.ndarray,
    values: np.ndarray,
) -> tuple[np.ndarray, int]:
    out = grid
    inserted = 0
    for value in values:
        out, did_insert = _insert_sorted_unique_value(out, float(value))
        inserted += int(did_insert)
    return out, inserted

def _search_indices(
    z_grid_next: np.ndarray,
    lower: float,
    upper: float,
) -> tuple[int, int] | None:
    if upper < lower:
        lower, upper = upper, lower
    lo = int(np.searchsorted(z_grid_next, lower, side="left"))
    hi_exclusive = int(np.searchsorted(z_grid_next, upper, side="right"))
    lo = max(0, lo)
    hi = min(z_grid_next.size - 1, hi_exclusive - 1)
    if lo > hi:
        return None
    return lo, hi


def _select_dp_labels(
    candidates: list[tuple[float, int, int, float]],
    keep: int,
    *,
    policy: str,
) -> list[tuple[float, int, int, float]]:
    """Retain distinct feasible continuation labels for one (s,z) state."""
    if keep <= 0 or not candidates:
        return []
    ordered = sorted(candidates, key=lambda item: item[0])

    def distinct_from(
        candidate: tuple[float, int, int, float],
        selected: list[tuple[float, int, int, float]],
    ) -> bool:
        slope = float(candidate[3])
        return all(
            abs(slope - float(existing[3]))
            > 1.0e-10
            * max(1.0, abs(slope), abs(float(existing[3])))
            for existing in selected
        )

    cheapest: list[tuple[float, int, int, float]] = []
    for candidate in ordered:
        if distinct_from(candidate, cheapest):
            cheapest.append(candidate)
            if len(cheapest) >= keep:
                break
    if policy == "cost" or len(cheapest) <= 1:
        return cheapest

    # Build the distinct candidate pool once, then greedily maximize the
    # minimum slope distance to already retained labels. The globally cheapest
    # label is always retained first to preserve a cost anchor.
    pool: list[tuple[float, int, int, float]] = []
    for candidate in ordered:
        if distinct_from(candidate, pool):
            pool.append(candidate)
    selected = [pool[0]]
    remaining = pool[1:]
    while remaining and len(selected) < keep:
        best_index = max(
            range(len(remaining)),
            key=lambda idx: (
                min(
                    abs(
                        float(remaining[idx][3])
                        - float(existing[3])
                    )
                    for existing in selected
                ),
                -float(remaining[idx][0]),
            ),
        )
        selected.append(remaining.pop(best_index))
    return selected


def solve_kaserer_dp3(
    problem: RetimingProblem,
    config: KasererDP3Config | None = None,
) -> KasererDP3Result:
    """Solve the paper DP3 recursion for the repository's full-C3 problem."""

    _require_problem(problem)
    cfg = config or KasererDP3Config()
    if cfg.torque_rate_mode == "analytic":
        authority = getattr(problem.dynamics, "torque_rate_backend_name", None)
        if authority != "pinocchio_computeRNEADerivatives_contraction":
            raise RuntimeError(
                "Kaserer_fixed requires analytic Pinocchio RNEA derivatives; "
                f"received {authority!r}"
            )
    # Native extension compilation is a one-time setup cost, not solve latency.
    if cfg.use_native:
        get_kaserer_dp3_fast_extension()
    started = time.perf_counter()

    grid = torch.linspace(
        0.0,
        1.0,
        cfg.grid_samples,
        device=problem.device,
        dtype=problem.dtype,
    )
    s = grid.detach().cpu().double().numpy()
    N = s.size
    ds_all = np.diff(s)

    velocity = _limit(problem.limits.velocity, cfg.safety_scale)
    acceleration = _limit(problem.limits.acceleration, cfg.safety_scale)
    torque = _limit(problem.limits.torque, cfg.safety_scale)
    jerk = _limit(problem.limits.jerk, cfg.safety_scale)
    torque_rate = _limit(problem.limits.torque_rate, cfg.safety_scale)

    prep_started = time.perf_counter()
    q_s, q_ss, m, c, gravity, edge_geometry = _precompute_geometry(
        problem,
        grid,
        cfg.check_points,
        cfg.path_segment_check_points,
        cfg.endpoint_path_segment_check_points,
        analytic_rate=(cfg.torque_rate_mode == "analytic"),
    )

    z_grids: list[np.ndarray] = []
    z_caps = np.zeros(N, dtype=np.float64)
    z_start = float(problem.initial_speed) ** 2
    z_end = float(problem.final_speed) ** 2
    for k in range(N):
        if k == 0:
            z_caps[k] = z_start
            z_grids.append(np.asarray([z_start], dtype=np.float64))
            continue
        if k == N - 1:
            z_caps[k] = z_end
            z_grids.append(np.asarray([z_end], dtype=np.float64))
            continue
        cap = _node_z_cap(
            k,
            q_s=q_s,
            q_ss=q_ss,
            m=m,
            c=c,
            gravity=gravity,
            velocity=velocity,
            acceleration=acceleration,
            torque=torque,
        )
        z_caps[k] = cap
        z_grids.append(
            np.linspace(0.0, cap, cfg.z_samples, dtype=np.float64)
            if cap > 0.0
            else np.zeros(cfg.z_samples, dtype=np.float64)
        )

    endpoint_start_z = 0.0
    endpoint_terminal_z = 0.0
    endpoint_start_values = np.empty(0, dtype=np.float64)
    endpoint_terminal_values = np.empty(0, dtype=np.float64)
    endpoint_start_inserted = 0
    endpoint_terminal_inserted = 0
    enrichment_geometry = edge_geometry
    enrichment_path_points = (
        cfg.endpoint_z_enrichment_check_points
        if cfg.endpoint_z_enrichment_check_points > 0
        else cfg.endpoint_path_segment_check_points
    )
    if (
        cfg.endpoint_z_enrichment
        and cfg.endpoint_z_enrichment_check_points > 0
        and cfg.endpoint_z_enrichment_check_points
        != cfg.endpoint_path_segment_check_points
    ):
        # Deliberately use a stricter, independent local geometry for locating
        # the enrichment threshold. Ordinary transition checking still uses
        # edge_geometry and its configured endpoint density.
        (
            _,
            _,
            _,
            _,
            _,
            enrichment_geometry,
        ) = _precompute_geometry(
            problem,
            grid,
            cfg.check_points,
            cfg.path_segment_check_points,
            cfg.endpoint_z_enrichment_check_points,
            analytic_rate=(cfg.torque_rate_mode == "analytic"),
        )

    if cfg.endpoint_z_enrichment and N >= 4:
        endpoint_start_z = _largest_feasible_c2_node_value(
            float(z_caps[1]),
            left_fixed=z_start,
            right_fixed=None,
            ds=float(ds_all[0]),
            geometry=enrichment_geometry[0],
            velocity=velocity,
            acceleration=acceleration,
            torque=torque,
            jerk=jerk,
            torque_rate=torque_rate,
            tolerance=cfg.feasibility_tolerance,
        )
        endpoint_start_values = _endpoint_enrichment_values(
            endpoint_start_z,
            cfg.endpoint_z_enrichment_count,
            left_fixed=z_start,
            right_fixed=None,
            ds=float(ds_all[0]),
            geometry=enrichment_geometry[0],
            velocity=velocity,
            acceleration=acceleration,
            torque=torque,
            jerk=jerk,
            torque_rate=torque_rate,
            tolerance=cfg.feasibility_tolerance,
        )
        z_grids[1], endpoint_start_inserted = _insert_sorted_unique_values(
            z_grids[1],
            endpoint_start_values,
        )

        endpoint_terminal_z = _largest_feasible_c2_node_value(
            float(z_caps[N - 2]),
            left_fixed=None,
            right_fixed=z_end,
            ds=float(ds_all[N - 2]),
            geometry=enrichment_geometry[N - 2],
            velocity=velocity,
            acceleration=acceleration,
            torque=torque,
            jerk=jerk,
            torque_rate=torque_rate,
            tolerance=cfg.feasibility_tolerance,
        )
        endpoint_terminal_values = _endpoint_enrichment_values(
            endpoint_terminal_z,
            cfg.endpoint_z_enrichment_count,
            left_fixed=None,
            right_fixed=z_end,
            ds=float(ds_all[N - 2]),
            geometry=enrichment_geometry[N - 2],
            velocity=velocity,
            acceleration=acceleration,
            torque=torque,
            jerk=jerk,
            torque_rate=torque_rate,
            tolerance=cfg.feasibility_tolerance,
        )
        z_grids[N - 2], endpoint_terminal_inserted = (
            _insert_sorted_unique_values(
                z_grids[N - 2],
                endpoint_terminal_values,
            )
        )

    quad_eta, quad_weight = _quadrature_rule(cfg.quadrature_order)
    edge_eta_native: list[Tensor] | None = None
    quad_eta_native = quad_weight_native = None
    velocity_native = acceleration_native = torque_native = None
    jerk_native = torque_rate_native = None
    if cfg.use_native:
        edge_eta_native = [
            torch.from_numpy(np.ascontiguousarray(g.eta))
            for g in edge_geometry
        ]
        quad_eta_native = torch.from_numpy(np.ascontiguousarray(quad_eta.copy()))
        quad_weight_native = torch.from_numpy(
            np.ascontiguousarray(quad_weight.copy())
        )
        velocity_native = torch.from_numpy(np.ascontiguousarray(velocity))
        acceleration_native = torch.from_numpy(
            np.ascontiguousarray(acceleration)
        )
        torque_native = torch.from_numpy(np.ascontiguousarray(torque))
        jerk_native = torch.from_numpy(np.ascontiguousarray(jerk))
        torque_rate_native = torch.from_numpy(
            np.ascontiguousarray(torque_rate)
        )
    preparation_ms = (time.perf_counter() - prep_started) * 1.0e3

    stages: list[_Stage | None] = [None] * N
    label_stages: list[_LabelStage | None] = [None] * N
    stage_finite_counts = np.full(N, -1, dtype=np.int64)
    terminal = _Stage(
        z_grid=z_grids[-1],
        cost=np.asarray([0.0], dtype=np.float64),
        successor=np.asarray([-1], dtype=np.int64),
        slope=np.asarray([np.nan], dtype=np.float64),
    )
    stages[-1] = terminal
    stage_finite_counts[-1] = 1

    transitions_checked = 0
    feasible_transitions = 0
    finite_states = 0
    search_started = time.perf_counter()

    # Algorithm 1: final interval, C2 interpolation.
    algorithm1_started = time.perf_counter()
    k = N - 2
    zg = z_grids[k]
    ds = float(ds_all[k])
    if cfg.use_native:
        assert edge_eta_native is not None
        assert velocity_native is not None
        assert acceleration_native is not None
        assert torque_native is not None
        assert jerk_native is not None
        assert torque_rate_native is not None
        cost, slope, checked, feasible = _terminal_stage_native(
            zg,
            z_end=z_end,
            ds=ds,
            eta_t=edge_eta_native[k],
            geometry=edge_geometry[k],
            velocity_t=velocity_native,
            acceleration_t=acceleration_native,
            torque_t=torque_native,
            jerk_t=jerk_native,
            torque_rate_t=torque_rate_native,
            rate_mode=cfg.torque_rate_mode,
            tolerance=cfg.feasibility_tolerance,
        )
        successor = np.where(
            np.isfinite(cost),
            np.zeros(zg.size, dtype=np.int64),
            np.full(zg.size, -1, dtype=np.int64),
        )
        transitions_checked += checked
        feasible_transitions += feasible
    else:
        cost = np.full(zg.size, np.inf, dtype=np.float64)
        successor = np.full(zg.size, -1, dtype=np.int64)
        slope = np.full(zg.size, np.nan, dtype=np.float64)
        for j in range(1, zg.size):
            z0 = float(zg[j])
            coeff = _c2(z0, z_end)
            transitions_checked += 1
            if not _edge_feasible(
                k,
                "c2",
                coeff,
                ds,
                edge_geometry[k],
                eta=edge_geometry[k].eta,
                velocity=velocity,
                acceleration=acceleration,
                torque=torque,
                jerk=jerk,
                torque_rate=torque_rate,
                tolerance=cfg.feasibility_tolerance,
            ):
                continue
            edge_cost = _edge_time("c2", coeff, ds, cfg.quadrature_order)
            if not math.isfinite(edge_cost):
                continue
            feasible_transitions += 1
            cost[j] = edge_cost
            successor[j] = 0
            slope[j] = (z_end - z0) / ds
    stages[k] = _Stage(zg, cost, successor, slope)
    if cfg.dp_labels_per_state > 1:
        M = int(cfg.dp_labels_per_state)
        label_cost = np.full((zg.size, M), np.inf, dtype=np.float64)
        label_successor = np.full((zg.size, M), -1, dtype=np.int64)
        label_successor_label = np.full(
            (zg.size, M), -1, dtype=np.int64
        )
        label_slope = np.full((zg.size, M), np.nan, dtype=np.float64)
        finite_terminal = np.flatnonzero(np.isfinite(cost))
        label_cost[finite_terminal, 0] = cost[finite_terminal]
        label_successor[finite_terminal, 0] = successor[finite_terminal]
        label_successor_label[finite_terminal, 0] = 0
        label_slope[finite_terminal, 0] = slope[finite_terminal]
        label_stages[k] = _LabelStage(
            zg,
            label_cost,
            label_successor,
            label_successor_label,
            label_slope,
        )
    stage_finite_counts[k] = int(np.count_nonzero(np.isfinite(cost)))
    finite_states += int(stage_finite_counts[k])
    algorithm1_ms = (time.perf_counter() - algorithm1_started) * 1.0e3

    # Algorithm 2: ordinary backward recursion, C3 interpolation.
    algorithm2_started = time.perf_counter()
    algorithm2_range_ms = 0.0
    algorithm2_kernel_ms = 0.0
    stage3_label_bank: dict[
        int, list[tuple[float, int, float]]
    ] = {}
    for k in range(N - 3, 1, -1):
        zg = z_grids[k]
        next_stage = stages[k + 1]
        assert next_stage is not None
        ds = float(ds_all[k])

        if cfg.use_native and cfg.dp_labels_per_state == 1:
            assert edge_eta_native is not None
            assert quad_eta_native is not None
            assert quad_weight_native is not None
            assert velocity_native is not None
            assert acceleration_native is not None
            assert torque_native is not None
            assert jerk_native is not None
            assert torque_rate_native is not None
            (
                cost,
                successor,
                slope,
                checked,
                feasible,
                stage_range_ms,
                stage_kernel_ms,
            ) = _c3_stage_native(
                zg,
                next_stage,
                ds=ds,
                node_q_s=q_s[k],
                node_q_ss=q_ss[k],
                node_m=m[k],
                node_c=c[k],
                node_gravity=gravity[k],
                acceleration=acceleration,
                torque=torque,
                eta_t=edge_eta_native[k],
                geometry=edge_geometry[k],
                quad_eta_t=quad_eta_native,
                quad_weight_t=quad_weight_native,
                velocity_t=velocity_native,
                acceleration_t=acceleration_native,
                torque_t=torque_native,
                jerk_t=jerk_native,
                torque_rate_t=torque_rate_native,
                rate_mode=cfg.torque_rate_mode,
                tolerance=cfg.feasibility_tolerance,
            )
            transitions_checked += checked
            feasible_transitions += feasible
            algorithm2_range_ms += stage_range_ms
            algorithm2_kernel_ms += stage_kernel_ms
        else:
            cost = np.full(zg.size, np.inf, dtype=np.float64)
            successor = np.full(zg.size, -1, dtype=np.int64)
            slope = np.full(zg.size, np.nan, dtype=np.float64)
            M = int(cfg.dp_labels_per_state)
            label_cost = np.full((zg.size, M), np.inf, dtype=np.float64)
            label_successor = np.full((zg.size, M), -1, dtype=np.int64)
            label_successor_label = np.full(
                (zg.size, M), -1, dtype=np.int64
            )
            label_slope = np.full((zg.size, M), np.nan, dtype=np.float64)
            next_labels = (
                label_stages[k + 1]
                if cfg.dp_labels_per_state > 1
                else None
            )

            for j in range(1, zg.size):
                z0 = float(zg[j])
                bounds = _zprime_interval(
                    k,
                    z0,
                    q_s=q_s,
                    q_ss=q_ss,
                    m=m,
                    c=c,
                    gravity=gravity,
                    acceleration=acceleration,
                    torque=torque,
                )
                if bounds is None:
                    continue
                lower_z = z0 + bounds[0] * ds
                upper_z = z0 + bounds[1] * ds
                search = _search_indices(next_stage.z_grid, lower_z, upper_z)
                if search is None:
                    continue

                candidates: list[
                    tuple[float, int, int, float]
                ] = []
                for l in range(search[0], search[1] + 1):
                    z1 = float(next_stage.z_grid[l])
                    if z1 <= 0.0:
                        continue
                    if next_labels is None:
                        successor_options = [
                            (
                                0,
                                float(next_stage.cost[l]),
                                float(next_stage.slope[l]),
                            )
                        ]
                    else:
                        successor_options = [
                            (
                                int(rank),
                                float(next_labels.cost[l, rank]),
                                float(next_labels.slope[l, rank]),
                            )
                            for rank in range(M)
                        ]
                    for next_rank, next_cost, right_slope in successor_options:
                        if not (
                            math.isfinite(next_cost)
                            and math.isfinite(right_slope)
                        ):
                            continue
                        coeff = _c3(z0, z1, right_slope, ds)
                        if coeff is None:
                            continue
                        transitions_checked += 1
                        if not _edge_feasible(
                            k,
                            "c3",
                            coeff,
                            ds,
                            edge_geometry[k],
                            eta=edge_geometry[k].eta,
                            velocity=velocity,
                            acceleration=acceleration,
                            torque=torque,
                            jerk=jerk,
                            torque_rate=torque_rate,
                            tolerance=cfg.feasibility_tolerance,
                        ):
                            continue
                        edge_cost = _edge_time(
                            "c3", coeff, ds, cfg.quadrature_order
                        )
                        if not math.isfinite(edge_cost):
                            continue
                        total = edge_cost + next_cost
                        left_slope = _left_slope("c3", coeff, ds)
                        feasible_transitions += 1
                        candidates.append(
                            (
                                float(total),
                                int(l),
                                int(next_rank),
                                float(left_slope),
                            )
                        )

                keep = (
                    M
                    if cfg.dp_labels_per_state > 1
                    else (
                        int(cfg.algorithm3_stage3_labels)
                        if k == 2
                        else 1
                    )
                )
                labels = _select_dp_labels(
                    candidates,
                    keep,
                    policy=(
                        cfg.dp_label_retention
                        if cfg.dp_labels_per_state > 1
                        else "cost"
                    ),
                )

                if not labels:
                    continue
                best, best_l, _best_next_rank, best_slope = labels[0]
                cost[j] = best
                successor[j] = best_l
                slope[j] = best_slope
                if cfg.dp_labels_per_state > 1:
                    for rank, (
                        total,
                        succ,
                        succ_rank,
                        left_slope,
                    ) in enumerate(labels):
                        label_cost[j, rank] = total
                        label_successor[j, rank] = succ
                        label_successor_label[j, rank] = succ_rank
                        label_slope[j, rank] = left_slope
                if k == 2 and cfg.algorithm3_stage3_labels > 1:
                    stage3_label_bank[int(j)] = [
                        (float(total), int(succ), float(left_slope))
                        for total, succ, _succ_rank, left_slope in labels
                    ]

            if cfg.dp_labels_per_state > 1:
                label_stages[k] = _LabelStage(
                    zg,
                    label_cost,
                    label_successor,
                    label_successor_label,
                    label_slope,
                )

        stages[k] = _Stage(zg, cost, successor, slope)
        stage_finite_counts[k] = int(np.count_nonzero(np.isfinite(cost)))
        finite_states += int(stage_finite_counts[k])
    algorithm2_ms = (time.perf_counter() - algorithm2_started) * 1.0e3

    # Algorithm 3: special treatment of intervals [s1,s2] and [s2,s3].
    algorithm3_started = time.perf_counter()
    node2 = z_grids[1]
    stage3 = stages[2]
    assert stage3 is not None
    start_bounds = _zprime_interval(
        0,
        z_start,
        q_s=q_s,
        q_ss=q_ss,
        m=m,
        c=c,
        gravity=gravity,
        acceleration=acceleration,
        torque=torque,
    )
    if start_bounds is None:
        search_ms = (time.perf_counter() - search_started) * 1.0e3
        return KasererDP3Result(
            False,
            grid,
            torch.full_like(grid, torch.nan),
            torch.full_like(grid, torch.nan),
            float("inf"),
            (time.perf_counter() - started) * 1.0e3,
            preparation_ms,
            search_ms,
            "no feasible initial z' interval",
            {"z_caps": z_caps.tolist()},
        )

    first_range = _search_indices(
        node2,
        z_start + start_bounds[0] * float(ds_all[0]),
        z_start + start_bounds[1] * float(ds_all[0]),
    )
    best_total = float("inf")
    best_j = -1
    best_l = -1
    best_stage3_label = 0
    best_start_slope = float("nan")
    stage2_cost = np.full(node2.size, np.inf, dtype=np.float64)
    stage2_successor = np.full(node2.size, -1, dtype=np.int64)
    stage2_slope = np.full(node2.size, np.nan, dtype=np.float64)
    first_cost = np.full(node2.size, np.inf, dtype=np.float64)

    if cfg.use_native and cfg.dp_labels_per_state == 1:
        assert edge_eta_native is not None
        assert quad_eta_native is not None
        assert quad_weight_native is not None
        assert velocity_native is not None
        assert acceleration_native is not None
        assert torque_native is not None
        assert jerk_native is not None
        assert torque_rate_native is not None
        (
            stage2_cost,
            stage2_successor,
            stage2_slope,
            first_cost,
            checked,
            feasible,
        ) = _start_stage_native(
            node2,
            stage3,
            first_range=first_range,
            z_start=z_start,
            ds0=float(ds_all[0]),
            ds1=float(ds_all[1]),
            node_q_s=q_s[1],
            node_q_ss=q_ss[1],
            node_m=m[1],
            node_c=c[1],
            node_gravity=gravity[1],
            acceleration=acceleration,
            torque=torque,
            eta0_t=edge_eta_native[0],
            eta1_t=edge_eta_native[1],
            edge0=edge_geometry[0],
            edge1=edge_geometry[1],
            quad_eta_t=quad_eta_native,
            quad_weight_t=quad_weight_native,
            velocity_t=velocity_native,
            acceleration_t=acceleration_native,
            torque_t=torque_native,
            jerk_t=jerk_native,
            torque_rate_t=torque_rate_native,
                rate_mode=cfg.torque_rate_mode,
            tolerance=cfg.feasibility_tolerance,
        )
        transitions_checked += checked
        feasible_transitions += feasible
        if first_range is not None:
            for j in range(first_range[0], first_range[1] + 1):
                if stage2_successor[j] < 0:
                    continue
                total = float(first_cost[j]) + float(stage2_cost[j])
                if total < best_total:
                    best_total = total
                    best_j = j
                    best_l = int(stage2_successor[j])
                    best_start_slope = float(stage2_slope[j])
    elif first_range is not None:
        for j in range(first_range[0], first_range[1] + 1):
            z2 = float(node2[j])
            c2 = _c2(z_start, z2)
            transitions_checked += 1
            if not _edge_feasible(
                0,
                "c2",
                c2,
                float(ds_all[0]),
                edge_geometry[0],
                eta=edge_geometry[0].eta,
                velocity=velocity,
                acceleration=acceleration,
                torque=torque,
                jerk=jerk,
                torque_rate=torque_rate,
                tolerance=cfg.feasibility_tolerance,
            ):
                continue
            edge1 = _edge_time(
                "c2", c2, float(ds_all[0]), cfg.quadrature_order
            )
            if not math.isfinite(edge1):
                continue
            first_cost[j] = edge1
            feasible_transitions += 1
            left_slope_s2 = (z2 - z_start) / float(ds_all[0])
            bounds2 = _zprime_interval(
                1,
                z2,
                q_s=q_s,
                q_ss=q_ss,
                m=m,
                c=c,
                gravity=gravity,
                acceleration=acceleration,
                torque=torque,
            )
            if bounds2 is None or z2 <= 0.0:
                continue
            second_range = _search_indices(
                stage3.z_grid,
                z2 + bounds2[0] * float(ds_all[1]),
                z2 + bounds2[1] * float(ds_all[1]),
            )
            if second_range is None:
                continue

            best_j_total = float("inf")
            best_j_l = -1
            best_j_label = 0
            stage3_labels = (
                label_stages[2]
                if cfg.dp_labels_per_state > 1
                else None
            )
            for l in range(second_range[0], second_range[1] + 1):
                z3 = float(stage3.z_grid[l])
                if z3 <= 0.0:
                    continue
                if stage3_labels is None:
                    label_options = [
                        (
                            0,
                            float(stage3.cost[l]),
                            float(stage3.slope[l]),
                        )
                    ]
                else:
                    label_options = [
                        (
                            rank,
                            float(stage3_labels.cost[l, rank]),
                            float(stage3_labels.slope[l, rank]),
                        )
                        for rank in range(int(cfg.dp_labels_per_state))
                    ]
                for label_rank, stage3_cost, right_slope in label_options:
                    if not (
                        math.isfinite(stage3_cost)
                        and math.isfinite(right_slope)
                    ):
                        continue
                    c4 = _c4(
                        z2,
                        z3,
                        left_slope_s2,
                        right_slope,
                        float(ds_all[1]),
                    )
                    if c4 is None:
                        continue
                    transitions_checked += 1
                    if not _edge_feasible(
                        1,
                        "c4",
                        c4,
                        float(ds_all[1]),
                        edge_geometry[1],
                        eta=edge_geometry[1].eta,
                        velocity=velocity,
                        acceleration=acceleration,
                        torque=torque,
                        jerk=jerk,
                        torque_rate=torque_rate,
                        tolerance=cfg.feasibility_tolerance,
                    ):
                        continue
                    edge2 = _edge_time(
                        "c4", c4, float(ds_all[1]), cfg.quadrature_order
                    )
                    if not math.isfinite(edge2):
                        continue
                    total2 = edge2 + stage3_cost
                    feasible_transitions += 1
                    if total2 < best_j_total:
                        best_j_total = total2
                        best_j_l = l
                        best_j_label = int(label_rank)

            if best_j_l >= 0:
                stage2_cost[j] = best_j_total
                stage2_successor[j] = best_j_l
                stage2_slope[j] = left_slope_s2
                total = edge1 + best_j_total
                if total < best_total:
                    best_total = total
                    best_j = j
                    best_l = best_j_l
                    best_stage3_label = best_j_label
                    best_start_slope = left_slope_s2

    stages[1] = _Stage(
        node2,
        stage2_cost,
        stage2_successor,
        stage2_slope,
    )
    stage_finite_counts[1] = int(
        np.count_nonzero(np.isfinite(stage2_cost))
    )
    finite_states += int(stage_finite_counts[1])
    algorithm3_ms = (time.perf_counter() - algorithm3_started) * 1.0e3
    search_ms = (time.perf_counter() - search_started) * 1.0e3

    if best_j < 0 or best_l < 0 or not math.isfinite(best_total):
        first_range_size = (
            first_range[1] - first_range[0] + 1
            if first_range is not None
            else 0
        )
        c2_feasible_indices = np.flatnonzero(np.isfinite(first_cost))
        second_range_states = 0
        second_range_pairs = 0
        finite_stage3_pairs = 0
        finite_stage3_unique: set[int] = set()
        for j in c2_feasible_indices:
            z2 = float(node2[j])
            if z2 <= 0.0:
                continue
            bounds2 = _zprime_interval(
                1,
                z2,
                q_s=q_s,
                q_ss=q_ss,
                m=m,
                c=c,
                gravity=gravity,
                acceleration=acceleration,
                torque=torque,
            )
            if bounds2 is None:
                continue
            second_range = _search_indices(
                stage3.z_grid,
                z2 + bounds2[0] * float(ds_all[1]),
                z2 + bounds2[1] * float(ds_all[1]),
            )
            if second_range is None:
                continue
            second_range_states += 1
            lo, hi = second_range
            second_range_pairs += hi - lo + 1
            finite = np.flatnonzero(
                np.isfinite(stage3.cost[lo : hi + 1])
                & np.isfinite(stage3.slope[lo : hi + 1])
                & (stage3.z_grid[lo : hi + 1] > 0.0)
            )
            finite_stage3_pairs += int(finite.size)
            finite_stage3_unique.update(
                int(lo + idx) for idx in finite.tolist()
            )

        c2_values = node2[c2_feasible_indices]

        # K2 diagnostic: keep the paper's downstream DP state and stored
        # stage-3 slope exactly fixed, but sample z2 continuously inside the
        # analytically reachable first-interval range. This does not change
        # solver success; it only asks whether the uniform z2 lattice missed a
        # C2/C4 bridge into an already retained stage-3 continuation.
        local_refinement = {
            "enabled": bool(cfg.algorithm3_local_refinement_points > 0),
            "points": int(cfg.algorithm3_local_refinement_points),
            "candidate_interval": None,
            "candidates_tested": 0,
            "first_interval_feasible": 0,
            "second_range_nonempty": 0,
            "finite_stage3_pairs": 0,
            "c4_polynomial_valid_pairs": 0,
            "c4_physical_feasible_pairs": 0,
            "free_slope_scan_points": int(
                cfg.algorithm3_free_slope_scan_points
            ),
            "free_slope_bounded_pairs": 0,
            "free_slope_unbounded_pairs": 0,
            "free_slope_candidates_tested": 0,
            "free_slope_physical_feasible": 0,
            "free_slope_pairs_with_physical": 0,
            "best_free_slope": float("nan"),
            "best_free_slope_z2": float("nan"),
            "best_free_slope_z3": float("nan"),
            "stage3_labels_requested": int(
                cfg.algorithm3_stage3_labels
            ),
            "stage3_label_pairs_tested": 0,
            "stage3_alt_label_pairs_tested": 0,
            "stage3_states_with_alternatives": int(
                sum(
                    len(labels) > 1
                    for labels in stage3_label_bank.values()
                )
            ),
            "bridge_found": False,
            "best_z2": float("nan"),
            "best_z3": float("nan"),
            "best_stage3_index": -1,
            "best_total_duration": float("inf"),
        }
        if cfg.algorithm3_local_refinement_points > 0:
            ds0 = float(ds_all[0])
            ds1 = float(ds_all[1])
            lower = max(0.0, z_start + start_bounds[0] * ds0)
            upper = min(float(z_caps[1]), z_start + start_bounds[1] * ds0)
            local_refinement["candidate_interval"] = [
                float(lower),
                float(upper),
            ]
            seed_values = np.asarray(
                [float(v) for v in c2_values if float(v) > 0.0],
                dtype=np.float64,
            )
            local_refinement["original_c2_seed_count"] = int(
                seed_values.size
            )
            local_refinement["original_c2_seed_values"] = (
                seed_values.tolist()
            )
            local_refinement[
                "original_c2_seed_python_feasible"
            ] = 0
            if upper > lower:
                candidate_parts = [
                    np.linspace(
                        lower,
                        upper,
                        int(cfg.algorithm3_local_refinement_points),
                        dtype=np.float64,
                    )
                ]
                if seed_values.size:
                    candidate_parts.append(seed_values)
                    for seed in seed_values:
                        index = int(np.searchsorted(node2, seed))
                        left = (
                            float(node2[index - 1])
                            if index > 0
                            else float(lower)
                        )
                        right = (
                            float(node2[index + 1])
                            if index + 1 < node2.size
                            else float(upper)
                        )
                        positive_spans = [
                            span
                            for span in (
                                float(seed) - left,
                                right - float(seed),
                            )
                            if span > 0.0 and math.isfinite(span)
                        ]
                        if positive_spans:
                            span = min(positive_spans)
                            local_lo = max(
                                lower, float(seed) - span
                            )
                            local_hi = min(
                                upper, float(seed) + span
                            )
                            if local_hi > local_lo:
                                candidate_parts.append(
                                    np.linspace(
                                        local_lo,
                                        local_hi,
                                        int(
                                            cfg.algorithm3_local_refinement_points
                                        ),
                                        dtype=np.float64,
                                    )
                                )
                candidates = np.unique(
                    np.concatenate(candidate_parts)
                )
                local_refinement["unique_candidates"] = int(
                    candidates.size
                )
                best_local = float("inf")
                for z2_value in candidates:
                    z2 = float(z2_value)
                    local_refinement["candidates_tested"] += 1
                    if z2 <= 0.0:
                        continue
                    c2 = _c2(z_start, z2)
                    c2_feasible_python = _edge_feasible(
                        0,
                        "c2",
                        c2,
                        ds0,
                        edge_geometry[0],
                        eta=edge_geometry[0].eta,
                        velocity=velocity,
                        acceleration=acceleration,
                        torque=torque,
                        jerk=jerk,
                        torque_rate=torque_rate,
                        tolerance=cfg.feasibility_tolerance,
                    )
                    if (
                        seed_values.size
                        and np.any(
                            np.isclose(
                                seed_values,
                                z2,
                                rtol=0.0,
                                atol=max(
                                    1.0e-14,
                                    1.0e-12 * max(1.0, abs(z2)),
                                ),
                            )
                        )
                        and c2_feasible_python
                    ):
                        local_refinement[
                            "original_c2_seed_python_feasible"
                        ] += 1
                    if not c2_feasible_python:
                        continue
                    edge1 = _edge_time(
                        "c2", c2, ds0, cfg.quadrature_order
                    )
                    if not math.isfinite(edge1):
                        continue
                    local_refinement["first_interval_feasible"] += 1
                    left_slope_s2 = (z2 - z_start) / ds0
                    bounds2 = _zprime_interval(
                        1,
                        z2,
                        q_s=q_s,
                        q_ss=q_ss,
                        m=m,
                        c=c,
                        gravity=gravity,
                        acceleration=acceleration,
                        torque=torque,
                    )
                    if bounds2 is None:
                        continue
                    second_range = _search_indices(
                        stage3.z_grid,
                        z2 + bounds2[0] * ds1,
                        z2 + bounds2[1] * ds1,
                    )
                    if second_range is None:
                        continue
                    local_refinement["second_range_nonempty"] += 1
                    for l in range(
                        second_range[0], second_range[1] + 1
                    ):
                        if not (
                            math.isfinite(float(stage3.cost[l]))
                            and math.isfinite(float(stage3.slope[l]))
                        ):
                            continue
                        z3 = float(stage3.z_grid[l])
                        if z3 <= 0.0:
                            continue
                        local_refinement["finite_stage3_pairs"] += 1
                        if cfg.algorithm3_free_slope_scan_points > 0:
                            free_bounds = _zprime_interval(
                                2,
                                z3,
                                q_s=q_s,
                                q_ss=q_ss,
                                m=m,
                                c=c,
                                gravity=gravity,
                                acceleration=acceleration,
                                torque=torque,
                            )
                            if (
                                free_bounds is None
                                or not math.isfinite(free_bounds[0])
                                or not math.isfinite(free_bounds[1])
                            ):
                                local_refinement[
                                    "free_slope_unbounded_pairs"
                                ] += 1
                            else:
                                local_refinement[
                                    "free_slope_bounded_pairs"
                                ] += 1
                                pair_has_physical = False
                                for free_slope in np.linspace(
                                    float(free_bounds[0]),
                                    float(free_bounds[1]),
                                    int(
                                        cfg.algorithm3_free_slope_scan_points
                                    ),
                                    dtype=np.float64,
                                ):
                                    free_c4 = _c4(
                                        z2,
                                        z3,
                                        left_slope_s2,
                                        float(free_slope),
                                        ds1,
                                    )
                                    if free_c4 is None:
                                        continue
                                    local_refinement[
                                        "free_slope_candidates_tested"
                                    ] += 1
                                    if not _edge_feasible(
                                        1,
                                        "c4",
                                        free_c4,
                                        ds1,
                                        edge_geometry[1],
                                        eta=edge_geometry[1].eta,
                                        velocity=velocity,
                                        acceleration=acceleration,
                                        torque=torque,
                                        jerk=jerk,
                                        torque_rate=torque_rate,
                                        tolerance=cfg.feasibility_tolerance,
                                    ):
                                        continue
                                    local_refinement[
                                        "free_slope_physical_feasible"
                                    ] += 1
                                    if not pair_has_physical:
                                        pair_has_physical = True
                                        local_refinement[
                                            "free_slope_pairs_with_physical"
                                        ] += 1
                                    if not math.isfinite(
                                        float(
                                            local_refinement[
                                                "best_free_slope"
                                            ]
                                        )
                                    ):
                                        local_refinement[
                                            "best_free_slope"
                                        ] = float(free_slope)
                                        local_refinement[
                                            "best_free_slope_z2"
                                        ] = z2
                                        local_refinement[
                                            "best_free_slope_z3"
                                        ] = z3
                        labels = stage3_label_bank.get(int(l))
                        if not labels:
                            labels = [
                                (
                                    float(stage3.cost[l]),
                                    int(stage3.successor[l]),
                                    float(stage3.slope[l]),
                                )
                            ]
                        for label_rank, (
                            label_cost,
                            _label_successor,
                            right_slope,
                        ) in enumerate(labels):
                            local_refinement[
                                "stage3_label_pairs_tested"
                            ] += 1
                            if label_rank > 0:
                                local_refinement[
                                    "stage3_alt_label_pairs_tested"
                                ] += 1
                            c4 = _c4(
                                z2,
                                z3,
                                left_slope_s2,
                                float(right_slope),
                                ds1,
                            )
                            if c4 is None:
                                continue
                            local_refinement[
                                "c4_polynomial_valid_pairs"
                            ] += 1
                            if not _edge_feasible(
                                1,
                                "c4",
                                c4,
                                ds1,
                                edge_geometry[1],
                                eta=edge_geometry[1].eta,
                                velocity=velocity,
                                acceleration=acceleration,
                                torque=torque,
                                jerk=jerk,
                                torque_rate=torque_rate,
                                tolerance=cfg.feasibility_tolerance,
                            ):
                                continue
                            local_refinement[
                                "c4_physical_feasible_pairs"
                            ] += 1
                            edge2 = _edge_time(
                                "c4", c4, ds1, cfg.quadrature_order
                            )
                            total_local = (
                                edge1 + edge2 + float(label_cost)
                            )
                            if (
                                math.isfinite(total_local)
                                and total_local < best_local
                            ):
                                best_local = total_local
                                local_refinement["bridge_found"] = True
                                local_refinement["best_z2"] = z2
                                local_refinement["best_z3"] = z3
                                local_refinement[
                                    "best_stage3_index"
                                ] = int(l)
                                local_refinement[
                                    "best_stage3_label_rank"
                                ] = int(label_rank)
                                local_refinement[
                                    "best_stage3_slope"
                                ] = float(right_slope)
                                local_refinement[
                                    "best_total_duration"
                                ] = float(total_local)

        start_bridge_diagnostics = {
            "algorithm3_first_range_size": int(first_range_size),
            "algorithm3_c2_feasible_states": int(c2_feasible_indices.size),
            "algorithm3_c2_feasible_z_min": (
                float(np.min(c2_values)) if c2_values.size else float("nan")
            ),
            "algorithm3_c2_feasible_z_max": (
                float(np.max(c2_values)) if c2_values.size else float("nan")
            ),
            "algorithm3_second_range_states": int(second_range_states),
            "algorithm3_second_range_pairs": int(second_range_pairs),
            "algorithm3_finite_stage3_pairs": int(finite_stage3_pairs),
            "algorithm3_finite_stage3_unique": int(len(finite_stage3_unique)),
            "algorithm3_c4_connected_z2_states": int(
                np.count_nonzero(stage2_successor >= 0)
            ),
            "algorithm3_stage3_finite_states_total": int(
                np.count_nonzero(np.isfinite(stage3.cost))
            ),
            "algorithm3_node2_grid_size": int(node2.size),
            "algorithm3_node3_grid_size": int(stage3.z_grid.size),
            "endpoint_z_enrichment_count": int(
                cfg.endpoint_z_enrichment_count
            ),
            "endpoint_start_inserted": int(endpoint_start_inserted),
            "endpoint_terminal_inserted": int(endpoint_terminal_inserted),
            "algorithm3_local_refinement": local_refinement,
        }
        return KasererDP3Result(
            False,
            grid,
            torch.full_like(grid, torch.nan),
            torch.full_like(grid, torch.nan),
            float("inf"),
            (time.perf_counter() - started) * 1.0e3,
            preparation_ms,
            search_ms,
            "DP3 start two-interval search found no feasible solution",
            {
                "z_caps": z_caps.tolist(),
                "transitions_checked": transitions_checked,
                "feasible_transitions": feasible_transitions,
                "algorithm1_terminal_ms": algorithm1_ms,
                "algorithm2_c3_ms": algorithm2_ms,
                "algorithm2_range_ms": algorithm2_range_ms,
                "algorithm2_kernel_ms": algorithm2_kernel_ms,
                "algorithm3_start_ms": algorithm3_ms,
                "finite_states": finite_states,
                "stage_finite_counts": stage_finite_counts.tolist(),
                **start_bridge_diagnostics,
            },
        )

    # Forward run: follow stored successors.
    forward_started = time.perf_counter()
    selected = np.full(N, -1, dtype=np.int64)
    selected_label = np.zeros(N, dtype=np.int64)
    selected[0] = 0
    selected[1] = best_j
    selected[2] = best_l
    selected_label[2] = int(best_stage3_label)
    for k in range(2, N - 1):
        stage = stages[k]
        assert stage is not None
        idx = int(selected[k])
        if idx < 0:
            raise RuntimeError("broken DP3 forward chain")
        if cfg.dp_labels_per_state > 1:
            label_stage = label_stages[k]
            assert label_stage is not None
            rank = int(selected_label[k])
            nxt = int(label_stage.successor[idx, rank])
            next_rank = int(
                label_stage.successor_label[idx, rank]
            )
        else:
            nxt = int(stage.successor[idx])
            next_rank = 0
        if k < N - 2:
            if nxt < 0 or next_rank < 0:
                raise RuntimeError("broken DP3 successor chain")
            selected[k + 1] = nxt
            selected_label[k + 1] = next_rank
        else:
            selected[k + 1] = 0

    z = np.empty(N, dtype=np.float64)
    z_prime = np.empty(N, dtype=np.float64)
    z[0] = z_start
    z_prime[0] = best_start_slope
    z[1] = float(z_grids[1][selected[1]])
    z_prime[1] = best_start_slope
    for k in range(2, N - 1):
        stage = stages[k]
        assert stage is not None
        idx = int(selected[k])
        z[k] = float(stage.z_grid[idx])
        if cfg.dp_labels_per_state > 1:
            label_stage = label_stages[k]
            assert label_stage is not None
            z_prime[k] = float(
                label_stage.slope[idx, int(selected_label[k])]
            )
        else:
            z_prime[k] = float(stage.slope[idx])
    z[-1] = z_end
    z_prime[-1] = z_prime[-2]

    forward_ms = (time.perf_counter() - forward_started) * 1.0e3

    finite = bool(
        np.all(np.isfinite(z))
        and np.all(np.isfinite(z_prime))
        and math.isfinite(best_total)
    )
    return KasererDP3Result(
        finite,
        grid,
        torch.as_tensor(z, device=problem.device, dtype=problem.dtype),
        torch.as_tensor(z_prime, device=problem.device, dtype=problem.dtype),
        float(best_total),
        (time.perf_counter() - started) * 1.0e3,
        preparation_ms,
        search_ms,
        "solved" if finite else "non-finite paper DP3 profile",
        diagnostics={
            "grid_samples": cfg.grid_samples,
            "torque_rate_mode": cfg.torque_rate_mode,
            "z_samples": cfg.z_samples,
            "check_points": cfg.check_points,
            "path_segment_check_points": cfg.path_segment_check_points,
            "endpoint_path_segment_check_points": (
                cfg.endpoint_path_segment_check_points
            ),
            "endpoint_z_enrichment": cfg.endpoint_z_enrichment,
            "endpoint_z_enrichment_check_points": enrichment_path_points,
            "endpoint_z_enrichment_count": cfg.endpoint_z_enrichment_count,
            "algorithm3_local_refinement_points": (
                cfg.algorithm3_local_refinement_points
            ),
            "algorithm3_free_slope_scan_points": (
                cfg.algorithm3_free_slope_scan_points
            ),
            "algorithm3_stage3_labels": (
                cfg.algorithm3_stage3_labels
            ),
            "dp_labels_per_state": cfg.dp_labels_per_state,
            "dp_label_retention": cfg.dp_label_retention,
            "selected_label_indices": selected_label.tolist(),
            "endpoint_start_z": endpoint_start_z,
            "endpoint_terminal_z": endpoint_terminal_z,
            "endpoint_start_values": endpoint_start_values.tolist(),
            "endpoint_terminal_values": endpoint_terminal_values.tolist(),
            "endpoint_start_inserted": endpoint_start_inserted,
            "endpoint_terminal_inserted": endpoint_terminal_inserted,
            "edge_check_points_min": min(g.eta.size for g in edge_geometry),
            "edge_check_points_max": max(g.eta.size for g in edge_geometry),
            "edge_check_points_mean": float(
                np.mean([g.eta.size for g in edge_geometry])
            ),
            "quadrature_order": cfg.quadrature_order,
            "safety_scale": cfg.safety_scale,
            "z_caps": z_caps.tolist(),
            "selected_indices": selected.tolist(),
            "finite_states": finite_states,
            "stage_finite_counts": stage_finite_counts.tolist(),
            "transitions_checked": transitions_checked,
            "feasible_transitions": feasible_transitions,
            "algorithm1_terminal_ms": algorithm1_ms,
            "algorithm2_c3_ms": algorithm2_ms,
            "algorithm2_range_ms": algorithm2_range_ms,
            "algorithm2_kernel_ms": algorithm2_kernel_ms,
            "algorithm3_start_ms": algorithm3_ms,
            "forward_ms": forward_ms,
            "interpolation": (
                "Kaserer C2 terminal/start, C3 interior, C4 second interval"
            ),
            "state": "(s,z=sdot^2); z_prime stored, not discretized",
            "search": "Algorithms 1-3 backward DP + forward successor run",
            "paper_doi": "10.1109/TRO.2018.2880120",
            "transition_backend": "native_c3_c4" if cfg.use_native else "python",
            "torque_rate_check": (
                "analytic path-coefficient JVP at interpolation checkpoints"
                if cfg.torque_rate_mode == "analytic"
                else "finite difference of interpolation-check-point torques in physical time"
            ),
        },
    )


def _profile_on_dense_grid(
    result: KasererDP3Result,
    dense_s: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    s_np = result.s.detach().cpu().double().numpy()
    z_np = result.z.detach().cpu().double().numpy()
    zp_np = result.z_prime.detach().cpu().double().numpy()
    query = dense_s.detach().cpu().double().numpy()

    out_z = np.empty_like(query)
    out_zp = np.empty_like(query)
    out_zpp = np.empty_like(query)
    N = s_np.size

    interval = np.searchsorted(s_np, query, side="right") - 1
    interval = np.clip(interval, 0, N - 2)
    for k in range(N - 1):
        mask = interval == k
        if not np.any(mask):
            continue
        ds = float(s_np[k + 1] - s_np[k])
        eta = (query[mask] - s_np[k + 1]) / ds
        if k == 0 or k == N - 2:
            kind: ProfileKind = "c2"
            coeff = _c2(float(z_np[k]), float(z_np[k + 1]))
        elif k == 1:
            kind = "c4"
            coeff = _c4(
                float(z_np[k]),
                float(z_np[k + 1]),
                float(zp_np[k]),
                float(zp_np[k + 1]),
                ds,
            )
            if coeff is None:
                raise RuntimeError("invalid reconstructed C4 profile")
        else:
            kind = "c3"
            coeff = _c3(
                float(z_np[k]),
                float(z_np[k + 1]),
                float(zp_np[k + 1]),
                ds,
            )
            if coeff is None:
                raise RuntimeError("invalid reconstructed C3 profile")
        zz, zzp, zzpp = _evaluate_profile(kind, coeff, ds, eta)
        out_z[mask] = zz
        out_zp[mask] = zzp
        out_zpp[mask] = zzpp

    return (
        torch.as_tensor(out_z, device=dense_s.device, dtype=dense_s.dtype),
        torch.as_tensor(out_zp, device=dense_s.device, dtype=dense_s.dtype),
        torch.as_tensor(out_zpp, device=dense_s.device, dtype=dense_s.dtype),
    )


def _stable_audit_grid_union(
    base: Tensor,
    local_grids: list[tuple[Tensor, float]],
) -> Tensor:
    """Merge independent audit grids without numerical near-collisions.

    Exact union can leave global/local points separated by only a few
    float32 ulps even though both grids target much coarser spacing. A
    finite-difference torque-rate check across such an accidental micro
    interval is dominated by inverse-dynamics roundoff.

    Each point carries the intended spacing of the grid that generated it.
    Points are coalesced only when their separation is a small numerical
    fraction of the smaller intended spacing, so genuinely tiny source
    segments are preserved. Source knots take priority over regular points.
    """

    if not local_grids:
        return base
    if not base.dtype.is_floating_point:
        raise TypeError("audit grid must use a floating-point dtype")

    eps = float(torch.finfo(base.dtype).eps)
    collision_fraction = 8.0 * math.sqrt(eps)
    base_np = base.detach().cpu().double().numpy()
    base_step = (
        float(np.median(np.diff(base_np)))
        if base_np.size >= 2
        else 1.0
    )

    values: list[np.ndarray] = [base_np]
    scales: list[np.ndarray] = [
        np.full(base_np.shape, base_step, dtype=np.float64)
    ]
    # Priority: source knot > global grid > local interior point.
    priorities: list[np.ndarray] = [
        np.ones(base_np.shape, dtype=np.int8)
    ]

    for grid, intended_step in local_grids:
        points = grid.detach().cpu().double().numpy()
        if points.size == 0:
            continue
        scale = max(float(intended_step), np.finfo(np.float64).tiny)
        values.append(points)
        scales.append(np.full(points.shape, scale, dtype=np.float64))
        priority = np.zeros(points.shape, dtype=np.int8)
        priority[0] = 2
        priority[-1] = 2
        priorities.append(priority)

    value = np.concatenate(values)
    scale = np.concatenate(scales)
    priority = np.concatenate(priorities)
    order = np.argsort(value, kind="mergesort")
    value = value[order]
    scale = scale[order]
    priority = priority[order]

    keep: list[float] = []
    cluster_start = 0
    for i in range(1, value.size + 1):
        close_cluster = False
        if i < value.size:
            gap = float(value[i] - value[i - 1])
            spacing_threshold = collision_fraction * min(
                float(scale[i]),
                float(scale[i - 1]),
            )
            # Also reject accidental float32 near-collisions directly in
            # representable-value units. Five-to-eight-ULP gaps can still
            # amplify inverse-dynamics roundoff in finite-difference torque
            # rate even when the intended-grid spacing criterion is just
            # missed. Genuine tiny source segments remain many ULPs apart.
            ulp_i = abs(float(np.spacing(np.float32(value[i]))))
            ulp_prev = abs(
                float(np.spacing(np.float32(value[i - 1])))
            )
            ulp_threshold = 8.0 * max(ulp_i, ulp_prev)
            threshold = max(spacing_threshold, ulp_threshold)
            close_cluster = gap <= threshold

        if close_cluster:
            continue

        sl = slice(cluster_start, i)
        cluster_priority = priority[sl]
        best_priority = int(np.max(cluster_priority))
        candidates = np.flatnonzero(cluster_priority == best_priority)
        cluster_values = value[sl]
        mean = float(np.mean(cluster_values))
        local = int(
            candidates[
                np.argmin(np.abs(cluster_values[candidates] - mean))
            ]
        )
        keep.append(float(cluster_values[local]))
        cluster_start = i

    merged = torch.as_tensor(
        np.asarray(keep, dtype=np.float64),
        device=base.device,
        dtype=base.dtype,
    )
    merged = torch.unique(merged, sorted=True)
    # The global audit grid defines the domain. A support knot that lies only
    # a few ULPs from an endpoint may be clustered with it, but must never
    # replace the exact domain boundary because the dense validator requires
    # the query grid to span [0, 1] exactly.
    if merged.numel() >= 1:
        merged[0] = base[0]
        merged[-1] = base[-1]
    return merged


def _audit_grid(
    problem: RetimingProblem,
    samples: int,
    path_segment_points: int = 0,
) -> Tensor:
    """Uniform audit grid augmented with stable local path sampling."""

    if samples < 2:
        raise ValueError("samples must be at least 2")
    base = torch.linspace(
        0.0,
        1.0,
        samples,
        device=problem.device,
        dtype=problem.dtype,
    )
    if path_segment_points == 0:
        return base
    if path_segment_points < 3:
        raise ValueError("path_segment_points must be 0 or at least 3")

    support = _path_support_grid(problem)
    if support.size < 2:
        return base

    local_grids: list[tuple[Tensor, float]] = []
    for a, b in zip(support[:-1], support[1:]):
        if not b > a:
            continue
        local_grids.append(
            (
                torch.linspace(
                    float(a),
                    float(b),
                    path_segment_points,
                    device=problem.device,
                    dtype=problem.dtype,
                ),
                float(b - a) / float(path_segment_points - 1),
            )
        )
    return _stable_audit_grid_union(base, local_grids)


@torch.no_grad()
def validate_kaserer_dp3_profile_gpu(
    problem: RetimingProblem,
    result: KasererDP3Result,
    *,
    samples: int = 4097,
    tolerance: float = 5.0e-4,
    path_segment_points: int = 0,
) -> DenseFeasibilityResult:
    """Independent full-C3 audit of the reconstructed paper profile.

    path_segment_points augments the global uniform audit inside every
    exposed geometric-path segment. This targets highly nonuniform natural
    cubic support that a practical uniform global audit can undersample.
    """

    if not result.success:
        raise ValueError("cannot validate an unsuccessful DP3 result")
    dense_s = _audit_grid(problem, samples, path_segment_points)
    z, zp, zpp = _profile_on_dense_grid(result, dense_s)
    speed = torch.sqrt(torch.clamp_min(z, 0.0))
    sddot = 0.5 * zp
    sdddot = 0.5 * zpp * speed
    return validate_dense_path_profile_gpu(
        problem,
        dense_s,
        z,
        sddot,
        sdddot,
        tolerance=tolerance,
    )


__all__ = [
    "KasererDP3Config",
    "KasererDP3Result",
    "solve_kaserer_dp3",
    "validate_kaserer_dp3_profile_gpu",
]
