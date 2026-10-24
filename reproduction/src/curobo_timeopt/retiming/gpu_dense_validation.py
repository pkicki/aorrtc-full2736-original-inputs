"""CUDA-resident batched dense physical validation for timing profiles."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor

from curobo_timeopt.dynamics import (
    PathDynamicsTerms,
    extract_path_dynamics,
    torque_rate_jvp,
    torque_rate_rnea_fd5,
)
from curobo_timeopt.path import BSplinePath
from curobo_timeopt.retiming.bspline.basis import BSplineBasis
from curobo_timeopt.retiming.kinematic_reductions_triton import (
    supports_fused_basis_timing_shape,
)
from curobo_timeopt.retiming.problem import RetimingProblem
from curobo_timeopt.types import ConstraintMode

_BASIS_CACHE: dict[tuple[str, torch.dtype, int, int, int], BSplineBasis] = {}


def _cached_basis(s: Tensor, coefficients: int, degree: int) -> BSplineBasis:
    key = (str(s.device), s.dtype, s.numel(), coefficients, degree)
    basis = _BASIS_CACHE.get(key)
    if basis is None:
        basis = BSplineBasis.create(s, coefficients, degree)
        _BASIS_CACHE[key] = basis
    return basis


def _should_use_fused_basis_timing(
    *,
    requested: bool,
    batch: int,
    all_kinematic: bool,
    device_type: str,
    dtype: torch.dtype,
    coefficients: int,
    dof: int,
    samples: int,
) -> bool:
    """Check both runtime conditions and the Triton kernel's supported shape."""
    return bool(
        requested
        and batch > 1
        and all_kinematic
        and device_type == "cuda"
        and dtype == torch.float32
        and supports_fused_basis_timing_shape(coefficients, dof, samples)
    )


@dataclass(frozen=True)
class ControllerRateValidationResult:
    """Physical torque-rate audit at fixed controller timestamps."""

    feasible: Tensor
    torque_rate_violation: Tensor
    max_abs_torque_rate: Tensor
    duration: Tensor
    controller_intervals: Tensor
    frequency_hz: float
    integration_samples: int


@dataclass(frozen=True)
class DenseFeasibilityResult:
    feasible: Tensor
    max_violation: Tensor
    velocity_violation: Tensor
    acceleration_violation: Tensor
    jerk_violation: Tensor
    torque_violation: Tensor
    torque_rate_violation: Tensor
    violating_intervals: Tensor
    interval_max_violation: Tensor
    x: Tensor
    s: Tensor


def _expand_limit(limit: Tensor, value: Tensor) -> Tensor:
    result = limit
    while result.ndim < value.ndim:
        result = result.unsqueeze(0)
    return result


def _node_derivatives(s: Tensor, x: Tensor) -> tuple[Tensor, Tensor]:
    x_s = torch.empty_like(x)
    x_s[:, 0] = (x[:, 1] - x[:, 0]) / (s[1] - s[0])
    x_s[:, -1] = (x[:, -1] - x[:, -2]) / (s[-1] - s[-2])
    x_s[:, 1:-1] = (x[:, 2:] - x[:, :-2]) / (s[2:] - s[:-2])
    x_ss = torch.empty_like(x)
    x_ss[:, 0] = (x_s[:, 1] - x_s[:, 0]) / (s[1] - s[0])
    x_ss[:, -1] = (x_s[:, -1] - x_s[:, -2]) / (s[-1] - s[-2])
    x_ss[:, 1:-1] = (x_s[:, 2:] - x_s[:, :-2]) / (s[2:] - s[:-2])
    return x_s, x_ss


def _linear_interpolate(s: Tensor, x: Tensor, query: Tensor) -> Tensor:
    index = (torch.searchsorted(s, query, right=True) - 1).clamp(0, s.numel() - 2)
    weight = (query - s[index]) / (s[index + 1] - s[index])
    return x[:, index] + weight[None] * (x[:, index + 1] - x[:, index])


def _validate_dense_grid_inputs(problem: RetimingProblem, dense_s: Tensor) -> None:
    """Run the canonical host-visible dense-grid checks outside capture."""
    if dense_s.ndim != 1 or dense_s.numel() < 3:
        raise ValueError("dense_s must be a one-dimensional grid")
    if (
        dense_s.device != problem.device
        or dense_s.dtype != problem.dtype
        or not torch.all(dense_s[1:] > dense_s[:-1])
    ):
        raise ValueError("dense_s must be increasing on the problem device/dtype")
    if abs(float(dense_s[0])) > 1e-7 or abs(float(dense_s[-1]) - 1.0) > 1e-7:
        raise ValueError("dense_s must span [0, 1]")


@torch.no_grad()
def validate_dense_path_profile_gpu(
    problem: RetimingProblem,
    dense_s: Tensor,
    x: Tensor,
    sddot: Tensor,
    sdddot: Tensor,
    *,
    tolerance: float = 5e-4,
    enforce_endpoint_speed: bool = True,
    torque_rate_mode: str = "sampled",
    torque_rate_jvp_chunk_size: int = 1024,
    research_shared_geometry: bool = False,
) -> DenseFeasibilityResult:
    """Audit an explicitly sampled external path-time profile.

    This compatibility entry point is intentionally additive: modern repository
    B-spline validators remain unchanged, while literature adapters can submit
    their native dense x=sdot^2, sddot and sdddot execution profiles to exactly
    the same robot-space limit checks.

    torque_rate_mode="sampled" preserves the historical finite-difference
    audit diff(tau)/dt. "analytic_jvp" evaluates the continuous-time
    inverse-dynamics derivative by JVP in the trajectory direction
    (qdot, qddot, qdddot). "rnea_fd5" independently differentiates RNEA along
    that direction with a centered five-point stencil, avoiding subtraction
    over the event-grid interval widths. Research-only ``fd5_fused`` dispatches
    to a backend-provided packed stencil and fails closed when absent; it is
    intended for cheap trial screening, not final feasibility acceptance.
    """
    if torque_rate_mode not in ("sampled", "analytic_jvp", "rnea_fd5", "fd5_fused"):
        raise ValueError(
            "torque_rate_mode must be 'sampled', 'analytic_jvp', 'rnea_fd5', "
            "or 'fd5_fused'"
        )
    if torque_rate_mode == "fd5_fused" and not callable(
        getattr(problem.dynamics, "inverse_dynamics_directional_fd5_fused", None)
    ):
        raise RuntimeError(
            "torque_rate_mode='fd5_fused' requires a dynamics backend with "
            "inverse_dynamics_directional_fd5_fused; no fallback is performed"
        )
    if torque_rate_jvp_chunk_size < 1:
        raise ValueError("torque_rate_jvp_chunk_size must be positive")
    if type(research_shared_geometry) is not bool:
        raise TypeError("research geometry flag must be Boolean")
    if research_shared_geometry and not isinstance(problem.path, BSplinePath):
        raise ValueError("shared derivative evaluation requires an exact BSplinePath")
    _validate_dense_grid_inputs(problem, dense_s)
    return _validate_dense_path_profile_gpu_tensor_core(
        problem,
        dense_s,
        x,
        sddot,
        sdddot,
        tolerance=tolerance,
        enforce_endpoint_speed=enforce_endpoint_speed,
        torque_rate_mode=torque_rate_mode,
        torque_rate_jvp_chunk_size=torque_rate_jvp_chunk_size,
        research_shared_geometry=research_shared_geometry,
    )


@torch.no_grad()
def _stable_profile_time_steps(dense_s: Tensor, x: Tensor, sddot: Tensor) -> Tensor:
    """Keep representable event sides without mistaking underflow for a stop.

    At nextafter(0), a valid linearly growing squared speed can round to zero.
    Both sampled speeds are then zero although the boundary acceleration is
    strictly positive. Use the exact infinitesimal constant-acceleration limit
    sqrt(2*ds/a), evaluated as sqrt(ds)*sqrt(2)/sqrt(a) to avoid underflow.
    Only subnormal boundary intervals qualify: real interior stops, plateaus,
    and zero boundary acceleration retain their ordinary invalid/infinite dt.
    """
    ds = torch.diff(dense_s)
    speed = x.clamp_min(0).sqrt()
    denominator = speed[:, :-1] + speed[:, 1:]
    dt = 2 * ds / denominator
    start = (dense_s[:-1] == 0)[None] & (sddot[:, :-1] > 0)
    end = (dense_s[1:] == 1)[None] & (sddot[:, 1:] < 0)
    acceleration = torch.where(start, sddot[:, :-1], -sddot[:, 1:])
    boundary_underflow = (
        (start | end)
        & (denominator == 0)
        & ((ds > 0) & (ds <= torch.finfo(dense_s.dtype).tiny))[None]
        & torch.isfinite(acceleration)
    )
    limit = (
        torch.sqrt(ds.double())[None]
        * (2.0**0.5)
        / torch.sqrt(acceleration.double().abs())
    ).to(dt.dtype)
    return torch.where(boundary_underflow, limit, dt)


@torch.no_grad()
def _validate_dense_path_profile_gpu_tensor_core(
    problem: RetimingProblem,
    dense_s: Tensor,
    x: Tensor,
    sddot: Tensor,
    sdddot: Tensor,
    *,
    tolerance: float = 5e-4,
    enforce_endpoint_speed: bool = True,
    torque_rate_mode: str = "sampled",
    torque_rate_jvp_chunk_size: int = 1024,
    research_shared_geometry: bool = False,
) -> DenseFeasibilityResult:
    """Tensor evaluation after the exact dense grid was checked outside capture."""
    if torque_rate_mode not in ("sampled", "analytic_jvp", "rnea_fd5", "fd5_fused"):
        raise ValueError(
            "torque_rate_mode must be 'sampled', 'analytic_jvp', 'rnea_fd5', "
            "or 'fd5_fused'"
        )
    if torque_rate_jvp_chunk_size < 1:
        raise ValueError("torque_rate_jvp_chunk_size must be positive")
    if type(research_shared_geometry) is not bool:
        raise TypeError("research geometry flag must be Boolean")
    if research_shared_geometry and not isinstance(problem.path, BSplinePath):
        raise ValueError("shared derivatives require a BSplinePath")

    def as_batch(value: Tensor, name: str) -> Tensor:
        if value.ndim == 1:
            value = value[None]
        if (
            value.ndim != 2
            or value.shape[1] != dense_s.numel()
            or value.device != problem.device
            or value.dtype != problem.dtype
        ):
            raise ValueError(
                f"{name} must have shape [B,{dense_s.numel()}] "
                "on the problem device/dtype"
            )
        return value

    x = as_batch(x, "x")
    sddot = as_batch(sddot, "sddot")
    sdddot = as_batch(sdddot, "sdddot")
    if sddot.shape[0] != x.shape[0] or sdddot.shape[0] != x.shape[0]:
        raise ValueError("x, sddot and sdddot must use the same batch size")

    # NEVER key an event-aware/nonuniform query grid by sample count alone.
    # Shared evaluation reuses ONLY the stable derivative-control recursion;
    # each derivative still evaluates its ORIGINAL lower-degree basis.
    if research_shared_geometry:
        q, q_s, q_ss, q_sss = (
            value[None]
            for value in problem.path.evaluate_all(
                dense_s,
                max_order=3,
                uniform_grid=False,
                cache_nonuniform_query=False,
            )
        )
    else:
        q = problem.path.evaluate(dense_s, 0)[None]
        q_s = problem.path.evaluate(dense_s, 1)[None]
        q_ss = problem.path.evaluate(dense_s, 2)[None]
        q_sss = problem.path.evaluate(dense_s, 3)[None]

    sdot = torch.sqrt(torch.clamp_min(x, 0))
    qd = q_s * sdot[..., None]
    qdd = q_ss * x[..., None] + q_s * sddot[..., None]
    qddd = (
        q_sss * (sdot**3)[..., None]
        + 3 * q_ss * (sddot * sdot)[..., None]
        + q_s * sdddot[..., None]
    )
    dt = _stable_profile_time_steps(dense_s, x, sddot)
    batch = x.shape[0]
    zeros = torch.zeros(batch, device=x.device, dtype=x.dtype)
    interval = torch.zeros((batch, dense_s.numel() - 1), device=x.device, dtype=x.dtype)

    def node_excess(value: Tensor, limit: Tensor) -> tuple[Tensor, Tensor]:
        excess = torch.clamp_min(value.abs() - _expand_limit(limit, value), 0).amax(-1)
        located = torch.maximum(excess[:, :-1], excess[:, 1:])
        return excess.amax(-1), located

    velocity = acceleration = jerk = torque = torque_rate = zeros
    if problem.mode & ConstraintMode.VELOCITY:
        velocity, local = node_excess(qd, problem.limits.velocity)
        interval = torch.maximum(interval, local)
    if problem.mode & ConstraintMode.ACCELERATION:
        acceleration, local = node_excess(qdd, problem.limits.acceleration)
        interval = torch.maximum(interval, local)
    if problem.mode & ConstraintMode.JERK:
        jerk, local = node_excess(qddd, problem.limits.jerk)
        interval = torch.maximum(interval, local)

    if problem.mode & (ConstraintMode.TORQUE | ConstraintMode.TORQUE_RATE):
        if problem.dynamics is None:
            raise ValueError("torque validation requires inverse dynamics")
        tau = problem.dynamics.inverse_dynamics(q.expand(batch, -1, -1), qd, qdd)
        if problem.mode & ConstraintMode.TORQUE:
            torque, local = node_excess(tau, problem.limits.torque)
            interval = torch.maximum(interval, local)
        if problem.mode & ConstraintMode.TORQUE_RATE:
            if torque_rate_mode == "sampled":
                tau_rate_value = torch.diff(tau, dim=1) / dt[..., None]
                rate_excess = torch.clamp_min(
                    tau_rate_value.abs()
                    - _expand_limit(problem.limits.torque_rate, tau_rate_value),
                    0,
                ).amax(-1)
                torque_rate = rate_excess.amax(-1)
                interval = torch.maximum(interval, rate_excess)
            elif torque_rate_mode == "analytic_jvp":
                # Event-aware grids may contain O(1e-5 s) intervals. Direct
                # float32 torque subtraction on those intervals can amplify
                # ordinary RNEA roundoff into O(1e2-1e3) Nm/s. Evaluate the
                # continuous-time derivative directly by inverse-dynamics JVP.
                analytic_parts: list[Tensor] = []
                with torch.enable_grad():
                    for start in range(
                        0, dense_s.numel(), int(torque_rate_jvp_chunk_size)
                    ):
                        stop = min(
                            start + int(torque_rate_jvp_chunk_size),
                            dense_s.numel(),
                        )
                        analytic_parts.append(
                            torque_rate_jvp(
                                problem.dynamics,
                                # Native JVP requires compact inputs: N>1 grid-axis
                                # slices keep a larger leading stride otherwise,
                                # triggering a sticky reverse-native fallback.
                                q.expand(batch, -1, -1)[:, start:stop].contiguous(),
                                qd[:, start:stop].contiguous(),
                                qdd[:, start:stop].contiguous(),
                                qddd[:, start:stop].contiguous(),
                            ).tangent.detach()
                        )
                tau_rate_value = torch.cat(analytic_parts, dim=1)
            elif torque_rate_mode == "fd5_fused":
                fd5_fused = getattr(
                    problem.dynamics, "inverse_dynamics_directional_fd5_fused", None
                )
                if not callable(fd5_fused):
                    raise RuntimeError(
                        "torque_rate_mode='fd5_fused' requires a dynamics backend "
                        "with inverse_dynamics_directional_fd5_fused; no fallback "
                        "is performed"
                    )
                tau_rate_value = fd5_fused(q.expand(batch, -1, -1), qd, qdd, qddd)
            else:
                tau_rate_value = torque_rate_rnea_fd5(
                    problem.dynamics,
                    q.expand(batch, -1, -1),
                    qd,
                    qdd,
                    qddd,
                )
            if torque_rate_mode != "sampled":
                torque_rate, local = node_excess(
                    tau_rate_value,
                    problem.limits.torque_rate,
                )
                interval = torch.maximum(interval, local)

    negative = torch.clamp_min(-x, 0).amax(-1)
    if enforce_endpoint_speed:
        requested = x.new_tensor([problem.initial_speed**2, problem.final_speed**2])
        endpoint = torch.maximum(
            (x[:, 0] - requested[0]).abs(),
            (x[:, -1] - requested[1]).abs(),
        )
    else:
        endpoint = zeros
    maximum = torch.stack(
        (velocity, acceleration, jerk, torque, torque_rate, negative, endpoint),
        dim=0,
    ).amax(0)
    finite = (
        torch.isfinite(x).all(-1)
        & torch.isfinite(sddot).all(-1)
        & torch.isfinite(sdddot).all(-1)
        & torch.isfinite(dt).all(-1)
        & (dt > 0).all(-1)
    )
    maximum = torch.where(finite, maximum, torch.full_like(maximum, torch.inf))
    return DenseFeasibilityResult(
        feasible=maximum <= tolerance,
        max_violation=maximum,
        velocity_violation=velocity,
        acceleration_violation=acceleration,
        jerk_violation=jerk,
        torque_violation=torque,
        torque_rate_violation=torque_rate,
        violating_intervals=interval > tolerance,
        interval_max_violation=interval,
        x=x,
        s=dense_s,
    )


@torch.no_grad()
def validate_controller_rate_timing_gpu(
    problem: RetimingProblem,
    timing: Tensor,
    *,
    representation: str = "bspline",
    degree: int = 3,
    frequency_hz: float = 1000.0,
    integration_samples: int = 8193,
    tolerance: float = 5e-4,
    include_terminal_endpoint: bool = False,
) -> ControllerRateValidationResult:
    """Validate torque rate at actual fixed-time controller timestamps.

    A dense uniform-s grid is used only to numerically construct the monotone
    time map t(s). For each timing profile, that map is inverted at
    t_k = k / frequency_hz. The timing spline and geometric path are then
    evaluated directly at the recovered s_k positions before running inverse
    dynamics. Thus the reported torque-rate is based on the controller's
    finite-time command difference rather than adjacent samples of a uniform-s
    audit grid.

    By default only exact controller ticks t_k = k/frequency_hz are used.
    Set include_terminal_endpoint=True to append the exact trajectory endpoint
    when it falls between ticks; that final interval is then intentionally
    shorter than the controller period.
    """
    if frequency_hz <= 0.0:
        raise ValueError("frequency_hz must be positive")
    if integration_samples < 3:
        raise ValueError("integration_samples must be at least 3")
    if timing.ndim == 1:
        timing = timing.unsqueeze(0)
    if timing.ndim != 2:
        raise ValueError("timing must have shape (batch, samples_or_coefficients)")
    if problem.dynamics is None:
        raise ValueError("controller-rate validation requires inverse dynamics")
    if not (problem.mode & ConstraintMode.TORQUE_RATE):
        raise ValueError("problem does not enable torque-rate constraints")

    dense_s = torch.linspace(
        0.0,
        1.0,
        integration_samples,
        device=timing.device,
        dtype=timing.dtype,
    )
    if representation == "bspline":
        basis = _cached_basis(dense_s, timing.shape[1], degree)
        dense_x = (timing @ basis.B.T).clamp_min(0.0)
    elif representation == "full_grid":
        dense_x = _linear_interpolate(problem.s, timing, dense_s).clamp_min(0.0)
    else:
        raise ValueError("representation must be 'bspline' or 'full_grid'")

    dense_speed = torch.sqrt(dense_x)
    ds = torch.diff(dense_s)
    dense_dt = (
        2.0 * ds[None] / (dense_speed[:, :-1] + dense_speed[:, 1:]).clamp_min(1e-12)
    )
    cumulative_t = torch.cat(
        (
            torch.zeros(
                (timing.shape[0], 1),
                device=timing.device,
                dtype=timing.dtype,
            ),
            torch.cumsum(dense_dt, dim=1),
        ),
        dim=1,
    )

    feasible_values = []
    violation_values = []
    max_rate_values = []
    duration_values = []
    interval_counts = []
    controller_dt = 1.0 / float(frequency_hz)
    torque_limit = problem.limits.torque_rate

    for batch_index in range(timing.shape[0]):
        duration = cumulative_t[batch_index, -1]
        fixed_intervals = int(torch.floor(duration * float(frequency_hz)).item())
        query_t = (
            torch.arange(
                fixed_intervals + 1,
                device=timing.device,
                dtype=timing.dtype,
            )
            * controller_dt
        )
        if query_t.numel() == 0:
            query_t = duration.new_zeros((1,))
        if include_terminal_endpoint:
            terminal_gap = duration - query_t[-1]
            if float(terminal_gap) > max(1e-9, 1e-6 * controller_dt):
                query_t = torch.cat((query_t, duration.reshape(1)))
            else:
                query_t[-1] = duration

        t_map = cumulative_t[batch_index]
        index = (torch.searchsorted(t_map, query_t, right=True) - 1).clamp(
            0, integration_samples - 2
        )
        denom = (t_map[index + 1] - t_map[index]).clamp_min(1e-12)
        weight = (query_t - t_map[index]) / denom
        query_s = dense_s[index] + weight * (dense_s[index + 1] - dense_s[index])
        query_s[0] = 0.0
        if include_terminal_endpoint:
            query_s[-1] = 1.0

        if representation == "bspline":
            timing_basis = BSplineBasis.create(
                query_s,
                timing.shape[1],
                degree,
            )
            x = (timing[batch_index] @ timing_basis.B.T).clamp_min(0.0)
            x_s = timing[batch_index] @ timing_basis.B_s.T
        else:
            x = _linear_interpolate(
                problem.s,
                timing[batch_index : batch_index + 1],
                query_s,
            )[0].clamp_min(0.0)
            x_s = _node_derivatives(query_s, x.unsqueeze(0))[0][0]

        if isinstance(problem.path, BSplinePath):
            q = problem.path.evaluate(query_s, 0)
            q_s = problem.path.evaluate(query_s, 1)
            q_ss = problem.path.evaluate(query_s, 2)
        else:
            q = problem.path.evaluate(query_s, 0)
            q_s = problem.path.evaluate(query_s, 1)
            q_ss = problem.path.evaluate(query_s, 2)

        sdot = torch.sqrt(x)
        sddot = 0.5 * x_s
        qd = q_s * sdot[:, None]
        qdd = q_ss * x[:, None] + q_s * sddot[:, None]
        tau = problem.dynamics.inverse_dynamics(q, qd, qdd)

        dt = torch.diff(query_t)
        if dt.numel() == 0:
            max_abs_rate = timing.new_zeros(())
            violation = timing.new_zeros(())
            intervals = 0
        else:
            tau_rate = torch.diff(tau, dim=0) / dt[:, None]
            abs_rate = tau_rate.abs()
            max_abs_rate = abs_rate.amax()
            violation = torch.clamp_min(
                abs_rate - torque_limit[None],
                0.0,
            ).amax()
            intervals = int(dt.numel())

        feasible_values.append(violation <= float(tolerance))
        violation_values.append(violation)
        max_rate_values.append(max_abs_rate)
        duration_values.append(duration)
        interval_counts.append(intervals)

    return ControllerRateValidationResult(
        feasible=torch.stack(feasible_values),
        torque_rate_violation=torch.stack(violation_values),
        max_abs_torque_rate=torch.stack(max_rate_values),
        duration=torch.stack(duration_values),
        controller_intervals=torch.tensor(
            interval_counts,
            device=timing.device,
            dtype=torch.int64,
        ),
        frequency_hz=float(frequency_hz),
        integration_samples=int(integration_samples),
    )


@dataclass(frozen=True)
class PreparedDenseBSplineDirectRNEAValidator:
    """Prepared dense B-spline validator preserving direct-RNEA semantics.

    This variant caches only quantities whose reuse is semantics preserving:
    dense grid, timing basis, and path derivatives. Torque is still evaluated
    through problem.dynamics.inverse_dynamics exactly as in
    validate_timing_gpu. It therefore removes fixed overhead without using the
    affine torque reconstruction that perturbs torque-rate in float32.
    """

    problem: RetimingProblem
    coefficients: int
    degree: int
    samples: int
    tolerance: float
    s: Tensor
    basis: BSplineBasis
    q: Tensor
    q_s: Tensor
    q_ss: Tensor
    q_sss: Tensor
    requested_endpoints: Tensor
    borrowed_rnea: bool = False
    fixed_path_rnea_cache: Tensor | None = None
    fixed_path_grouped_rnea: bool = False
    fused_kinematic_reductions: bool = False
    fused_timing_kinematics: bool = False
    fused_basis_timing_kinematics: bool = False
    fused_torque_final_reductions: bool = False
    fused_final_reductions: bool = False
    q_batch_cache: dict[int, Tensor] = field(
        default_factory=dict,
        compare=False,
        repr=False,
    )

    def repeated_q_batch(self, batch: int) -> Tensor:
        """Return a contiguous repeated fixed-path q batch.

        The direct cuRobo adapter flattens q with reshape before native RNEA.
        Reshaping a zero-stride expand can materialize the whole repeated path
        on every validation call. Cache the identical contiguous q batch once
        per online candidate count so the flatten is a view.
        """
        batch = int(batch)
        if batch < 1:
            raise ValueError("batch must be positive")
        cached = self.q_batch_cache.get(batch)
        if cached is None:
            cached = self.q[None].expand(batch, -1, -1).contiguous()
            # Validators are used with one fixed online batch shape. Keep the
            # cache bounded if a diagnostic calls several shapes.
            self.q_batch_cache.clear()
            self.q_batch_cache[batch] = cached
        return cached

    @classmethod
    @torch.no_grad()
    def prepare(
        cls,
        problem: RetimingProblem,
        coefficients: int,
        *,
        degree: int = 3,
        samples: int = 4097,
        grid: Tensor | None = None,
        requested_endpoint_x: Tensor | None = None,
        tolerance: float = 5e-4,
        borrowed_rnea: bool = False,
        fixed_path_rnea: bool = False,
        fixed_path_grouped_rnea: bool = False,
        fused_kinematic_reductions: bool = False,
        fused_timing_kinematics: bool = False,
        fused_basis_timing_kinematics: bool = False,
        fused_torque_final_reductions: bool = False,
        fused_final_reductions: bool = False,
    ) -> PreparedDenseBSplineDirectRNEAValidator:
        if grid is None:
            if samples < problem.s.numel():
                raise ValueError(
                    "dense validation grid cannot be coarser than the problem grid"
                )
            s = torch.linspace(
                0,
                1,
                samples,
                device=problem.device,
                dtype=problem.dtype,
            )
            basis = _cached_basis(s, coefficients, degree)
            uniform_grid = True
        else:
            s = grid
            if (
                s.ndim != 1
                or s.numel() < problem.s.numel()
                or s.device != problem.device
                or s.dtype != problem.dtype
                or not bool(torch.all(s[1:] > s[:-1]))
                or abs(float(s[0])) > 1e-7
                or abs(float(s[-1]) - 1.0) > 1e-7
            ):
                raise ValueError(
                    "custom validation grid must be increasing, span [0,1], "
                    "and live on the problem device/dtype"
                )
            samples = int(s.numel())
            basis = BSplineBasis.create_cached(
                s,
                coefficients,
                degree,
                uniform_grid=False,
            )
            uniform_grid = False
        evaluate_all = getattr(problem.path, "evaluate_all", None)
        if callable(evaluate_all):
            # Any path wrapper may expose a semantics-preserving fused
            # derivative evaluator. This keeps generic/composed paths on the
            # same interface while avoiding four duplicate basis traversals.
            q, q_s, q_ss, q_sss = evaluate_all(
                s,
                3,
                uniform_grid=uniform_grid,
            )
        else:
            q = problem.path.evaluate(s, 0)
            q_s = problem.path.evaluate(s, 1)
            q_ss = problem.path.evaluate(s, 2)
            q_sss = problem.path.evaluate(s, 3)
        if requested_endpoint_x is None:
            requested = s.new_tensor([problem.initial_speed**2, problem.final_speed**2])
        else:
            requested = requested_endpoint_x.to(
                device=problem.device,
                dtype=problem.dtype,
            ).reshape(-1)
            if requested.numel() != 2:
                raise ValueError("requested_endpoint_x must contain exactly two values")
            if not bool(torch.isfinite(requested).all()):
                raise ValueError("requested_endpoint_x must be finite")
            if bool((requested < 0.0).any()):
                raise ValueError("requested_endpoint_x must be nonnegative")
            requested = requested.contiguous()
        if fixed_path_grouped_rnea:
            fixed_path_rnea = True
        fixed_path_cache = None
        if fixed_path_rnea:
            if not borrowed_rnea:
                raise ValueError("fixed-path RNEA requires borrowed_rnea=True")
            if problem.dynamics is None:
                raise ValueError("fixed-path RNEA requires dynamics")
            prepare_fixed = getattr(
                problem.dynamics,
                "prepare_fixed_path_rnea",
                None,
            )
            if not callable(prepare_fixed):
                raise RuntimeError("dynamics backend does not support fixed-path RNEA")
            fixed_path_cache = prepare_fixed(q)
            if fixed_path_cache is None:
                raise RuntimeError(
                    "fixed-path RNEA is unavailable for this cuRobo backend"
                )
        return cls(
            problem=problem,
            coefficients=coefficients,
            degree=degree,
            samples=samples,
            tolerance=tolerance,
            s=s,
            basis=basis,
            q=q,
            q_s=q_s,
            q_ss=q_ss,
            q_sss=q_sss,
            requested_endpoints=requested,
            borrowed_rnea=bool(borrowed_rnea),
            fixed_path_rnea_cache=fixed_path_cache,
            fixed_path_grouped_rnea=bool(fixed_path_grouped_rnea),
            fused_kinematic_reductions=bool(fused_kinematic_reductions),
            fused_timing_kinematics=bool(fused_timing_kinematics),
            fused_basis_timing_kinematics=bool(fused_basis_timing_kinematics),
            fused_torque_final_reductions=bool(fused_torque_final_reductions),
            fused_final_reductions=bool(fused_final_reductions),
        )

    @torch.no_grad()
    def validate_kinematic_prefilter(
        self,
        timing: Tensor,
    ) -> tuple[DenseFeasibilityResult, Tensor]:
        """Screen exact non-dynamic constraints before direct RNEA."""
        if timing.ndim == 1:
            timing = timing[None]
        if (
            timing.ndim != 2
            or timing.shape[1] != self.coefficients
            or timing.device != self.problem.device
            or timing.dtype != self.problem.dtype
        ):
            raise ValueError(
                "timing must be [B,K] on the prepared problem device and dtype"
            )

        # This prefilter deliberately uses the evaluated-timing route for all
        # K; unlike validate(), it has no fused basis-to-kinematics dispatch.
        x = self.basis.evaluate(timing)
        x_s = self.basis.evaluate(timing, 1)
        x_ss = self.basis.evaluate(timing, 2)
        speed = torch.sqrt(torch.clamp_min(x, 0))
        sddot = 0.5 * x_s
        sdddot = 0.5 * x_ss * speed

        qd = self.q_s[None] * speed[..., None]
        qdd = self.q_ss[None] * x[..., None] + self.q_s[None] * sddot[..., None]
        qddd = (
            self.q_sss[None] * (speed**3)[..., None]
            + 3 * self.q_ss[None] * (sddot * speed)[..., None]
            + self.q_s[None] * sdddot[..., None]
        )
        dt = 2 * torch.diff(self.s) / (speed[:, :-1] + speed[:, 1:])

        batch = timing.shape[0]
        zeros = torch.zeros(
            batch,
            device=timing.device,
            dtype=timing.dtype,
        )
        interval = torch.zeros(
            (batch, self.samples - 1),
            device=timing.device,
            dtype=timing.dtype,
        )

        def node_excess(value: Tensor, limit: Tensor) -> tuple[Tensor, Tensor]:
            excess = torch.clamp_min(
                value.abs() - _expand_limit(limit, value),
                0,
            ).amax(-1)
            located = torch.maximum(excess[:, :-1], excess[:, 1:])
            return excess.amax(-1), located

        velocity = zeros
        acceleration = zeros
        jerk = zeros
        if self.problem.mode & ConstraintMode.VELOCITY:
            velocity, local = node_excess(
                qd,
                self.problem.limits.velocity,
            )
            interval = torch.maximum(interval, local)
        if self.problem.mode & ConstraintMode.ACCELERATION:
            acceleration, local = node_excess(
                qdd,
                self.problem.limits.acceleration,
            )
            interval = torch.maximum(interval, local)
        if self.problem.mode & ConstraintMode.JERK:
            jerk, local = node_excess(
                qddd,
                self.problem.limits.jerk,
            )
            interval = torch.maximum(interval, local)

        negative = torch.clamp_min(-x, 0).amax(-1)
        endpoint = torch.maximum(
            (x[:, 0] - self.requested_endpoints[0]).abs(),
            (x[:, -1] - self.requested_endpoints[1]).abs(),
        )
        maximum = torch.stack(
            (
                velocity,
                acceleration,
                jerk,
                negative,
                endpoint,
            ),
            dim=0,
        ).amax(0)
        finite = (
            torch.isfinite(x).all(-1) & torch.isfinite(dt).all(-1) & (dt > 0).all(-1)
        )
        maximum = torch.where(
            finite,
            maximum,
            torch.full_like(maximum, torch.inf),
        )
        needs_dynamics = maximum <= self.tolerance
        result = DenseFeasibilityResult(
            needs_dynamics,
            maximum,
            velocity,
            acceleration,
            jerk,
            zeros,
            zeros,
            interval > self.tolerance,
            interval,
            x,
            self.s,
        )
        return result, needs_dynamics

    @torch.no_grad()
    def validate(self, timing: Tensor) -> DenseFeasibilityResult:
        if timing.ndim == 1:
            timing = timing[None]
        if (
            timing.ndim != 2
            or timing.shape[1] != self.coefficients
            or timing.device != self.problem.device
            or timing.dtype != self.problem.dtype
        ):
            raise ValueError(
                "timing must be [B,K] on the prepared problem device and dtype"
            )

        batch = timing.shape[0]
        zeros = torch.zeros(
            batch,
            device=timing.device,
            dtype=timing.dtype,
        )
        interval = torch.zeros(
            (batch, self.samples - 1),
            device=timing.device,
            dtype=timing.dtype,
        )

        def node_excess(value: Tensor, limit: Tensor) -> tuple[Tensor, Tensor]:
            excess = torch.clamp_min(
                value.abs() - _expand_limit(limit, value),
                0,
            ).amax(-1)
            located = torch.maximum(excess[:, :-1], excess[:, 1:])
            return excess.amax(-1), located

        velocity = zeros
        acceleration = zeros
        jerk = zeros
        all_kinematic = (
            bool(self.problem.mode & ConstraintMode.VELOCITY)
            and bool(self.problem.mode & ConstraintMode.ACCELERATION)
            and bool(self.problem.mode & ConstraintMode.JERK)
        )
        use_fused_basis_timing = _should_use_fused_basis_timing(
            requested=self.fused_basis_timing_kinematics,
            batch=batch,
            all_kinematic=all_kinematic,
            device_type=timing.device.type,
            dtype=timing.dtype,
            coefficients=self.coefficients,
            dof=self.problem.path.dof,
            samples=self.samples,
        )
        if use_fused_basis_timing:
            from curobo_timeopt.retiming.kinematic_reductions_triton import (
                fused_basis_timing_kinematics_reductions,
            )

            (
                x,
                qd,
                qdd,
                dt,
                metrics,
                interval,
            ) = fused_basis_timing_kinematics_reductions(
                timing.contiguous(),
                self.basis.B.contiguous(),
                self.basis.B_s.contiguous(),
                self.basis.B_ss.contiguous(),
                self.s,
                self.q_s.contiguous(),
                self.q_ss.contiguous(),
                self.q_sss.contiguous(),
                self.problem.limits.velocity,
                self.problem.limits.acceleration,
                self.problem.limits.jerk,
            )
            velocity = metrics[0]
            acceleration = metrics[1]
            jerk = metrics[2]
        else:
            x = self.basis.evaluate(timing)
            x_s = self.basis.evaluate(timing, 1)
            x_ss = self.basis.evaluate(timing, 2)

            use_fused_timing_kinematics = bool(
                self.fused_timing_kinematics
                and batch > 1
                and all_kinematic
                and timing.device.type == "cuda"
                and timing.dtype == torch.float32
            )
            if use_fused_timing_kinematics:
                from curobo_timeopt.retiming.kinematic_reductions_triton import (
                    fused_timing_kinematics_reductions,
                )

                qd, qdd, dt, metrics, interval = fused_timing_kinematics_reductions(
                    x.contiguous(),
                    x_s.contiguous(),
                    x_ss.contiguous(),
                    self.s,
                    self.q_s.contiguous(),
                    self.q_ss.contiguous(),
                    self.q_sss.contiguous(),
                    self.problem.limits.velocity,
                    self.problem.limits.acceleration,
                    self.problem.limits.jerk,
                )
                velocity = metrics[0]
                acceleration = metrics[1]
                jerk = metrics[2]
            else:
                speed = torch.sqrt(torch.clamp_min(x, 0))
                sddot = 0.5 * x_s
                sdddot = 0.5 * x_ss * speed

                qd = self.q_s[None] * speed[..., None]
                qdd = self.q_ss[None] * x[..., None] + self.q_s[None] * sddot[..., None]
                qddd = (
                    self.q_sss[None] * (speed**3)[..., None]
                    + 3 * self.q_ss[None] * (sddot * speed)[..., None]
                    + self.q_s[None] * sdddot[..., None]
                )
                dt = 2 * torch.diff(self.s) / (speed[:, :-1] + speed[:, 1:])

                use_fused_kinematic = bool(
                    self.fused_kinematic_reductions
                    and batch > 1
                    and all_kinematic
                    and timing.device.type == "cuda"
                    and timing.dtype == torch.float32
                )
                if use_fused_kinematic:
                    from curobo_timeopt.retiming.kinematic_reductions_triton import (
                        fused_kinematic_reductions,
                    )

                    metrics, kinematic_interval = fused_kinematic_reductions(
                        qd.contiguous(),
                        qdd.contiguous(),
                        qddd.contiguous(),
                        self.problem.limits.velocity,
                        self.problem.limits.acceleration,
                        self.problem.limits.jerk,
                    )
                    velocity = metrics[0]
                    acceleration = metrics[1]
                    jerk = metrics[2]
                    interval = torch.maximum(
                        interval,
                        kinematic_interval,
                    )
                else:
                    if self.problem.mode & ConstraintMode.VELOCITY:
                        velocity, local = node_excess(
                            qd,
                            self.problem.limits.velocity,
                        )
                        interval = torch.maximum(interval, local)
                    if self.problem.mode & ConstraintMode.ACCELERATION:
                        acceleration, local = node_excess(
                            qdd,
                            self.problem.limits.acceleration,
                        )
                        interval = torch.maximum(interval, local)
                    if self.problem.mode & ConstraintMode.JERK:
                        jerk, local = node_excess(
                            qddd,
                            self.problem.limits.jerk,
                        )
                        interval = torch.maximum(interval, local)

        torque = zeros
        torque_rate = zeros
        if self.problem.mode & (ConstraintMode.TORQUE | ConstraintMode.TORQUE_RATE):
            if self.problem.dynamics is None:
                raise RuntimeError("prepared direct-RNEA validation lacks dynamics")
            fixed_grouped = getattr(
                self.problem.dynamics,
                "inverse_dynamics_fixed_path_grouped_borrowed",
                None,
            )
            fixed_borrowed = getattr(
                self.problem.dynamics,
                "inverse_dynamics_fixed_path_borrowed",
                None,
            )
            borrowed = getattr(
                self.problem.dynamics,
                "inverse_dynamics_borrowed",
                None,
            )
            if (
                self.fixed_path_grouped_rnea
                and self.fixed_path_rnea_cache is not None
                and callable(fixed_grouped)
            ):
                tau = fixed_grouped(
                    self.fixed_path_rnea_cache,
                    qd,
                    qdd,
                )
            elif self.fixed_path_rnea_cache is not None and callable(fixed_borrowed):
                tau = fixed_borrowed(
                    self.fixed_path_rnea_cache,
                    qd,
                    qdd,
                )
            elif self.borrowed_rnea and callable(borrowed):
                q_batch = self.repeated_q_batch(batch)
                tau = borrowed(
                    q_batch,
                    qd,
                    qdd,
                    direct_native=True,
                )
            else:
                q_batch = self.repeated_q_batch(batch)
                tau = self.problem.dynamics.inverse_dynamics(
                    q_batch,
                    qd,
                    qdd,
                )
            use_fused_torque_final = bool(
                self.fused_torque_final_reductions
                and batch > 1
                and bool(self.problem.mode & ConstraintMode.TORQUE)
                and bool(self.problem.mode & ConstraintMode.TORQUE_RATE)
                and timing.device.type == "cuda"
                and timing.dtype == torch.float32
            )
            if use_fused_torque_final:
                from curobo_timeopt.retiming.kinematic_reductions_triton import (
                    fused_torque_final_validation_reductions,
                )

                (
                    torque,
                    torque_rate,
                    interval,
                    maximum,
                    feasible,
                    violating_intervals,
                ) = fused_torque_final_validation_reductions(
                    tau.contiguous(),
                    x.contiguous(),
                    dt.contiguous(),
                    interval.contiguous(),
                    velocity.contiguous(),
                    acceleration.contiguous(),
                    jerk.contiguous(),
                    self.problem.limits.torque,
                    self.problem.limits.torque_rate,
                    self.requested_endpoints,
                    tolerance=self.tolerance,
                )
                return DenseFeasibilityResult(
                    feasible,
                    maximum,
                    velocity,
                    acceleration,
                    jerk,
                    torque,
                    torque_rate,
                    violating_intervals,
                    interval,
                    x,
                    self.s,
                )

            if self.problem.mode & ConstraintMode.TORQUE:
                torque, local = node_excess(
                    tau,
                    self.problem.limits.torque,
                )
                interval = torch.maximum(interval, local)
            if self.problem.mode & ConstraintMode.TORQUE_RATE:
                tau_rate = torch.diff(tau, dim=1) / dt[..., None]
                rate_excess = torch.clamp_min(
                    tau_rate.abs()
                    - _expand_limit(
                        self.problem.limits.torque_rate,
                        tau_rate,
                    ),
                    0,
                ).amax(-1)
                torque_rate = rate_excess.amax(-1)
                interval = torch.maximum(interval, rate_excess)

        use_fused_final = bool(
            self.fused_final_reductions
            and batch > 1
            and timing.device.type == "cuda"
            and timing.dtype == torch.float32
        )
        if use_fused_final:
            from curobo_timeopt.retiming.kinematic_reductions_triton import (
                fused_final_validation_reductions,
            )

            maximum, feasible, violating_intervals = fused_final_validation_reductions(
                x.contiguous(),
                dt.contiguous(),
                interval.contiguous(),
                velocity.contiguous(),
                acceleration.contiguous(),
                jerk.contiguous(),
                torque.contiguous(),
                torque_rate.contiguous(),
                self.requested_endpoints,
                tolerance=self.tolerance,
            )
        else:
            negative = torch.clamp_min(-x, 0).amax(-1)
            endpoint = torch.maximum(
                (x[:, 0] - self.requested_endpoints[0]).abs(),
                (x[:, -1] - self.requested_endpoints[1]).abs(),
            )
            maximum = torch.stack(
                (
                    velocity,
                    acceleration,
                    jerk,
                    torque,
                    torque_rate,
                    negative,
                    endpoint,
                ),
                dim=0,
            ).amax(0)
            finite = (
                torch.isfinite(x).all(-1)
                & torch.isfinite(dt).all(-1)
                & (dt > 0).all(-1)
            )
            maximum = torch.where(
                finite,
                maximum,
                torch.full_like(maximum, torch.inf),
            )
            feasible = maximum <= self.tolerance
            violating_intervals = interval > self.tolerance

        return DenseFeasibilityResult(
            feasible,
            maximum,
            velocity,
            acceleration,
            jerk,
            torque,
            torque_rate,
            violating_intervals,
            interval,
            x,
            self.s,
        )


class CapturedDenseBSplineDirectRNEAValidator:
    """CUDA-graph replay of a fixed-size direct-RNEA validation batch."""

    def __init__(
        self,
        validator: PreparedDenseBSplineDirectRNEAValidator,
        batch_size: int,
        *,
        warmup: int = 3,
    ) -> None:
        if validator.problem.device.type != "cuda":
            raise ValueError("captured validation requires CUDA")
        if validator.problem.dtype != torch.float32:
            raise ValueError("captured validation requires float32")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if warmup < 1:
            raise ValueError("warmup must be positive")
        if not validator.borrowed_rnea:
            raise ValueError("captured direct validation requires borrowed native RNEA")
        self.validator = validator
        self.batch_size = int(batch_size)
        self.warmup = int(warmup)
        self.input = torch.empty(
            (self.batch_size, validator.coefficients),
            device=validator.problem.device,
            dtype=validator.problem.dtype,
        )
        self.graph: torch.cuda.CUDAGraph | None = None
        self.output: DenseFeasibilityResult | None = None

    @torch.no_grad()
    def _capture(self, example: Tensor) -> None:
        if example.shape != self.input.shape:
            raise ValueError(f"example must have shape {tuple(self.input.shape)}")
        self.input.copy_(example)
        device = self.input.device
        current = torch.cuda.current_stream(device)
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(current)
        with torch.cuda.stream(stream):
            for _ in range(self.warmup):
                self.validator.validate(self.input)
        current.wait_stream(stream)
        torch.cuda.synchronize(device)

        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.output = self.validator.validate(self.input)

    @torch.no_grad()
    def validate(self, timing: Tensor) -> DenseFeasibilityResult:
        if timing.ndim == 1:
            timing = timing[None]
        if (
            timing.shape != self.input.shape
            or timing.device != self.input.device
            or timing.dtype != self.input.dtype
        ):
            raise ValueError("captured timing batch shape/device/dtype changed")
        if self.graph is None:
            self._capture(timing)
        assert self.graph is not None
        assert self.output is not None
        self.input.copy_(timing)
        self.graph.replay()
        return self.output


_CAPTURED_DIRECT_VALIDATOR_CACHE: dict[
    tuple[int, int, int, int, float, int],
    CapturedDenseBSplineDirectRNEAValidator,
] = {}


def captured_direct_rnea_validator(
    problem: RetimingProblem,
    coefficients: int,
    *,
    degree: int,
    samples: int,
    tolerance: float,
    batch_size: int,
) -> CapturedDenseBSplineDirectRNEAValidator:
    """Return the current-path fixed-batch direct-RNEA CUDA graph."""
    key = (
        id(problem),
        int(coefficients),
        int(degree),
        int(samples),
        float(tolerance),
        int(batch_size),
    )
    cached = _CAPTURED_DIRECT_VALIDATOR_CACHE.get(key)
    if cached is not None:
        return cached

    _CAPTURED_DIRECT_VALIDATOR_CACHE.clear()
    base = PreparedDenseBSplineDirectRNEAValidator.prepare(
        problem,
        coefficients=coefficients,
        degree=degree,
        samples=samples,
        tolerance=tolerance,
        borrowed_rnea=True,
    )
    result = CapturedDenseBSplineDirectRNEAValidator(
        base,
        batch_size,
    )
    _CAPTURED_DIRECT_VALIDATOR_CACHE[key] = result
    return result


@dataclass(frozen=True)
class PreparedDenseBSplineValidator:
    """Prepared dense physical validator for a fixed path and K-coefficient timing spline.

    Path samples, path derivatives, timing B-spline basis matrices, and exact
    fixed-path rigid-body TOPP dynamics terms are precomputed once.  Online
    validation then contains only timing-profile algebra and reductions.
    """

    problem: RetimingProblem
    coefficients: int
    degree: int
    samples: int
    tolerance: float
    s: Tensor
    basis: BSplineBasis
    q: Tensor
    q_s: Tensor
    q_ss: Tensor
    q_sss: Tensor
    dynamics_terms: PathDynamicsTerms | None
    requested_endpoints: Tensor

    @classmethod
    @torch.no_grad()
    def prepare(
        cls,
        problem: RetimingProblem,
        coefficients: int,
        *,
        degree: int = 3,
        samples: int = 4097,
        tolerance: float = 5e-4,
    ) -> PreparedDenseBSplineValidator:
        if samples < problem.s.numel():
            raise ValueError(
                "dense validation grid cannot be coarser than the problem grid"
            )
        s = torch.linspace(
            0,
            1,
            samples,
            device=problem.device,
            dtype=problem.dtype,
        )
        basis = _cached_basis(s, coefficients, degree)
        if isinstance(problem.path, BSplinePath):
            q = problem.path.evaluate(s, 0, uniform_grid=True)
            q_s = problem.path.evaluate(s, 1, uniform_grid=True)
            q_ss = problem.path.evaluate(s, 2, uniform_grid=True)
            q_sss = problem.path.evaluate(s, 3, uniform_grid=True)
        else:
            q = problem.path.evaluate(s, 0)
            q_s = problem.path.evaluate(s, 1)
            q_ss = problem.path.evaluate(s, 2)
            q_sss = problem.path.evaluate(s, 3)

        dynamics_terms = None
        if problem.mode & (ConstraintMode.TORQUE | ConstraintMode.TORQUE_RATE):
            if problem.dynamics is None:
                raise ValueError("torque validation requires inverse dynamics")
            dynamics_terms = extract_path_dynamics(
                problem.dynamics,
                q,
                q_s,
                q_ss,
            )

        requested = s.new_tensor([problem.initial_speed**2, problem.final_speed**2])
        return cls(
            problem=problem,
            coefficients=coefficients,
            degree=degree,
            samples=samples,
            tolerance=tolerance,
            s=s,
            basis=basis,
            q=q,
            q_s=q_s,
            q_ss=q_ss,
            q_sss=q_sss,
            dynamics_terms=dynamics_terms,
            requested_endpoints=requested,
        )

    @torch.no_grad()
    def validate(self, timing: Tensor) -> DenseFeasibilityResult:
        if timing.ndim == 1:
            timing = timing[None]
        if (
            timing.ndim != 2
            or timing.shape[1] != self.coefficients
            or timing.device != self.problem.device
            or timing.dtype != self.problem.dtype
        ):
            raise ValueError(
                "timing must be [B,K] on the prepared problem device and dtype"
            )

        x = self.basis.evaluate(timing)
        x_s = self.basis.evaluate(timing, 1)
        x_ss = self.basis.evaluate(timing, 2)
        speed = torch.sqrt(torch.clamp_min(x, 0))
        sddot = 0.5 * x_s
        sdddot = 0.5 * x_ss * speed

        qd = self.q_s[None] * speed[..., None]
        qdd = self.q_ss[None] * x[..., None] + self.q_s[None] * sddot[..., None]
        qddd = (
            self.q_sss[None] * (speed**3)[..., None]
            + 3 * self.q_ss[None] * (sddot * speed)[..., None]
            + self.q_s[None] * sdddot[..., None]
        )
        dt = 2 * torch.diff(self.s) / (speed[:, :-1] + speed[:, 1:])

        batch = timing.shape[0]
        zeros = torch.zeros(
            batch,
            device=timing.device,
            dtype=timing.dtype,
        )
        interval = torch.zeros(
            (batch, self.samples - 1),
            device=timing.device,
            dtype=timing.dtype,
        )

        def node_excess(value: Tensor, limit: Tensor) -> tuple[Tensor, Tensor]:
            excess = torch.clamp_min(
                value.abs() - _expand_limit(limit, value),
                0,
            ).amax(-1)
            located = torch.maximum(excess[:, :-1], excess[:, 1:])
            return excess.amax(-1), located

        velocity = zeros
        acceleration = zeros
        jerk = zeros
        if self.problem.mode & ConstraintMode.VELOCITY:
            velocity, local = node_excess(
                qd,
                self.problem.limits.velocity,
            )
            interval = torch.maximum(interval, local)
        if self.problem.mode & ConstraintMode.ACCELERATION:
            acceleration, local = node_excess(
                qdd,
                self.problem.limits.acceleration,
            )
            interval = torch.maximum(interval, local)
        if self.problem.mode & ConstraintMode.JERK:
            jerk, local = node_excess(
                qddd,
                self.problem.limits.jerk,
            )
            interval = torch.maximum(interval, local)

        torque = zeros
        torque_rate = zeros
        if self.problem.mode & (ConstraintMode.TORQUE | ConstraintMode.TORQUE_RATE):
            if self.dynamics_terms is None:
                raise RuntimeError("prepared torque validation lacks dynamics terms")
            terms = self.dynamics_terms
            tau = (
                terms.m[None] * sddot[..., None]
                + terms.c[None] * x[..., None]
                + terms.g[None]
            )
            if self.problem.mode & ConstraintMode.TORQUE:
                torque, local = node_excess(
                    tau,
                    self.problem.limits.torque,
                )
                interval = torch.maximum(interval, local)
            if self.problem.mode & ConstraintMode.TORQUE_RATE:
                tau_rate = torch.diff(tau, dim=1) / dt[..., None]
                rate_excess = torch.clamp_min(
                    tau_rate.abs()
                    - _expand_limit(
                        self.problem.limits.torque_rate,
                        tau_rate,
                    ),
                    0,
                ).amax(-1)
                torque_rate = rate_excess.amax(-1)
                interval = torch.maximum(interval, rate_excess)

        negative = torch.clamp_min(-x, 0).amax(-1)
        endpoint = torch.maximum(
            (x[:, 0] - self.requested_endpoints[0]).abs(),
            (x[:, -1] - self.requested_endpoints[1]).abs(),
        )
        maximum = torch.stack(
            (
                velocity,
                acceleration,
                jerk,
                torque,
                torque_rate,
                negative,
                endpoint,
            ),
            dim=0,
        ).amax(0)
        finite = (
            torch.isfinite(x).all(-1) & torch.isfinite(dt).all(-1) & (dt > 0).all(-1)
        )
        maximum = torch.where(
            finite,
            maximum,
            torch.full_like(maximum, torch.inf),
        )
        return DenseFeasibilityResult(
            maximum <= self.tolerance,
            maximum,
            velocity,
            acceleration,
            jerk,
            torque,
            torque_rate,
            interval > self.tolerance,
            interval,
            x,
            self.s,
        )


@torch.no_grad()
def validate_timing_gpu(
    problem: RetimingProblem,
    timing: Tensor,
    *,
    representation: str = "full_grid",
    degree: int = 3,
    samples: int = 4097,
    tolerance: float = 5e-4,
    enforce_endpoint_speed: bool = True,
    torque_rate_mode: str = "sampled",
) -> DenseFeasibilityResult:
    """Evaluate the same physical quantities as the CPU dense validator."""
    if torque_rate_mode not in ("sampled", "analytic_jvp", "rnea_fd5", "fd5_fused"):
        raise ValueError(
            "torque_rate_mode must be 'sampled', 'analytic_jvp', 'rnea_fd5', "
            "or 'fd5_fused'"
        )
    if torque_rate_mode == "fd5_fused" and not callable(
        getattr(problem.dynamics, "inverse_dynamics_directional_fd5_fused", None)
    ):
        raise RuntimeError(
            "torque_rate_mode='fd5_fused' requires a dynamics backend with "
            "inverse_dynamics_directional_fd5_fused; no fallback is performed"
        )
    if timing.ndim == 1:
        timing = timing[None]
    if (
        timing.ndim != 2
        or timing.device != problem.device
        or timing.dtype != problem.dtype
    ):
        raise ValueError("timing must be [B,N/K] on the problem device and dtype")
    if samples < problem.s.numel():
        raise ValueError(
            "dense validation grid cannot be coarser than the problem grid"
        )
    dense_s = torch.linspace(0, 1, samples, device=timing.device, dtype=timing.dtype)
    if representation == "full_grid":
        if timing.shape[1] != problem.s.numel():
            raise ValueError("full-grid timing width must match problem.s")
        x = _linear_interpolate(problem.s, timing, dense_s)
        x_s, x_ss = _node_derivatives(dense_s, x)
    elif representation == "bspline":
        basis = _cached_basis(dense_s, timing.shape[1], degree)
        x = basis.evaluate(timing)
        x_s = basis.evaluate(timing, 1)
        x_ss = basis.evaluate(timing, 2)
    else:
        raise ValueError("representation must be full_grid or bspline")

    if isinstance(problem.path, BSplinePath):
        q = problem.path.evaluate(dense_s, 0, uniform_grid=True)[None]
        q_s = problem.path.evaluate(dense_s, 1, uniform_grid=True)[None]
        q_ss = problem.path.evaluate(dense_s, 2, uniform_grid=True)[None]
        q_sss = problem.path.evaluate(dense_s, 3, uniform_grid=True)[None]
    else:
        q = problem.path.evaluate(dense_s, 0)[None]
        q_s = problem.path.evaluate(dense_s, 1)[None]
        q_ss = problem.path.evaluate(dense_s, 2)[None]
        q_sss = problem.path.evaluate(dense_s, 3)[None]
    sdot = torch.sqrt(torch.clamp_min(x, 0))
    sddot = 0.5 * x_s
    sdddot = 0.5 * x_ss * sdot
    qd = q_s * sdot[..., None]
    qdd = q_ss * x[..., None] + q_s * sddot[..., None]
    analytic_qddd = (
        q_sss * (sdot**3)[..., None]
        + 3 * q_ss * (sddot * sdot)[..., None]
        + q_s * sdddot[..., None]
    )
    dt = 2 * torch.diff(dense_s) / (sdot[:, :-1] + sdot[:, 1:])
    batch = timing.shape[0]
    zeros = torch.zeros(batch, device=timing.device, dtype=timing.dtype)
    interval = torch.zeros(
        (batch, samples - 1), device=timing.device, dtype=timing.dtype
    )

    def node_excess(value: Tensor, limit: Tensor) -> tuple[Tensor, Tensor]:
        excess = torch.clamp_min(value.abs() - _expand_limit(limit, value), 0).amax(-1)
        located = torch.maximum(excess[:, :-1], excess[:, 1:])
        return excess.amax(-1), located

    velocity = zeros
    acceleration = zeros
    jerk = zeros
    if problem.mode & ConstraintMode.VELOCITY:
        velocity, local = node_excess(qd, problem.limits.velocity)
        interval = torch.maximum(interval, local)
    if problem.mode & ConstraintMode.ACCELERATION:
        acceleration, local = node_excess(qdd, problem.limits.acceleration)
        interval = torch.maximum(interval, local)
    if problem.mode & ConstraintMode.JERK:
        if representation == "full_grid":
            jerk_value = torch.diff(qdd, dim=1) / dt[..., None]
            rate_excess = torch.clamp_min(
                jerk_value.abs() - _expand_limit(problem.limits.jerk, jerk_value), 0
            ).amax(-1)
            jerk = rate_excess.amax(-1)
            interval = torch.maximum(interval, rate_excess)
        else:
            jerk, local = node_excess(analytic_qddd, problem.limits.jerk)
            interval = torch.maximum(interval, local)
    torque = zeros
    torque_rate = zeros
    if problem.mode & (ConstraintMode.TORQUE | ConstraintMode.TORQUE_RATE):
        assert problem.dynamics is not None
        tau = problem.dynamics.inverse_dynamics(q.expand(batch, -1, -1), qd, qdd)
        if problem.mode & ConstraintMode.TORQUE:
            torque, local = node_excess(tau, problem.limits.torque)
            interval = torch.maximum(interval, local)
        if problem.mode & ConstraintMode.TORQUE_RATE:
            if torque_rate_mode == "sampled":
                tau_rate = torch.diff(tau, dim=1) / dt[..., None]
            elif torque_rate_mode == "analytic_jvp":
                analytic_parts: list[Tensor] = []
                with torch.enable_grad():
                    for start in range(0, samples, 1024):
                        stop = min(start + 1024, samples)
                        analytic_parts.append(
                            torque_rate_jvp(
                                problem.dynamics,
                                q.expand(batch, -1, -1)[:, start:stop].contiguous(),
                                qd[:, start:stop].contiguous(),
                                qdd[:, start:stop].contiguous(),
                                analytic_qddd[:, start:stop].contiguous(),
                            ).tangent.detach()
                        )
                tau_rate = torch.cat(analytic_parts, dim=1)
            elif torque_rate_mode == "fd5_fused":
                fd5_fused = getattr(
                    problem.dynamics, "inverse_dynamics_directional_fd5_fused", None
                )
                if not callable(fd5_fused):
                    raise RuntimeError(
                        "torque_rate_mode='fd5_fused' requires a dynamics backend "
                        "with inverse_dynamics_directional_fd5_fused; no fallback "
                        "is performed"
                    )
                tau_rate = fd5_fused(q.expand(batch, -1, -1), qd, qdd, analytic_qddd)
            else:
                tau_rate = torque_rate_rnea_fd5(
                    problem.dynamics, q.expand(batch, -1, -1), qd, qdd, analytic_qddd
                )
            rate_excess = torch.clamp_min(
                tau_rate.abs() - _expand_limit(problem.limits.torque_rate, tau_rate), 0
            ).amax(-1)
            torque_rate = rate_excess.amax(-1)
            located_rate_excess = (
                torch.maximum(rate_excess[:, :-1], rate_excess[:, 1:])
                if rate_excess.shape[1] == samples
                else rate_excess
            )
            interval = torch.maximum(interval, located_rate_excess)
    negative = torch.clamp_min(-x, 0).amax(-1)
    requested = x.new_tensor([problem.initial_speed**2, problem.final_speed**2])
    endpoint = torch.maximum(
        (x[:, 0] - requested[0]).abs(),
        (x[:, -1] - requested[1]).abs(),
    )
    endpoint_term = endpoint if enforce_endpoint_speed else zeros
    maximum = torch.stack(
        (
            velocity,
            acceleration,
            jerk,
            torque,
            torque_rate,
            negative,
            endpoint_term,
        ),
        dim=0,
    ).amax(0)
    finite = torch.isfinite(x).all(-1) & torch.isfinite(dt).all(-1) & (dt > 0).all(-1)
    maximum = torch.where(finite, maximum, torch.full_like(maximum, torch.inf))
    return DenseFeasibilityResult(
        maximum <= tolerance,
        maximum,
        velocity,
        acceleration,
        jerk,
        torque,
        torque_rate,
        interval > tolerance,
        interval,
        x,
        dense_s,
    )
