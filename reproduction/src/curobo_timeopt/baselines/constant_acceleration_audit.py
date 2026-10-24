"""Independent physical reconstruction for native second-order profiles."""

from __future__ import annotations

import torch

from curobo_timeopt.retiming.gpu_dense_validation import validate_dense_path_profile_gpu


@torch.no_grad()
def audit_constant_acceleration(problem, result, *, samples: int, tolerance: float):
    """Include path knots and both representable sides of every solver switch.

    No event coalescing is allowed: it could delete the acceleration on one
    side of a switch. This is a sampled C2 certificate, not a C3 certificate.
    """
    s, x, a = result.s, result.x, result.path_acceleration
    if s.ndim != 1 or s.numel() < 2 or x.shape != s.shape:
        raise ValueError("s/x must be matching one-dimensional profiles")
    if a.shape != s[:-1].shape:
        raise ValueError("path acceleration must have one value per interval")
    if samples < s.numel():
        raise ValueError("dense audit cannot be coarser than the solver grid")
    if not all(bool(torch.isfinite(v).all()) for v in (s, x, a)):
        raise ValueError("profile must be finite")
    if not bool((s[1:] > s[:-1]).all()) or float(s[0]) != 0 or float(s[-1]) != 1:
        raise ValueError("solver grid must increase and span [0, 1]")
    # Detect malformed adapters instead of auditing disconnected segments.
    reconstructed = x[:-1] + 2 * a * torch.diff(s)
    eps = torch.finfo(x.dtype).eps
    if not torch.allclose(reconstructed, x[1:], rtol=64 * eps, atol=64 * eps):
        raise ValueError("inconsistent constant-acceleration profile continuity")
    knots = getattr(problem.path, "knots", s.new_empty(0))
    events = torch.unique(torch.cat((s, knots)), sorted=True)
    events = events[(events >= 0) & (events <= 1)]
    left = torch.nextafter(events[events > 0], s.new_tensor(-torch.inf))
    right = torch.nextafter(events[events < 1], s.new_tensor(torch.inf))
    dense = torch.linspace(0, 1, samples, device=s.device, dtype=s.dtype)
    grid = torch.unique(torch.cat((dense, events, left, right)), sorted=True)
    interval = (torch.searchsorted(s, grid, right=True) - 1).clamp(0, s.numel() - 2)
    acceleration = a[interval]
    profile_x = x[interval] + 2 * acceleration * (grid - s[interval])
    # The closed final endpoint is a native returned speed state. Evaluating
    # the left polynomial there can cancel to a tiny negative number even
    # when x[-1] is exactly zero. Preserve that exact state, not a tolerance
    # or a clamp on the interior/one-sided samples.
    profile_x = torch.where(grid == s[-1], x[-1], profile_x)
    return validate_dense_path_profile_gpu(
        problem,
        grid,
        profile_x,
        acceleration,
        torch.zeros_like(profile_x),
        tolerance=tolerance,
    )
