"""Opt-in Kaserer_fixed: compact DP3 with analytical inverse-dynamics torque rate.

This is a research variant, not an author-code reproduction.  It keeps the
same (s,z) DP recursion and velocity/acceleration/torque/jerk constraints, but
replaces the baseline's interpolation-checkpoint torque secants with analytical
RNEA-JVP evaluation at *every* interpolation checkpoint, including the C2
endpoint intervals.  Requires an independently qualified analytical Pinocchio
bridge; never silently falls back to finite differences.
"""

from __future__ import annotations

from dataclasses import replace

from curobo_timeopt.baselines.kaserer_dp3 import (
    KasererDP3Config,
    KasererDP3Result,
    solve_kaserer_dp3,
)
from curobo_timeopt.retiming.problem import RetimingProblem


def solve_kaserer_fixed(
    problem: RetimingProblem,
    config: KasererDP3Config | None = None,
) -> KasererDP3Result:
    """Solve with analytical checkpoint torque rate using the AORRTC compact grid.

    Defaults are Ns=40, Nz=128, Nch=129, with unchanged 0.995 internal scale.
    An explicit config may override resolution, never the analytical rate mode.
    The result is not a dense/continuous feasibility certificate; independently
    re-audit on both event-aware S8193 and S32768 before reporting it.
    """
    cfg = config or KasererDP3Config(grid_samples=40, z_samples=128, check_points=129)
    return solve_kaserer_dp3(problem, replace(cfg, torque_rate_mode="analytic"))
