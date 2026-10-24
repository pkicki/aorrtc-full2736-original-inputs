#!/usr/bin/env python3
"""Public independent one-path Full-C3 replay from the byte-identical frozen Kaserer source.

The DP solver/native extension/geometry/Pinocchio backends under src/ are
copied byte-for-byte from private research commit 98e9f780bebc6f94b62a4ee8072da230dcecd1fd.
This file is new publication glue, reproducing the original analytical bridge,
path fingerprint and five-family physical audit. It is NOT the frozen runner.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

from curobo_timeopt.baselines.kaserer_dp3 import (
    KasererDP3Config,
    _profile_on_dense_grid,
    solve_kaserer_dp3,
)
from curobo_timeopt.baselines.kaserer_dp3_fast import warm_kaserer_dp3_fast_extension
from curobo_timeopt.baselines.kaserer_fixed import solve_kaserer_fixed
from curobo_timeopt.path import BSplinePath
from curobo_timeopt.retiming.bspline.cpu_pinocchio import PinocchioCpuDynamics
from curobo_timeopt.retiming.bspline.pinocchio_torch_bridge import PinocchioTorchBridge
from curobo_timeopt.retiming.problem import RetimingProblem
from curobo_timeopt.types import ConstraintMode, RobotLimits

FROZEN_SHA = "98e9f780bebc6f94b62a4ee8072da230dcecd1fd"
VAMP_SHA = "f6de6a72e0725ba08e0829f0a34261a3f465aed7"
MODEL_SHA = {
    "panda": "f5853b4ff48619cc6980a470f5793d45b03afd45e32207b4e2f8cf6809ff7d17",
    "ur5": "eed1d28c2856f8b7c399477343da474231fac8eb43e3094253884f01dc800325",
}
TRACK_MODEL = {"mbm_panda": "panda", "mpinets_panda": "panda", "mbm_ur5": "ur5"}
DEFAULT_ID = "motion_benchmaker/panda/bookshelf_small/0008"
DEFAULT_DURATION = 1.463802100095559
DEFAULT_FINGERPRINT = "db4bfc2faad5ddb9ecddf94df7979d88c3ae59f778d54af7fec9899f0fe9f645"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for data in iter(lambda: f.read(1 << 20), b""):
            h.update(data)
    return h.hexdigest()


def original_fingerprint(row: dict) -> tuple[str, str]:
    """Exact original 'tro_source_geometry_fingerprint_v1' byte contract."""
    identity = {
        "problem_id": f"{row['dataset']}/{row['robot']}/{row['group']}/{row['index']:04d}",
        "joint_names": row["joint_names"],
        "source_planner": row["source_planner"],
        "source": row["source"],
        "spline_degree": row["spline_degree"],
        "protocol": "tro_source_geometry_fingerprint_v1",
    }
    if row["spline_degree"] != 5:
        raise ValueError("frozen source spline degree is not five")
    h = hashlib.sha256(json.dumps(
        identity, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode())
    for label, name in (("q", "q"), ("s", "s"), ("spline_controls", "spline_control_points"), ("spline_knots", "spline_knots")):
        x = row[name]
        if x.device.type != "cpu" or x.dtype not in (torch.float32, torch.float64):
            raise ValueError(f"invalid original source dtype/device: {name}")
        a = x.detach().contiguous().numpy()
        a = a.astype(a.dtype.newbyteorder("<"), copy=False)
        header = {"name": label, "dtype": str(a.dtype), "shape": list(a.shape)}
        h.update((json.dumps(header, sort_keys=True) + "\n").encode())
        h.update(a.tobytes(order="C"))
    fingerprint = h.hexdigest()
    stored = row.get("metadata", {}).get("path_fingerprint_sha256")
    if stored is not None and stored != fingerprint:
        raise ValueError("source-stored geometric fingerprint differs from independent computation")
    return identity["problem_id"], fingerprint


class AnalyticPinocchioBridge(PinocchioTorchBridge):
    """Original source-model RNEA/JVP contraction semantics (not FD in time)."""

    kaserer_analytic_precompute_mode = "reference"

    def kaserer_fast_analytic_coefficients(self, q, qs, qss, qsss):
        from curobo_timeopt.baselines.kaserer_pinocchio_precompute import (
            precompute_kaserer_analytic_coefficients,
        )
        return precompute_kaserer_analytic_coefficients(self.backend, q, qs, qss, qsss)

    @property
    def torque_rate_backend_name(self):
        return "pinocchio_computeRNEADerivatives_contraction"

    def inverse_dynamics_directional_derivative(self, q, v, a, qt, vt, at):
        return self.inverse_dynamics_with_directional_derivative(q, v, a, qt, vt, at)[1]

    @torch.no_grad()
    def inverse_dynamics_with_directional_derivative(self, q, v, a, qt, vt, at):
        import pinocchio as pin

        values = (q, v, a, qt, vt, at)
        if any(t.shape != q.shape or t.device.type != "cpu" for t in values):
            raise ValueError("analytic audit requires matching CPU states/tangents")
        b = self.backend
        arrays = [t.detach().double().reshape(-1, b.dof).numpy() for t in values]
        tau, tangent = [], []
        data = b.model.createData()
        for qi, vi, ai, qti, vti, ati in zip(*arrays, strict=True):
            full_q = pin.neutral(b.model)
            full_v, full_a = np.zeros(b.model.nv), np.zeros(b.model.nv)
            full_q[b.q_indices], full_v[b.v_indices], full_a[b.v_indices] = qi, vi, ai
            tau.append(pin.rnea(b.model, data, full_q, full_v, full_a)[b.v_indices].copy())
            dq, dv, da = pin.computeRNEADerivatives(b.model, data, full_q, full_v, full_a)
            rows = np.ix_(b.v_indices, b.v_indices)
            tangent.append(dq[rows] @ qti + dv[rows] @ vti + da[rows] @ ati)
        return tuple(
            torch.as_tensor(np.asarray(x), dtype=q.dtype).reshape(q.shape)
            for x in (tau, tangent)
        )


def physical_audit(problem: RetimingProblem, result, samples: int) -> dict:
    """Original event-aware five-family S8193/S32768 CPU RNEA/JVP audit."""
    base = torch.linspace(0, 1, samples, dtype=torch.float64)
    events = torch.unique(torch.cat((problem.path.knots.double(), result.s.double())))
    left = torch.nextafter(events, torch.full_like(events, -torch.inf))
    right = torch.nextafter(events, torch.full_like(events, torch.inf))
    grid = torch.unique(torch.cat((base, events, left, right)).clamp(0, 1), sorted=True)
    x, xs, xss = _profile_on_dense_grid(result, grid)
    if not torch.isfinite(torch.stack((x, xs, xss))).all() or float(x.min()) < -1e-12:
        raise ValueError("nonfinite or negative execution profile")
    speed = x.clamp_min(0).sqrt()
    q, qs, qss, qsss = [problem.path.evaluate(grid, d) for d in range(4)]
    v = qs * speed[:, None]
    a = qss * x[:, None] + qs * (0.5 * xs)[:, None]
    jerk = qsss * speed.pow(3)[:, None] + 1.5 * qss * (xs * speed)[:, None] + 0.5 * qs * (xss * speed)[:, None]
    tau, rate = problem.dynamics.inverse_dynamics_with_directional_derivative(q, v, a, v, a, jerk)
    utilizations = {}
    for name, values in zip(
        ("velocity", "acceleration", "jerk", "torque", "torque_rate"),
        (v, a, jerk, tau, rate),
        strict=True,
    ):
        normalized = values.abs() / getattr(problem.limits, name)
        utilizations[name] = float(normalized.max())
    sw = result.s[1:-1]
    before = torch.nextafter(sw, torch.full_like(sw, -torch.inf))
    after = torch.nextafter(sw, torch.full_like(sw, torch.inf))
    _, ls, _ = _profile_on_dense_grid(result, before)
    _, rs, _ = _profile_on_dense_grid(result, after)
    slope_jump = float((rs - ls).abs().max()) if ls.numel() else 0.0
    feasible = (
        float(speed[[0, -1]].abs().max()) == 0.0
        and slope_jump <= 1e-8
        and all(v <= 1.0 for v in utilizations.values())
    )
    return {
        "base_samples": samples,
        "actual_samples": len(grid),
        "max_utilization": utilizations,
        "max_slope_jump": slope_jump,
        "endpoint_speed": speed[[0, -1]].tolist(),
        "strict_five_family_certified": feasible,
    }


def run(args: argparse.Namespace) -> dict:
    root = args.release_root.resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    mapping = manifest["files"]
    relative = f"inputs/{args.track}/aorrtc_all.pt"
    item = mapping[relative]
    cache = root / relative
    if cache.stat().st_size != item["size_bytes"] or sha256_file(cache) != item["sha256"]:
        raise ValueError("original cache not hydrated or SHA256 mismatch")
    if args.vamp is None:
        raise ValueError("provide --vamp checkout at the public pinned VAMP source SHA")
    robot_name = TRACK_MODEL[args.track]
    urdf = args.vamp / "resources" / robot_name / (robot_name + ".urdf")
    if sha256_file(urdf) != MODEL_SHA[robot_name]:
        raise ValueError("wrong VAMP URDF hash or model/source version")

    torch.set_num_threads(1)
    payload = torch.load(cache, map_location="cpu", weights_only=False)
    rows = payload["records"]
    selected = None
    selected_fp = None
    for raw in rows:
        pid, fp = original_fingerprint(raw)
        if pid == args.problem_id:
            selected, selected_fp = raw, fp
            break
    if selected is None:
        raise ValueError(f"problem ID not in frozen original cache: {args.problem_id}")
    if args.problem_id == DEFAULT_ID and selected_fp != DEFAULT_FINGERPRINT:
        raise ValueError("reference problem fingerprint differs from frozen source")

    projected = json.loads((root / "limits/full_c3_limits.public_projection.json").read_text())
    vectors = projected["robots"][robot_name]
    rate_time = 0.1
    # Exactly the archived rise-time convention, never silently relax limits.
    limits_data = {**vectors, "jerk": [a / rate_time for a in vectors["acceleration"]],
                   "torque_rate": [t / rate_time for t in vectors["torque"]]}

    backend = PinocchioCpuDynamics.from_urdf(
        urdf, joint_names=selected["joint_names"], rnea_threads=1,
    )
    dynamics = AnalyticPinocchioBridge(backend)
    dynamics.kaserer_analytic_precompute_mode = (
        "pinocchio_coefficients_fast_v1" if args.method == "dp3_fixed_fast" else "reference"
    )
    path = BSplinePath(selected["spline_control_points"].double(), 5,
                       selected["spline_knots"].double())
    problem = RetimingProblem(
        path, RobotLimits.from_optional(backend.dof, **limits_data),
        torch.linspace(0, 1, 1025, dtype=torch.float64), dynamics,
        ConstraintMode.all(), initial_speed=0.0, final_speed=0.0,
        feasibility_tolerance=0.0,
    )
    cfg = KasererDP3Config(
        grid_samples=40, z_samples=128, check_points=129, safety_scale=0.995,
        torque_rate_mode="finite_difference" if args.method == "kaserer_dp3" else "analytic",
    )
    warm_kaserer_dp3_fast_extension()  # one-time native compilation excluded from request
    started = time.perf_counter()
    result = solve_kaserer_dp3(problem, cfg) if args.method == "kaserer_dp3" else solve_kaserer_fixed(problem, cfg)
    request_ms = 1000 * (time.perf_counter() - started)
    if not result.success:
        raise RuntimeError(f"solver did not return certified expected profile: {result.status}")
    audits = {f"S{n}": physical_audit(problem, result, n) for n in (8193, 32768)}
    replay = {
        "frozen_solver_source_commit": FROZEN_SHA, "vamp_revision": VAMP_SHA,
        "method": args.method, "track": args.track, "problem_id": args.problem_id,
        "path_fingerprint": selected_fp, "solver_status": result.status,
        "duration_s": float(result.duration), "request_wall_ms": request_ms,
        "audits": audits, "five_family_certified": all(
            a["strict_five_family_certified"] for a in audits.values()
        ),
    }
    if args.method == "dp3_fixed_fast" and args.problem_id == DEFAULT_ID:
        replay["expected_frozen_duration_s"] = DEFAULT_DURATION
        replay["duration_difference_s"] = float(result.duration) - DEFAULT_DURATION
        if not math.isclose(float(result.duration), DEFAULT_DURATION, rel_tol=1e-8, abs_tol=1e-8):
            raise AssertionError(f"frozen duration mismatch: {replay['duration_difference_s']}")
        if not replay["five_family_certified"]:
            raise AssertionError("original frozen test ID failed independent dual-grid C3 certification")
    return replay


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--release-root", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--vamp", type=Path, required=True)
    p.add_argument("--track", choices=TRACK_MODEL, default="mbm_panda")
    p.add_argument("--problem-id", default=DEFAULT_ID)
    p.add_argument("--method", choices=("dp3_fixed_fast", "kaserer_fixed", "kaserer_dp3"), default="dp3_fixed_fast")
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    receipt = run(args)
    result = json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n"
    print(result)
    if args.output is not None:
        args.output.write_text(result)


if __name__ == "__main__":
    main()
