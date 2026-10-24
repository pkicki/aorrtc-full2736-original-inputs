"""Native CPU fixed-path dynamics preparation with Pinocchio.

This module intentionally contains no CUDA calls. It prepares the rigid-body
TOPP affinity

    tau(s) = m(s) * sddot + c(s) * sdot**2 + g(s)

on a CPU path grid using Pinocchio RNEA. The implementation is an experimental
online backend for the all-CPU elite retimer.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Sequence

import numpy as np
import torch
from torch import Tensor

from curobo_timeopt.dynamics import PathDynamicsTerms


def _pin():
    try:
        import pinocchio as pin
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Pinocchio CPU dynamics are unavailable. Install the optional "
            "CPU dependencies, for example with pip install pin."
        ) from exc
    return pin


@dataclass(frozen=True)
class PinocchioPreparationProfile:
    samples: int
    derivative_input_ms: float
    rnea_ms: float
    total_ms: float


class PinocchioCpuDynamics:
    """Fixed-base Pinocchio RNEA adapter with optional point-mass payload."""

    def __init__(
        self,
        model,
        *,
        joint_names: Sequence[str] | None = None,
        gravity: tuple[float, float, float] = (0.0, 0.0, -9.81),
        rnea_threads: int = 1,
    ) -> None:
        pin = _pin()
        self.model = model
        self.model.gravity.linear[:] = np.asarray(gravity, dtype=np.float64)
        self.data = self.model.createData()
        requested_threads = max(1, int(rnea_threads))
        self.rnea_max_threads = (
            int(pin.omp_get_max_threads())
            if hasattr(pin, "omp_get_max_threads")
            else requested_threads
        )
        self.rnea_threads = min(requested_threads, self.rnea_max_threads)
        self._parallel_rnea = (
            hasattr(pin, "ModelPool")
            and hasattr(pin, "rneaInParallel")
        )
        if self._parallel_rnea:
            try:
                self._model_pool = pin.ModelPool(
                    self.model,
                    self.rnea_threads,
                )
            except TypeError:
                self._model_pool = pin.ModelPool(self.model)
        else:
            self._model_pool = None
        self._parallel_capacity = 0
        self._parallel_q = None
        self._parallel_v = None
        self._parallel_a = None
        self._parallel_tau = None

        if joint_names is None:
            names: list[str] = []
            q_indices: list[int] = []
            v_indices: list[int] = []
            for jid in range(1, self.model.njoints):
                joint = self.model.joints[jid]
                if joint.nq == 0 and joint.nv == 0:
                    continue
                if joint.nq != 1 or joint.nv != 1:
                    raise ValueError(
                        "automatic CPU RNEA mapping currently requires 1-DoF joints; "
                        "pass a reduced fixed-base arm model"
                    )
                names.append(str(self.model.names[jid]))
                q_indices.append(int(self.model.idx_qs[jid]))
                v_indices.append(int(self.model.idx_vs[jid]))
        else:
            names = [str(name) for name in joint_names]
            q_indices = []
            v_indices = []
            for name in names:
                jid = int(self.model.getJointId(name))
                if jid <= 0 or jid >= self.model.njoints:
                    raise ValueError(f"joint {name!r} is not present in Pinocchio model")
                joint = self.model.joints[jid]
                if joint.nq != 1 or joint.nv != 1:
                    raise ValueError(
                        f"joint {name!r} is not a scalar configuration/velocity joint"
                    )
                q_indices.append(int(self.model.idx_qs[jid]))
                v_indices.append(int(self.model.idx_vs[jid]))

        self.joint_names = tuple(names)
        self.q_indices = np.asarray(q_indices, dtype=np.int64)
        self.v_indices = np.asarray(v_indices, dtype=np.int64)
        self.dof = len(self.joint_names)
        if self.dof < 1:
            raise ValueError("Pinocchio model has no selected actuated joints")
        self._direct_q_layout = (
            self.model.nq == self.dof
            and np.array_equal(
                self.q_indices,
                np.arange(self.dof, dtype=np.int64),
            )
        )
        self._direct_v_layout = (
            self.model.nv == self.dof
            and np.array_equal(
                self.v_indices,
                np.arange(self.dof, dtype=np.int64),
            )
        )

        self._q = pin.neutral(self.model).astype(np.float64, copy=True)
        self._v = np.zeros(self.model.nv, dtype=np.float64)
        self._a = np.zeros(self.model.nv, dtype=np.float64)

    @classmethod
    def from_curobo_kinematics_params(
        cls,
        kinematics_params,
        *,
        joint_names: Sequence[str] | None = None,
        gravity: tuple[float, float, float] = (0.0, 0.0, -9.81),
        rnea_threads: int = 1,
    ) -> "PinocchioCpuDynamics":
        """Build Pinocchio from cuRobo's processed KinematicsParams.

        cuRobo's own RNEA-vs-Pinocchio tests use this export route. It preserves
        locked-joint transforms, extra fixed links, COMs, inertias, and any
        benchmark-time modifications such as attached payload mass.
        """
        pin = _pin()
        with TemporaryDirectory(prefix="curobo_timeopt_pin_") as tmp:
            urdf_path = Path(tmp) / "curobo_dynamics.urdf"
            kinematics_params.export_to_urdf(
                robot_name="curobo_timeopt_dynamics",
                output_path=str(urdf_path),
                include_spheres=False,
            )
            model = pin.buildModelFromUrdf(str(urdf_path))
        return cls(
            model,
            joint_names=joint_names,
            gravity=gravity,
            rnea_threads=rnea_threads,
        )

    @classmethod
    def from_urdf(
        cls,
        urdf: str | Path,
        *,
        joint_names: Sequence[str] | None = None,
        payload_frame: str | None = None,
        payload_mass: float = 0.0,
        payload_com: tuple[float, float, float] = (0.0, 0.0, 0.0),
        gravity: tuple[float, float, float] = (0.0, 0.0, -9.81),
        reduce_to_joint_names: bool = True,
        locked_joint_positions: dict[str, float] | None = None,
        rnea_threads: int = 1,
    ) -> "PinocchioCpuDynamics":
        pin = _pin()
        urdf_path = str(Path(urdf).expanduser().resolve())
        model = pin.buildModelFromUrdf(urdf_path)

        requested = None if joint_names is None else tuple(str(v) for v in joint_names)
        if requested and reduce_to_joint_names:
            wanted = set(requested)
            lock: list[int] = []
            reference = pin.neutral(model)
            locked_values = locked_joint_positions or {}
            for jid in range(1, model.njoints):
                joint = model.joints[jid]
                if joint.nq == 0 and joint.nv == 0:
                    continue
                name = str(model.names[jid])
                if name not in wanted:
                    lock.append(jid)
                    if name in locked_values:
                        if joint.nq != 1:
                            raise ValueError(
                                f"locked joint {name!r} is not scalar"
                            )
                        reference[int(model.idx_qs[jid])] = float(
                            locked_values[name]
                        )
            if lock:
                model = pin.buildReducedModel(model, lock, reference)

        if payload_mass < 0.0:
            raise ValueError("payload_mass must be nonnegative")
        if payload_mass > 0.0:
            if not payload_frame:
                raise ValueError("payload_frame is required when payload_mass > 0")
            frame_id = int(model.getFrameId(str(payload_frame)))
            if frame_id >= len(model.frames):
                raise ValueError(f"frame {payload_frame!r} is not present in model")
            frame = model.frames[frame_id]
            parent_joint = int(
                getattr(frame, "parentJoint", getattr(frame, "parent", -1))
            )
            if parent_joint <= 0:
                raise ValueError(
                    f"payload frame {payload_frame!r} has no movable parent joint"
                )
            local = pin.Inertia(
                float(payload_mass),
                np.asarray(payload_com, dtype=np.float64),
                np.zeros((3, 3), dtype=np.float64),
            )
            model.inertias[parent_joint] += frame.placement.act(local)

        return cls(
            model,
            joint_names=requested,
            gravity=gravity,
            rnea_threads=rnea_threads,
        )

    def _scatter(
        self,
        q: np.ndarray,
        v: np.ndarray,
        a: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        self._q[self.q_indices] = q
        self._v.fill(0.0)
        self._a.fill(0.0)
        self._v[self.v_indices] = v
        self._a[self.v_indices] = a
        return self._q, self._v, self._a

    def _ensure_parallel_workspace(self, batch: int) -> None:
        if not self._parallel_rnea:
            return
        if batch <= self._parallel_capacity:
            return
        capacity = max(
            int(batch),
            max(64, 2 * self._parallel_capacity),
        )
        # Eigen::MatrixXd is column-major. Keeping these arrays Fortran ordered
        # lets eigenpy expose them without hidden row/column-major copies.
        self._parallel_q = np.empty(
            (self.model.nq, capacity),
            dtype=np.float64,
            order="F",
        )
        self._parallel_v = np.empty(
            (self.model.nv, capacity),
            dtype=np.float64,
            order="F",
        )
        self._parallel_a = np.empty(
            (self.model.nv, capacity),
            dtype=np.float64,
            order="F",
        )
        self._parallel_tau = np.empty(
            (self.model.nv, capacity),
            dtype=np.float64,
            order="F",
        )
        self._parallel_capacity = capacity

    def _run_parallel_workspace(self, batch: int) -> np.ndarray:
        if not self._parallel_rnea or self._model_pool is None:
            raise RuntimeError("parallel RNEA workspace is unavailable")
        assert self._parallel_q is not None
        assert self._parallel_v is not None
        assert self._parallel_a is not None
        assert self._parallel_tau is not None
        pin = _pin()
        q_full = self._parallel_q[:, :batch]
        v_full = self._parallel_v[:, :batch]
        a_full = self._parallel_a[:, :batch]
        tau_full = self._parallel_tau[:, :batch]
        pin.rneaInParallel(
            min(self.rnea_threads, batch),
            self._model_pool,
            q_full,
            v_full,
            a_full,
            tau_full,
        )
        return tau_full

    def inverse_dynamics_numpy(
        self,
        q: np.ndarray,
        qd: np.ndarray,
        qdd: np.ndarray,
    ) -> np.ndarray:
        pin = _pin()
        q = np.asarray(q, dtype=np.float64)
        qd = np.asarray(qd, dtype=np.float64)
        qdd = np.asarray(qdd, dtype=np.float64)
        if q.ndim != 2 or q.shape != qd.shape or q.shape != qdd.shape:
            raise ValueError("q/qd/qdd must have identical [samples,dof] shapes")
        if q.shape[1] != self.dof:
            raise ValueError(
                f"expected {self.dof} dynamics DoFs, got {q.shape[1]}"
            )
        if self._parallel_rnea:
            batch = q.shape[0]
            self._ensure_parallel_workspace(batch)
            assert self._parallel_q is not None
            assert self._parallel_v is not None
            assert self._parallel_a is not None
            q_full = self._parallel_q[:, :batch]
            v_full = self._parallel_v[:, :batch]
            a_full = self._parallel_a[:, :batch]
            if self._direct_q_layout:
                q_full[:, :] = q.T
            else:
                q_full[:] = self._q[:, None]
                q_full[self.q_indices, :] = q.T
            v_full.fill(0.0)
            a_full.fill(0.0)
            if self._direct_v_layout:
                v_full[:, :] = qd.T
                a_full[:, :] = qdd.T
            else:
                v_full[self.v_indices, :] = qd.T
                a_full[self.v_indices, :] = qdd.T
            tau = self._run_parallel_workspace(batch)
            selected = tau if self._direct_v_layout else tau[self.v_indices, :]
            return selected.T.copy()

        out = np.empty_like(q)
        for i in range(q.shape[0]):
            q_full, v_full, a_full = self._scatter(q[i], qd[i], qdd[i])
            tau = pin.rnea(self.model, self.data, q_full, v_full, a_full)
            out[i] = np.asarray(tau)[self.v_indices]
        return out

    def _path_terms_numpy(
        self,
        q_np: np.ndarray,
        qs_np: np.ndarray,
        qss_np: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate all three fixed-path probes with no per-call large buffers."""
        samples = int(q_np.shape[0])
        if self._parallel_rnea:
            batch = 3 * samples
            self._ensure_parallel_workspace(batch)
            assert self._parallel_q is not None
            assert self._parallel_v is not None
            assert self._parallel_a is not None

            q_full = self._parallel_q[:, :batch]
            v_full = self._parallel_v[:, :batch]
            a_full = self._parallel_a[:, :batch]
            q_t = q_np.T
            if self._direct_q_layout:
                q_full[:, :samples] = q_t
                q_full[:, samples : 2 * samples] = q_t
                q_full[:, 2 * samples : 3 * samples] = q_t
            else:
                q_full[:] = self._q[:, None]
                q_full[self.q_indices, :samples] = q_t
                q_full[self.q_indices, samples : 2 * samples] = q_t
                q_full[self.q_indices, 2 * samples : 3 * samples] = q_t

            v_full.fill(0.0)
            a_full.fill(0.0)

            # Probe layout matches cuRobo:
            # [g(q), tau(q,0,q_s), tau(q,q_s,q_ss)].
            if self._direct_v_layout:
                a_full[:, samples : 2 * samples] = qs_np.T
                v_full[:, 2 * samples : 3 * samples] = qs_np.T
                a_full[:, 2 * samples : 3 * samples] = qss_np.T
            else:
                a_full[self.v_indices, samples : 2 * samples] = qs_np.T
                v_full[self.v_indices, 2 * samples : 3 * samples] = qs_np.T
                a_full[self.v_indices, 2 * samples : 3 * samples] = qss_np.T

            tau = self._run_parallel_workspace(batch)
            selected = tau if self._direct_v_layout else tau[self.v_indices, :]
            gravity = selected[:, :samples].T
            unit_acc = selected[:, samples : 2 * samples].T
            unit_speed = selected[:, 2 * samples : 3 * samples].T
            return gravity, unit_acc, unit_speed

        zeros = np.zeros_like(q_np)
        gravity = self.inverse_dynamics_numpy(q_np, zeros, zeros)
        unit_acc = self.inverse_dynamics_numpy(q_np, zeros, qs_np)
        unit_speed = self.inverse_dynamics_numpy(q_np, qs_np, qss_np)
        return gravity, unit_acc, unit_speed

    def extract_path_dynamics_terms(
        self,
        q: Tensor,
        q_s: Tensor,
        q_ss: Tensor,
        *,
        dtype: torch.dtype = torch.float32,
    ) -> PathDynamicsTerms:
        if q.device.type != "cpu" or q_s.device.type != "cpu" or q_ss.device.type != "cpu":
            raise ValueError("Pinocchio CPU dynamics require CPU path tensors")
        if q.shape != q_s.shape or q.shape != q_ss.shape:
            raise ValueError("q/q_s/q_ss shapes must match")
        if q.ndim != 2 or q.shape[1] != self.dof:
            raise ValueError("path tensors must have shape [samples,dof]")

        q_np = q.detach().to(dtype=torch.float64).contiguous().numpy()
        qs_np = q_s.detach().to(dtype=torch.float64).contiguous().numpy()
        qss_np = q_ss.detach().to(dtype=torch.float64).contiguous().numpy()
        gravity, unit_acc, unit_speed = self._path_terms_numpy(
            q_np, qs_np, qss_np
        )

        return PathDynamicsTerms(
            m=torch.from_numpy(unit_acc - gravity).to(dtype=dtype),
            c=torch.from_numpy(unit_speed - gravity).to(dtype=dtype),
            g=torch.from_numpy(gravity).to(dtype=dtype),
        )

    def profile_path_dynamics_terms(
        self,
        q: Tensor,
        q_s: Tensor,
        q_ss: Tensor,
        *,
        dtype: torch.dtype = torch.float32,
    ) -> tuple[PathDynamicsTerms, PinocchioPreparationProfile]:
        started = perf_counter()
        convert_started = perf_counter()
        q_cpu = q.detach().to(device="cpu", dtype=torch.float64).contiguous()
        qs_cpu = q_s.detach().to(device="cpu", dtype=torch.float64).contiguous()
        qss_cpu = q_ss.detach().to(device="cpu", dtype=torch.float64).contiguous()
        converted = perf_counter()

        q_np = q_cpu.numpy()
        qs_np = qs_cpu.numpy()
        qss_np = qss_cpu.numpy()
        rnea_started = perf_counter()
        gravity, unit_acc, unit_speed = self._path_terms_numpy(
            q_np, qs_np, qss_np
        )
        rnea_done = perf_counter()

        terms = PathDynamicsTerms(
            m=torch.from_numpy(unit_acc - gravity).to(dtype=dtype),
            c=torch.from_numpy(unit_speed - gravity).to(dtype=dtype),
            g=torch.from_numpy(gravity).to(dtype=dtype),
        )
        done = perf_counter()
        return terms, PinocchioPreparationProfile(
            samples=int(q.shape[0]),
            derivative_input_ms=(converted - convert_started) * 1e3,
            rnea_ms=(rnea_done - rnea_started) * 1e3,
            total_ms=(done - started) * 1e3,
        )


__all__ = [
    "PinocchioCpuDynamics",
    "PinocchioPreparationProfile",
]
