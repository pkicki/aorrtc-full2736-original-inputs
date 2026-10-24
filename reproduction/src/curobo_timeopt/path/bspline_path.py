"""Torch-native B-spline geometric paths with analytic derivatives."""

from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict

import torch
from torch import Tensor

from .base import validate_query


_CPU_DERIVATIVE_BASIS_CACHE: dict[
    tuple[torch.dtype, int, int, int, tuple[float, ...], tuple[float, ...]], Tensor
] = {}
_UNIFORM_DERIVATIVE_BASIS_CACHE: dict[
    tuple[str, torch.dtype, int, int, int, int], Tensor
] = {}
_NONUNIFORM_UNIFORM_GRID_DERIVATIVE_BASIS_CACHE: OrderedDict[
    tuple[str, torch.dtype, int, int, int, int, tuple[float, ...]], Tensor
] = OrderedDict()
_NONUNIFORM_UNIFORM_GRID_DERIVATIVE_BASIS_CACHE_LIMIT = 32
_NONUNIFORM_QUERY_DERIVATIVE_BASIS_CACHE: OrderedDict[
    tuple[str, torch.dtype, str, str, int, int, int], Tensor
] = OrderedDict()
_NONUNIFORM_QUERY_DERIVATIVE_BASIS_CACHE_MAX_BYTES = 128 * 1024 * 1024
_NONUNIFORM_QUERY_DERIVATIVE_BASIS_CACHE_BYTES = 0
_NONUNIFORM_QUERY_DERIVATIVE_BASIS_CACHE_HITS = 0
_NONUNIFORM_QUERY_DERIVATIVE_BASIS_CACHE_MISSES = 0
_NONUNIFORM_QUERY_DERIVATIVE_BASIS_CACHE_LOCK = threading.Lock()


def nonuniform_query_derivative_basis_cache_info() -> dict[str, int]:
    """Return bounded opt-in path-basis cache statistics."""
    with _NONUNIFORM_QUERY_DERIVATIVE_BASIS_CACHE_LOCK:
        return {
            "entries": len(_NONUNIFORM_QUERY_DERIVATIVE_BASIS_CACHE),
            "bytes": _NONUNIFORM_QUERY_DERIVATIVE_BASIS_CACHE_BYTES,
            "hits": _NONUNIFORM_QUERY_DERIVATIVE_BASIS_CACHE_HITS,
            "misses": _NONUNIFORM_QUERY_DERIVATIVE_BASIS_CACHE_MISSES,
            "max_bytes": _NONUNIFORM_QUERY_DERIVATIVE_BASIS_CACHE_MAX_BYTES,
        }


def reset_nonuniform_query_derivative_basis_cache_stats() -> None:
    """Reset counters without evicting reusable cached tensors."""
    global _NONUNIFORM_QUERY_DERIVATIVE_BASIS_CACHE_HITS
    global _NONUNIFORM_QUERY_DERIVATIVE_BASIS_CACHE_MISSES
    with _NONUNIFORM_QUERY_DERIVATIVE_BASIS_CACHE_LOCK:
        _NONUNIFORM_QUERY_DERIVATIVE_BASIS_CACHE_HITS = 0
        _NONUNIFORM_QUERY_DERIVATIVE_BASIS_CACHE_MISSES = 0


def clear_nonuniform_query_derivative_basis_cache() -> None:
    """Release all tensors held by the opt-in basis cache."""
    global _NONUNIFORM_QUERY_DERIVATIVE_BASIS_CACHE_BYTES
    global _NONUNIFORM_QUERY_DERIVATIVE_BASIS_CACHE_HITS
    global _NONUNIFORM_QUERY_DERIVATIVE_BASIS_CACHE_MISSES
    with _NONUNIFORM_QUERY_DERIVATIVE_BASIS_CACHE_LOCK:
        _NONUNIFORM_QUERY_DERIVATIVE_BASIS_CACHE.clear()
        _NONUNIFORM_QUERY_DERIVATIVE_BASIS_CACHE_BYTES = 0
        _NONUNIFORM_QUERY_DERIVATIVE_BASIS_CACHE_HITS = 0
        _NONUNIFORM_QUERY_DERIVATIVE_BASIS_CACHE_MISSES = 0


def _tensor_fingerprint(value: Tensor) -> str:
    raw = value.detach().contiguous().cpu().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def clamped_uniform_knots(
    control_point_count: int,
    degree: int,
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
) -> Tensor:
    """Construct an open uniform knot vector spanning ``[0, 1]``."""
    if degree < 1 or control_point_count <= degree:
        raise ValueError("control_point_count must be greater than a positive degree")
    interior_count = control_point_count - degree - 1
    interior = torch.linspace(0, 1, interior_count + 2, dtype=dtype, device=device)[1:-1]
    return torch.cat(
        (
            torch.zeros(degree + 1, dtype=dtype, device=device),
            interior,
            torch.ones(degree + 1, dtype=dtype, device=device),
        )
    )


class BSplinePath:
    """A non-rational B-spline path evaluated with Cox--de Boor recursion."""

    def __init__(
        self, control_points: Tensor, degree: int = 3, knots: Tensor | None = None
    ) -> None:
        if control_points.ndim != 2:
            raise ValueError("control_points must have shape (count, dof)")
        if not control_points.is_floating_point():
            raise TypeError("control_points must use a floating dtype")
        if not torch.all(torch.isfinite(control_points)):
            raise ValueError("control_points must contain only finite values")
        if degree < 1 or control_points.shape[0] <= degree:
            raise ValueError("control point count must be greater than a positive degree")
        expected_uniform_knots = clamped_uniform_knots(
            control_points.shape[0],
            degree,
            dtype=control_points.dtype,
            device=control_points.device,
        )
        uniform_knots = knots is None
        if knots is None:
            knots = expected_uniform_knots
        else:
            uniform_knots = torch.equal(knots, expected_uniform_knots)
        expected = control_points.shape[0] + degree + 1
        if knots.ndim != 1 or knots.numel() != expected:
            raise ValueError(f"knots must have shape ({expected},)")
        if knots.device != control_points.device or knots.dtype != control_points.dtype:
            raise ValueError("knots and control_points must have the same device and dtype")
        if not torch.all(torch.isfinite(knots)):
            raise ValueError("knots must contain only finite values")
        if not torch.all(knots[1:] >= knots[:-1]):
            raise ValueError("knots must be nondecreasing")
        if not torch.isclose(knots[degree], knots.new_tensor(0.0)) or not torch.isclose(
            knots[-degree - 1], knots.new_tensor(1.0)
        ):
            raise ValueError("the active knot domain must be [0, 1]")
        self.control_points = control_points
        self.degree = degree
        self.knots = knots
        self._uniform_knots = uniform_knots
        self._exact_query_grid_fingerprints: dict[int, tuple[Tensor, str]] = {}

    def _exact_query_grid_fingerprint(self, s: Tensor) -> str:
        entry = self._exact_query_grid_fingerprints.get(id(s))
        if entry is not None and entry[0] is s:
            return entry[1]
        fingerprint = _tensor_fingerprint(s)
        self._exact_query_grid_fingerprints[id(s)] = (s, fingerprint)
        return fingerprint

    @classmethod
    def _from_prevalidated(
        cls,
        control_points: Tensor,
        degree: int,
        knots: Tensor,
        *,
        uniform_knots: bool,
    ) -> "BSplinePath":
        """Construct an internally generated path without CUDA validation syncs.

        This private constructor is for trusted repository code that has already
        produced shape/device/dtype-consistent finite tensors. It intentionally
        skips the defensive reductions in __init__, which otherwise introduce
        several host/device synchronizations for tiny CUDA splines.
        """

        path = object.__new__(cls)
        path.control_points = control_points
        path.degree = degree
        path.knots = knots
        path._uniform_knots = uniform_knots
        path._exact_query_grid_fingerprints = {}
        return path

    @property
    def dof(self) -> int:
        return self.control_points.shape[1]

    def _derivative_representation(self, order: int) -> tuple[Tensor, Tensor, int]:
        points, knots, degree = self.control_points, self.knots, self.degree
        for _ in range(order):
            if degree == 0:
                return points.new_zeros((1, self.dof)), knots[1:-1], 0
            count = points.shape[0] - 1
            denominator = knots[degree + 1 : degree + 1 + count] - knots[1 : 1 + count]
            scale = torch.where(
                denominator > 0, degree / denominator, torch.zeros_like(denominator)
            )
            points = (points[1:] - points[:-1]) * scale[:, None]
            knots = knots[1:-1]
            degree -= 1
        return points, knots, degree

    @staticmethod
    def _basis(s: Tensor, knots: Tensor, degree: int, count: int) -> Tensor:
        flat = s.reshape(-1)
        basis = ((flat[:, None] >= knots[:-1]) & (flat[:, None] < knots[1:])).to(s.dtype)
        for p in range(1, degree + 1):
            next_count = basis.shape[1] - 1
            left_den = knots[p : p + next_count] - knots[:next_count]
            right_den = knots[p + 1 : p + 1 + next_count] - knots[1 : 1 + next_count]
            safe_left_den = torch.where(left_den > 0, left_den, torch.ones_like(left_den))
            safe_right_den = torch.where(right_den > 0, right_den, torch.ones_like(right_den))
            left = torch.where(
                left_den > 0,
                (flat[:, None] - knots[:next_count]) / safe_left_den * basis[:, :next_count],
                torch.zeros_like(basis[:, :next_count]),
            )
            right = torch.where(
                right_den > 0,
                (knots[p + 1 : p + 1 + next_count] - flat[:, None])
                / safe_right_den
                * basis[:, 1 : next_count + 1],
                torch.zeros_like(basis[:, :next_count]),
            )
            basis = left + right
        basis = basis[:, :count]
        # Half-open intervals make the repeated right knot empty. By continuity,
        # an open B-spline evaluates to its final control point at that endpoint.
        at_end = flat == knots[-degree - 1]
        endpoint = torch.nn.functional.one_hot(
            torch.tensor(count - 1, device=flat.device), num_classes=count
        ).to(basis.dtype)
        basis = torch.where(at_end[:, None], endpoint[None], basis)
        return basis

    def _cached_derivative_basis(
        self,
        s: Tensor,
        derivative_order: int,
        knots: Tensor,
        degree: int,
        count: int,
        *,
        uniform_grid: bool,
        cache_nonuniform_query: bool = False,
    ) -> Tensor:
        """Reuse only Cox--de Boor basis work, preserving stable derivative order.

        The original evaluator computes derivative control points first and
        then applies the lower-degree basis: B_d @ P_d. Caching a single matrix
        mapping original control points would instead evaluate (B_d @ D) @ P,
        which is algebraically equivalent but substantially less stable for
        high-order derivatives in float32 because D contains large alternating
        coefficients. Cache B_d only and retain the original P_d recursion.
        """

        cache_key = None
        cache = None
        if not s.requires_grad and uniform_grid and self._uniform_knots:
            cache_key = (
                str(s.device),
                s.dtype,
                s.numel(),
                self.control_points.shape[0],
                self.degree,
                derivative_order,
            )
            cache = _UNIFORM_DERIVATIVE_BASIS_CACHE
        elif not s.requires_grad and uniform_grid:
            # The query coordinates are the canonical uniform grid, but the
            # geometric knot vector may be nonuniform (for example VAMP path
            # splines). Cache the exact lower-degree Cox--de Boor basis by the
            # full source-knot fingerprint. The cache is bounded because each
            # dense basis can be several MiB at 8193 samples.
            knot_fingerprint = getattr(
                self, "_uniform_grid_nonuniform_knot_fingerprint", None
            )
            if knot_fingerprint is None:
                knot_fingerprint = tuple(
                    float(value)
                    for value in self.knots.detach().cpu().double().tolist()
                )
                self._uniform_grid_nonuniform_knot_fingerprint = (
                    knot_fingerprint
                )
            cache_key = (
                str(s.device),
                s.dtype,
                s.numel(),
                self.control_points.shape[0],
                self.degree,
                derivative_order,
                knot_fingerprint,
            )
            cache = _NONUNIFORM_UNIFORM_GRID_DERIVATIVE_BASIS_CACHE
        elif not s.requires_grad and s.device.type == "cpu":
            cache_key = (
                s.dtype,
                self.control_points.shape[0],
                self.degree,
                derivative_order,
                tuple(float(value) for value in self.knots.tolist()),
                tuple(float(value) for value in s.reshape(-1).tolist()),
            )
            cache = _CPU_DERIVATIVE_BASIS_CACHE
        if cache is not None and cache_key in cache:
            basis = cache[cache_key]
            if cache is _NONUNIFORM_UNIFORM_GRID_DERIVATIVE_BASIS_CACHE:
                cache.move_to_end(cache_key)
            return basis

        exact_key = None
        if (cache_nonuniform_query and cache is None
                and not s.requires_grad):
            exact_key = (
                str(s.device), s.dtype, self._exact_query_grid_fingerprint(s),
                _tensor_fingerprint(knots), self.control_points.shape[0],
                self.degree, derivative_order,
            )
            global _NONUNIFORM_QUERY_DERIVATIVE_BASIS_CACHE_HITS
            global _NONUNIFORM_QUERY_DERIVATIVE_BASIS_CACHE_MISSES
            with _NONUNIFORM_QUERY_DERIVATIVE_BASIS_CACHE_LOCK:
                basis = _NONUNIFORM_QUERY_DERIVATIVE_BASIS_CACHE.get(exact_key)
                if basis is not None:
                    _NONUNIFORM_QUERY_DERIVATIVE_BASIS_CACHE.move_to_end(exact_key)
                    _NONUNIFORM_QUERY_DERIVATIVE_BASIS_CACHE_HITS += 1
                    return basis
                _NONUNIFORM_QUERY_DERIVATIVE_BASIS_CACHE_MISSES += 1

        basis = self._basis(s, knots, degree, count)
        if cache is not None and cache_key is not None:
            cache[cache_key] = basis
            if cache is _NONUNIFORM_UNIFORM_GRID_DERIVATIVE_BASIS_CACHE:
                cache.move_to_end(cache_key)
                while (
                    len(cache)
                    > _NONUNIFORM_UNIFORM_GRID_DERIVATIVE_BASIS_CACHE_LIMIT
                ):
                    cache.popitem(last=False)
        if exact_key is not None:
            global _NONUNIFORM_QUERY_DERIVATIVE_BASIS_CACHE_BYTES
            size = basis.numel() * basis.element_size()
            if size <= _NONUNIFORM_QUERY_DERIVATIVE_BASIS_CACHE_MAX_BYTES:
                with _NONUNIFORM_QUERY_DERIVATIVE_BASIS_CACHE_LOCK:
                    previous = _NONUNIFORM_QUERY_DERIVATIVE_BASIS_CACHE.pop(
                        exact_key, None
                    )
                    if previous is not None:
                        _NONUNIFORM_QUERY_DERIVATIVE_BASIS_CACHE_BYTES -= (
                            previous.numel() * previous.element_size()
                        )
                    while (
                        _NONUNIFORM_QUERY_DERIVATIVE_BASIS_CACHE
                        and _NONUNIFORM_QUERY_DERIVATIVE_BASIS_CACHE_BYTES + size
                        > _NONUNIFORM_QUERY_DERIVATIVE_BASIS_CACHE_MAX_BYTES
                    ):
                        _, evicted = _NONUNIFORM_QUERY_DERIVATIVE_BASIS_CACHE.popitem(
                            last=False
                        )
                        _NONUNIFORM_QUERY_DERIVATIVE_BASIS_CACHE_BYTES -= (
                            evicted.numel() * evicted.element_size()
                        )
                    _NONUNIFORM_QUERY_DERIVATIVE_BASIS_CACHE[exact_key] = basis
                    _NONUNIFORM_QUERY_DERIVATIVE_BASIS_CACHE_BYTES += size
        return basis

    def evaluation_matrix(
        self,
        s: Tensor,
        derivative_order: int = 0,
        *,
        uniform_grid: bool = False,
        cache_nonuniform_query: bool = False,
    ) -> Tensor:
        """Matrix mapping original control points to one derivative.

        This helper is retained for diagnostics/tests. Runtime path evaluation
        intentionally does not use this combined matrix in order to preserve
        the numerically stable derivative-control-point evaluation order.
        """

        validate_query(s, derivative_order)
        if s.device != self.control_points.device or s.dtype != self.control_points.dtype:
            raise ValueError("query and path must have the same device and dtype")
        count = self.control_points.shape[0]
        if derivative_order > self.degree:
            return self.control_points.new_zeros((s.numel(), count))

        transform = torch.eye(
            count,
            dtype=self.control_points.dtype,
            device=self.control_points.device,
        )
        knots = self.knots
        degree = self.degree
        for _ in range(derivative_order):
            reduced_count = transform.shape[0] - 1
            denominator = (
                knots[degree + 1 : degree + 1 + reduced_count]
                - knots[1 : 1 + reduced_count]
            )
            scale = torch.where(
                denominator > 0,
                degree / denominator,
                torch.zeros_like(denominator),
            )
            transform = (transform[1:] - transform[:-1]) * scale[:, None]
            knots = knots[1:-1]
            degree -= 1

        basis = self._cached_derivative_basis(
            s,
            derivative_order,
            knots,
            degree,
            transform.shape[0],
            uniform_grid=uniform_grid,
            cache_nonuniform_query=cache_nonuniform_query,
        )
        return basis @ transform

    def evaluate_all(
        self,
        s: Tensor,
        max_order: int = 3,
        *,
        uniform_grid: bool = False,
        cache_nonuniform_query: bool = False,
    ) -> tuple[Tensor, ...]:
        """Evaluate orders 0..max_order with one derivative-control recursion."""

        validate_query(s, 0)
        if max_order < 0:
            raise ValueError("max_order must be nonnegative")
        if s.device != self.control_points.device or s.dtype != self.control_points.dtype:
            raise ValueError("query and path must have the same device and dtype")

        outputs: list[Tensor] = []
        points = self.control_points
        knots = self.knots
        degree = self.degree
        for order in range(max_order + 1):
            if order <= self.degree:
                basis = self._cached_derivative_basis(
                    s,
                    order,
                    knots,
                    degree,
                    points.shape[0],
                    uniform_grid=uniform_grid,
                    cache_nonuniform_query=cache_nonuniform_query,
                )
                values = basis @ points
            else:
                values = self.control_points.new_zeros((s.numel(), self.dof))
            outputs.append(values.reshape((*s.shape, self.dof)))

            if order == max_order:
                break
            if degree > 0:
                count = points.shape[0] - 1
                denominator = (
                    knots[degree + 1 : degree + 1 + count]
                    - knots[1 : 1 + count]
                )
                scale = torch.where(
                    denominator > 0,
                    degree / denominator,
                    torch.zeros_like(denominator),
                )
                points = (points[1:] - points[:-1]) * scale[:, None]
                knots = knots[1:-1]
                degree -= 1

        return tuple(outputs)

    def evaluate(
        self,
        s: Tensor,
        derivative_order: int = 0,
        *,
        uniform_grid: bool = False,
        cache_nonuniform_query: bool = False,
    ) -> Tensor:
        validate_query(s, derivative_order)
        if s.device != self.control_points.device or s.dtype != self.control_points.dtype:
            raise ValueError("query and path must have the same device and dtype")
        if derivative_order > self.degree:
            return self.control_points.new_zeros((*s.shape, self.dof))

        points, knots, degree = self._derivative_representation(derivative_order)
        basis = self._cached_derivative_basis(
            s,
            derivative_order,
            knots,
            degree,
            points.shape[0],
            uniform_grid=uniform_grid,
            cache_nonuniform_query=cache_nonuniform_query,
        )
        return (basis @ points).reshape((*s.shape, self.dof))
