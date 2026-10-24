"""Device-aware clamped B-spline basis and analytic derivatives."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch
from torch import Tensor

_CPU_BASIS_CACHE: dict[
    tuple[torch.dtype, int, int, tuple[float, ...]], BSplineBasis
] = {}
_UNIFORM_BASIS_CACHE: dict[
    tuple[str, torch.dtype, int, int, int], BSplineBasis
] = {}
_CUDA_QUERY_BASIS_CACHE: dict[
    tuple[str, torch.dtype, int, int, int, int], BSplineBasis
] = {}
_CUDA_VECTORIZED_BASIS_CACHE: dict[
    tuple[str, torch.dtype, int, int, int, int], BSplineBasis
] = {}


def clear_cuda_path_basis_cache() -> int:
    """Release cached non-uniform CUDA bases for completed path lifetimes.

    Cached bases retain both their query grids and all derivative matrices.
    Cohort evaluators that process many distinct path grids should clear this
    cache between paths; prepared route objects keep their own references, so
    clearing the lookup does not invalidate a route that is still in use.
    The returned count is diagnostic only.
    """
    count = len(_CUDA_QUERY_BASIS_CACHE) + len(_CUDA_VECTORIZED_BASIS_CACHE)
    _CUDA_QUERY_BASIS_CACHE.clear()
    _CUDA_VECTORIZED_BASIS_CACHE.clear()
    return count


def prune_cuda_vectorized_basis_cache(retain_grids: Iterable[Tensor]) -> int:
    """Explicit quiescent prune of the *vectorized* global lookup dictionary.

    This must ONLY be called by an exclusively owned, single-workspace CUDA
    service when no other active or concurrent owner needs lookup residency.
    Existing prepared routes and captured graph objects retain references to
    their bases; removing a dictionary key never mutates those basis tensors.
    A future request can recreate an evicted basis on demand.

    Never invoke this at a request boundary shared by unrelated workspaces;
    there is intentionally no implicit eviction or automatic production hook.
    The unrelated _CUDA_QUERY_BASIS_CACHE is left untouched.
    """
    owned = {id(grid) for grid in retain_grids}
    doomed = [
        key for key, basis in _CUDA_VECTORIZED_BASIS_CACHE.items()
        if id(basis.s) not in owned
    ]
    for key in doomed:
        del _CUDA_VECTORIZED_BASIS_CACHE[key]
    return len(doomed)


def clamped_uniform_knots(
    coefficients: int, degree: int, *, device: torch.device, dtype: torch.dtype
) -> Tensor:
    if degree < 1 or coefficients <= degree:
        raise ValueError("coefficients must be larger than the positive degree")
    interior_count = coefficients - degree - 1
    interior = torch.linspace(0, 1, interior_count + 2, device=device, dtype=dtype)[1:-1]
    return torch.cat(
        (
            torch.zeros(degree + 1, device=device, dtype=dtype),
            interior,
            torch.ones(degree + 1, device=device, dtype=dtype),
        )
    )


def _basis_at_degree(s: Tensor, knots: Tensor, degree: int) -> Tensor:
    count0 = knots.numel() - 1
    columns = []
    for i in range(count0):
        inside = (s >= knots[i]) & (s < knots[i + 1])
        columns.append(inside.to(s.dtype))
    basis = torch.stack(columns, dim=-1)
    basis[..., -1] = torch.where(s == knots[-1], torch.ones_like(s), basis[..., -1])
    for p in range(1, degree + 1):
        next_columns = []
        for i in range(count0 - p):
            left_den = knots[i + p] - knots[i]
            right_den = knots[i + p + 1] - knots[i + 1]
            safe_left = torch.where(left_den > 0, left_den, torch.ones_like(left_den))
            safe_right = torch.where(right_den > 0, right_den, torch.ones_like(right_den))
            left = torch.where(
                left_den > 0,
                (s - knots[i]) / safe_left * basis[..., i],
                torch.zeros_like(s),
            )
            right = torch.where(
                right_den > 0,
                (knots[i + p + 1] - s) / safe_right * basis[..., i + 1],
                torch.zeros_like(s),
            )
            next_columns.append(left + right)
        basis = torch.stack(next_columns, dim=-1)
    at_end = s == knots[-1]
    # When this routine evaluates a lower-degree basis for a derivative, the
    # original clamped knot vector has excess copies of the final knot. Its
    # trailing basis functions then have zero-width support; select the last
    # basis function with nonempty support, the one-sided limit from inside.
    support = knots[degree + 1 :] > knots[: -(degree + 1)]
    reverse_rank = torch.cumsum(support.flip(0).to(torch.int64), dim=0).flip(0)
    last_supported = support & (reverse_rank == 1)
    endpoint = last_supported.to(basis.dtype).expand_as(basis)
    basis = torch.where(at_end[..., None], endpoint, basis)
    return basis


def _basis_at_degree_vectorized(s: Tensor, knots: Tensor, degree: int) -> Tensor:
    """Cox-de Boor recurrence with coefficient-wise tensor operations.

    The reference implementation above launches separate small tensor
    operations for every basis column. This form is intended for large CUDA
    query grids, where those launches dominate the O(samples * coefficients)
    arithmetic.
    """
    if s.ndim != 1:
        raise ValueError("vectorized basis construction requires a one-dimensional grid")
    count0 = knots.numel() - 1
    basis = (
        (s[:, None] >= knots[:-1][None, :])
        & (s[:, None] < knots[1:][None, :])
    ).to(s.dtype)
    basis = torch.cat(
        (
            basis[:, :-1],
            torch.where(
                (s == knots[-1])[:, None],
                torch.ones_like(basis[:, -1:]),
                basis[:, -1:],
            ),
        ),
        dim=1,
    )
    for order in range(1, degree + 1):
        count = count0 - order
        left_den = knots[order : order + count] - knots[:count]
        right_den = knots[order + 1 : order + 1 + count] - knots[1 : count + 1]
        safe_left = torch.where(left_den > 0, left_den, torch.ones_like(left_den))
        safe_right = torch.where(right_den > 0, right_den, torch.ones_like(right_den))
        left = torch.where(
            left_den[None, :] > 0,
            (s[:, None] - knots[:count][None, :])
            / safe_left[None, :]
            * basis[:, :count],
            torch.zeros_like(basis[:, :count]),
        )
        right = torch.where(
            right_den[None, :] > 0,
            (knots[order + 1 : order + 1 + count][None, :] - s[:, None])
            / safe_right[None, :]
            * basis[:, 1 : count + 1],
            torch.zeros_like(basis[:, 1 : count + 1]),
        )
        basis = left + right

    at_end = s == knots[-1]
    support = knots[degree + 1 :] > knots[: -(degree + 1)]
    reverse_rank = torch.cumsum(support.flip(0).to(torch.int64), dim=0).flip(0)
    last_supported = support & (reverse_rank == 1)
    endpoint = last_supported.to(basis.dtype).expand_as(basis)
    return torch.where(at_end[:, None], endpoint, basis)


def _derivative_matrix(knots: Tensor, degree: int) -> Tensor:
    output_count = knots.numel() - degree - 1
    lower_count = output_count + 1
    matrix = torch.zeros((lower_count, output_count), device=knots.device, dtype=knots.dtype)
    index = torch.arange(output_count, device=knots.device)
    left_den = knots[degree : degree + output_count] - knots[:output_count]
    right_den = knots[degree + 1 : degree + 1 + output_count] - knots[1 : output_count + 1]
    safe_left = torch.where(left_den > 0, left_den, torch.ones_like(left_den))
    safe_right = torch.where(right_den > 0, right_den, torch.ones_like(right_den))
    matrix[index, index] = torch.where(left_den > 0, degree / safe_left, 0.0)
    matrix[index + 1, index] = torch.where(right_den > 0, -degree / safe_right, 0.0)
    return matrix


@dataclass(frozen=True)
class BSplineBasis:
    """Precomputed matrices mapping coefficients to ``x, x', x''``."""

    s: Tensor
    knots: Tensor
    degree: int
    B: Tensor
    B_s: Tensor
    B_ss: Tensor

    @classmethod
    def create(cls, s: Tensor, coefficients: int, degree: int = 3) -> BSplineBasis:
        knots = clamped_uniform_knots(coefficients, degree, device=s.device, dtype=s.dtype)
        B = _basis_at_degree(s, knots, degree)
        lower = _basis_at_degree(s, knots, degree - 1)
        d1 = _derivative_matrix(knots, degree)
        B_s = lower @ d1
        if degree >= 2:
            lower2 = _basis_at_degree(s, knots, degree - 2)
            d2 = _derivative_matrix(knots, degree - 1) @ d1
            B_ss = lower2 @ d2
        else:
            B_ss = torch.zeros_like(B)
        return cls(s, knots, degree, B, B_s, B_ss)

    @classmethod
    def create_vectorized(
        cls, s: Tensor, coefficients: int, degree: int = 3
    ) -> BSplineBasis:
        """Construct a basis with vectorized Cox-de Boor CUDA operations."""
        if degree < 1 or coefficients <= degree:
            raise ValueError("coefficients must be larger than the positive degree")
        knots = clamped_uniform_knots(coefficients, degree, device=s.device, dtype=s.dtype)
        B = _basis_at_degree_vectorized(s, knots, degree)
        lower = _basis_at_degree_vectorized(s, knots, degree - 1)
        d1 = _derivative_matrix(knots, degree)
        B_s = lower @ d1
        if degree >= 2:
            lower2 = _basis_at_degree_vectorized(s, knots, degree - 2)
            d2 = _derivative_matrix(knots, degree - 1) @ d1
            B_ss = lower2 @ d2
        else:
            B_ss = torch.zeros_like(B)
        return cls(s, knots, degree, B, B_s, B_ss)

    @classmethod
    def create_vectorized_cached(
        cls, s: Tensor, coefficients: int, degree: int = 3
    ) -> BSplineBasis:
        """Cache vectorized bases by the exact CUDA grid tensor identity."""
        if not s.requires_grad and s.device.type == "cuda":
            key = (
                str(s.device),
                s.dtype,
                int(s.data_ptr()),
                s.numel(),
                coefficients,
                degree,
            )
            cached = _CUDA_VECTORIZED_BASIS_CACHE.get(key)
            if cached is None:
                cached = cls.create_vectorized(s, coefficients, degree)
                _CUDA_VECTORIZED_BASIS_CACHE[key] = cached
            return cached
        return cls.create_vectorized(s, coefficients, degree)

    @classmethod
    def create_cached(
        cls,
        s: Tensor,
        coefficients: int,
        degree: int = 3,
        *,
        uniform_grid: bool = False,
    ) -> BSplineBasis:
        """Return a reusable basis when the query grid structure is invariant.

        CPU span-native grids are keyed by their exact values. Fixed linspace
        grids can use a structural key and therefore avoid host inspection.
        Returned tensors must be treated as immutable.
        """

        if not s.requires_grad and uniform_grid:
            key = (str(s.device), s.dtype, s.numel(), coefficients, degree)
            cached = _UNIFORM_BASIS_CACHE.get(key)
            if cached is None:
                cached = cls.create(s, coefficients, degree)
                _UNIFORM_BASIS_CACHE[key] = cached
            return cached

        if not s.requires_grad and s.device.type == "cpu":
            key = (
                s.dtype,
                coefficients,
                degree,
                tuple(float(value) for value in s.tolist()),
            )
            cached = _CPU_BASIS_CACHE.get(key)
            if cached is None:
                cached = cls.create(s, coefficients, degree)
                _CPU_BASIS_CACHE[key] = cached
            return cached

        if not s.requires_grad and s.device.type == "cuda":
            # Exact non-uniform CUDA grids cannot be hashed without a host
            # synchronization. Span-native structural grid caching reuses the
            # same tensor object across trajectories, so its stable data_ptr is
            # sufficient as an in-process cache identity.
            key = (
                str(s.device),
                s.dtype,
                int(s.data_ptr()),
                s.numel(),
                coefficients,
                degree,
            )
            cached = _CUDA_QUERY_BASIS_CACHE.get(key)
            if cached is None:
                cached = cls.create(s, coefficients, degree)
                _CUDA_QUERY_BASIS_CACHE[key] = cached
            return cached

        return cls.create(s, coefficients, degree)

    @property
    def coefficients(self) -> int:
        return self.B.shape[-1]

    def evaluate(self, coefficients: Tensor, derivative_order: int = 0) -> Tensor:
        matrices = (self.B, self.B_s, self.B_ss)
        if derivative_order not in (0, 1, 2):
            raise ValueError("timing basis supports derivatives 0, 1, and 2")
        return coefficients @ matrices[derivative_order].T
