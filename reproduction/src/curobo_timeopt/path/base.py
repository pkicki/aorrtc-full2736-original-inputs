"""Interfaces and validation helpers for geometric joint-space paths."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch
from torch import Tensor


@runtime_checkable
class GeometricPath(Protocol):
    """A timing-independent path :math:`q(s)`, with ``s`` in ``[0, 1]``."""

    @property
    def dof(self) -> int:
        """Number of robot joints represented by the path."""

    def evaluate(self, s: Tensor, derivative_order: int = 0) -> Tensor:
        """Evaluate ``q`` or one of its first three path derivatives."""


def validate_query(s: Tensor, derivative_order: int) -> None:
    """Validate the common path evaluation contract."""
    if derivative_order not in (0, 1, 2, 3):
        raise ValueError("derivative_order must be one of 0, 1, 2, or 3")
    if not s.is_floating_point():
        raise TypeError("path coordinates must have a floating dtype")
    if s.numel():
        in_domain = torch.all((s >= 0) & (s <= 1))
        if s.device.type == "cuda":
            # Preserve bounds checking without synchronizing every path
            # coefficient evaluation in GPU SLP construction.
            torch._assert_async(in_domain, "path coordinates must lie in [0, 1]")
        elif not bool(in_domain):
            raise ValueError("path coordinates must lie in [0, 1]")
