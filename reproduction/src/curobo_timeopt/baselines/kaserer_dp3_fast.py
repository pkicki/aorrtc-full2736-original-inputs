"""Lazy loader for compiled paper-Kaserer DP3 kernels."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any


@lru_cache(maxsize=1)
def get_kaserer_dp3_fast_extension() -> Any:
    from torch.utils.cpp_extension import load

    source = Path(__file__).with_name("_kaserer_dp3_fast.cpp")
    return load(
        name="curobo_timeopt_kaserer_dp3_fast",
        sources=[str(source)],
        extra_cflags=["-O3", "-DNDEBUG"],
        verbose=False,
    )


def warm_kaserer_dp3_fast_extension() -> None:
    get_kaserer_dp3_fast_extension()


__all__ = [
    "get_kaserer_dp3_fast_extension",
    "warm_kaserer_dp3_fast_extension",
]
