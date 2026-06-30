"""Small decorators used by API helpers."""

from __future__ import annotations

import functools
import time
from typing import Any, Callable, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


def timed(fn: F) -> F:
    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        result = fn(*args, **kwargs)
        elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
        if isinstance(result, dict):
            result.setdefault("elapsed_ms", elapsed_ms)
        return result

    return wrapper  # type: ignore[return-value]

