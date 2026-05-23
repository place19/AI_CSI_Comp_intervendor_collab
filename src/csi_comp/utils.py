"""Tiny shared helpers."""
from __future__ import annotations

import inspect
from typing import Any


def filter_kwargs(callable_or_cls, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Keep only kwargs whose names appear in the target signature.

    Lets a single config block forward to factories that each understand a
    different subset of the keys. If the target's signature includes
    `**kwargs`, everything is forwarded.
    """
    try:
        sig = inspect.signature(callable_or_cls)
    except (TypeError, ValueError):
        return kwargs
    accepts_var_kw = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )
    if accepts_var_kw:
        return kwargs
    known = {p.name for p in sig.parameters.values()}
    return {k: v for k, v in kwargs.items() if k in known}
