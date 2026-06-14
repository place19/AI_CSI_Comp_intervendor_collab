"""Tiny shared helpers."""
from __future__ import annotations

import inspect
import struct
from typing import Any


def parse_scale(value: Any) -> float:
    """Parse a scale value given either as a number or as a hex string.

    A numeric `value` passes straight through as `float(value)`. A *string*
    `value` is interpreted as the **little-endian IEEE-754 float64 bit pattern**
    (16 hex digits = 8 bytes), with an optional ``0x`` prefix and surrounding
    whitespace tolerated — e.g. ``1/128 == 0.0078125`` is ``"0x000000000000803F"``.
    This lets a config carry the exact double a tool emitted without decimal
    round-trip loss (used for the per-component encoder-input scales, which come
    from phase-augmentation tooling). Endianness is little-endian to match how
    that tooling serialises the bytes.
    """
    if isinstance(value, str):
        s = value.strip()
        if s[:2].lower() == "0x":
            s = s[2:]
        try:
            raw = bytes.fromhex(s)
        except ValueError as e:
            raise ValueError(f"invalid hex scale {value!r}: {e}") from e
        if len(raw) != 8:
            raise ValueError(
                f"hex scale {value!r} must encode a float64 bit pattern "
                f"(8 bytes / 16 hex digits); got {len(raw)} bytes"
            )
        return struct.unpack("<d", raw)[0]
    return float(value)


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
