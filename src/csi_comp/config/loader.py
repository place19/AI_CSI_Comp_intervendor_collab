"""YAML config loading + CLI override merging."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable, List, Sequence, Union

import yaml

from .resolver import resolve

_SEG_RE = re.compile(r"^([^\[\]\.]+)((?:\[\d+\])*)$")
_IDX_RE = re.compile(r"\[(\d+)\]")


def _parse_path(path: str) -> List[Union[str, int]]:
    if not path:
        raise ValueError("empty override path")
    tokens: List[Union[str, int]] = []
    for seg in path.split("."):
        m = _SEG_RE.match(seg)
        if not m:
            raise ValueError(f"malformed override path: {path!r}")
        tokens.append(m.group(1))
        for idx_m in _IDX_RE.finditer(m.group(2)):
            tokens.append(int(idx_m.group(1)))
    return tokens


def _set_by_path(root: Any, tokens: Sequence[Union[str, int]], value: Any) -> None:
    cur = root
    for t in tokens[:-1]:
        if isinstance(t, int):
            if not isinstance(cur, list):
                raise TypeError(f"expected list for index [{t}], got {type(cur).__name__}")
            cur = cur[t]
        else:
            if not isinstance(cur, dict):
                raise TypeError(f"expected dict for key {t!r}, got {type(cur).__name__}")
            if t not in cur:
                cur[t] = {}
            cur = cur[t]
    last = tokens[-1]
    if isinstance(last, int):
        if not isinstance(cur, list):
            raise TypeError(f"expected list for index [{last}]")
        cur[last] = value
    else:
        if not isinstance(cur, dict):
            raise TypeError(f"expected dict for key {last!r}")
        cur[last] = value


def apply_overrides(cfg: dict, overrides: Iterable[str]) -> dict:
    """Apply CLI overrides like 'training.epochs=50' or 'model.blocks[0].channels=64'.
    The right-hand side is parsed as a YAML scalar."""
    for ov in overrides or ():
        if "=" not in ov:
            raise ValueError(f"override must be key=value, got {ov!r}")
        path, raw_val = ov.split("=", 1)
        tokens = _parse_path(path.strip())
        value = yaml.safe_load(raw_val)
        _set_by_path(cfg, tokens, value)
    return cfg


def load_config(path: Union[str, Path], overrides: Iterable[str] = ()) -> dict:
    """Load a YAML file and apply CLI overrides. Expressions are NOT resolved here."""
    p = Path(path)
    with p.open("r") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise TypeError(f"top-level config must be a mapping, got {type(cfg).__name__}")
    apply_overrides(cfg, overrides)
    return cfg


def load_and_resolve(path: Union[str, Path], overrides: Iterable[str] = ()) -> dict:
    """Convenience: load + overrides + ${...} resolution. base_dir = parent of the yaml."""
    p = Path(path)
    cfg = load_config(p, overrides)
    return resolve(cfg, base_dir=p.parent)
