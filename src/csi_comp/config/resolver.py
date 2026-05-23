"""Custom ${...} expression resolver for config YAML.

Supported tokens (innermost-first, fixed-point):
    ${path.to.key}              reference another value in the config
    ${env:VAR}                  environment variable
    ${include:path/to.yaml}     splice a yaml file (path resolved from base_dir)
    ${mul:a,b} ${add:a,b}
    ${sub:a,b} ${div:a,b}       arithmetic (args may themselves be ${...})
"""
from __future__ import annotations

import os
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Union

import yaml

_INNERMOST_RE = re.compile(r"\$\{([^${}]*)\}")
_MAX_PASSES = 64


class ResolveError(RuntimeError):
    pass


def _lookup_path(cfg: Any, path: str) -> Any:
    cur = cfg
    for tok in path.split("."):
        if isinstance(cur, list):
            try:
                cur = cur[int(tok)]
            except (ValueError, IndexError) as e:
                raise ResolveError(f"bad path segment {tok!r} in {path!r}") from e
        elif isinstance(cur, dict):
            if tok not in cur:
                raise ResolveError(f"path {path!r} not found in config")
            cur = cur[tok]
        else:
            raise ResolveError(
                f"cannot descend into {type(cur).__name__} at {tok!r} (path {path!r})"
            )
    return cur


def _as_number(x: Any) -> Union[int, float]:
    if isinstance(x, bool):
        raise ResolveError(f"expected number, got bool {x!r}")
    if isinstance(x, (int, float)):
        return x
    if isinstance(x, str):
        v = yaml.safe_load(x)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return v
    raise ResolveError(f"expected number, got {x!r}")


def _eval_expr(expr: str, cfg: Any, base_dir: Path) -> Any:
    expr = expr.strip()
    if not expr:
        raise ResolveError("empty ${} expression")
    if ":" in expr:
        func, _, args = expr.partition(":")
        func = func.strip()
        arg_list = [a.strip() for a in args.split(",")] if args else []
        if func == "env":
            if len(arg_list) != 1:
                raise ResolveError(f"env needs 1 arg, got {arg_list}")
            name = arg_list[0]
            if name not in os.environ:
                raise ResolveError(f"env var {name!r} not set")
            return os.environ[name]
        if func == "include":
            if len(arg_list) != 1:
                raise ResolveError(f"include needs 1 arg, got {arg_list}")
            p = (base_dir / arg_list[0]).resolve()
            with p.open("r") as f:
                return yaml.safe_load(f)
        if func in ("mul", "add", "sub", "div"):
            if len(arg_list) != 2:
                raise ResolveError(f"{func} needs 2 args, got {arg_list}")
            a = _as_number(arg_list[0])
            b = _as_number(arg_list[1])
            if func == "mul":
                return a * b
            if func == "add":
                return a + b
            if func == "sub":
                return a - b
            if func == "div":
                if b == 0:
                    raise ResolveError("division by zero")
                return a / b
        raise ResolveError(f"unknown function {func!r} in ${{{expr}}}")
    # bare path reference
    return _lookup_path(cfg, expr)


def _resolve_string(s: str, cfg: Any, base_dir: Path) -> Any:
    for _ in range(_MAX_PASSES):
        m = _INNERMOST_RE.search(s)
        if not m:
            return s
        value = _eval_expr(m.group(1), cfg, base_dir)
        if m.start() == 0 and m.end() == len(s):
            # Single-token: preserve non-string types. If the looked-up value
            # is itself a string with another ${...}, loop on it.
            if isinstance(value, str) and _INNERMOST_RE.search(value):
                s = value
                continue
            return value
        # Interpolation inside a larger string → stringify
        s = s[: m.start()] + str(value) + s[m.end() :]
    raise ResolveError(f"resolution did not converge: {s!r}")


def _walk(node: Any, cfg: Any, base_dir: Path) -> Any:
    if isinstance(node, dict):
        return {k: _walk(v, cfg, base_dir) for k, v in node.items()}
    if isinstance(node, list):
        return [_walk(v, cfg, base_dir) for v in node]
    if isinstance(node, str):
        return _resolve_string(node, cfg, base_dir)
    return node


def resolve(cfg: dict, base_dir: Union[str, Path] = ".") -> dict:
    """Resolve all ${...} expressions in the config.

    Runs a fixed-point loop because resolved values may themselves reference
    other resolved values. Raises ResolveError on cycle / unknown reference.
    """
    base = Path(base_dir)
    out = deepcopy(cfg)
    for _ in range(_MAX_PASSES):
        new = _walk(out, out, base)
        if new == out:
            return new
        out = new
    raise ResolveError("config resolution did not converge")
