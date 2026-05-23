import textwrap
from pathlib import Path

import pytest

from csi_comp.config import (
    ResolveError,
    apply_overrides,
    load_and_resolve,
    load_config,
    resolve,
)


def _write(tmp: Path, name: str, content: str) -> Path:
    p = tmp / name
    p.write_text(textwrap.dedent(content))
    return p


def test_load_basic(tmp_path):
    p = _write(tmp_path, "cfg.yaml", "a: 1\nb: [2, 3]\n")
    cfg = load_config(p)
    assert cfg == {"a": 1, "b": [2, 3]}


def test_apply_overrides_scalar():
    cfg = {"training": {"lr": 1e-3, "epochs": 10}}
    apply_overrides(cfg, ["training.epochs=50", "training.lr=2.5e-4"])
    assert cfg["training"]["epochs"] == 50
    assert cfg["training"]["lr"] == pytest.approx(2.5e-4)


def test_apply_overrides_list_index():
    cfg = {"model": {"blocks": [{"channels": 32}, {"channels": 64}]}}
    apply_overrides(cfg, ["model.blocks[0].channels=16"])
    assert cfg["model"]["blocks"][0]["channels"] == 16
    assert cfg["model"]["blocks"][1]["channels"] == 64


def test_apply_overrides_yaml_value():
    cfg = {"x": None}
    apply_overrides(cfg, ["x=[1, 2, 3]"])
    assert cfg["x"] == [1, 2, 3]


def test_apply_overrides_bool_and_null():
    cfg = {"f": False, "y": "old"}
    apply_overrides(cfg, ["f=true", "y=null"])
    assert cfg["f"] is True
    assert cfg["y"] is None


def test_apply_overrides_creates_missing_dict_path():
    cfg = {}
    apply_overrides(cfg, ["a.b.c=7"])
    assert cfg == {"a": {"b": {"c": 7}}}


def test_apply_overrides_bad_path_raises():
    with pytest.raises(ValueError):
        apply_overrides({}, ["bogus_no_equals"])


def test_resolve_path_reference():
    cfg = {"a": 5, "b": "${a}"}
    out = resolve(cfg)
    assert out == {"a": 5, "b": 5}


def test_resolve_nested_path():
    cfg = {"data": {"max_subband": 52, "max_port": 32}, "x": "${data.max_subband}"}
    out = resolve(cfg)
    assert out["x"] == 52


def test_resolve_mul_nested():
    cfg = {"data": {"S": 52, "P": 32}, "x": "${mul:${data.S},${data.P}}"}
    out = resolve(cfg)
    assert out["x"] == 52 * 32


def test_resolve_all_arithmetic():
    cfg = {
        "add": "${add:2,3}",
        "sub": "${sub:7,4}",
        "div": "${div:9,2}",
        "mul": "${mul:6,7}",
    }
    out = resolve(cfg)
    assert out == {"add": 5, "sub": 3, "div": 4.5, "mul": 42}


def test_resolve_env(monkeypatch):
    monkeypatch.setenv("CSI_TEST_VAR", "hello")
    out = resolve({"x": "${env:CSI_TEST_VAR}"})
    assert out["x"] == "hello"


def test_resolve_include(tmp_path):
    inc = _write(tmp_path, "inc.yaml", "k: 7\nl: 8\n")
    out = resolve({"sub": f"${{include:{inc.name}}}"}, base_dir=tmp_path)
    assert out["sub"] == {"k": 7, "l": 8}


def test_resolve_string_interpolation():
    out = resolve({"who": "world", "msg": "hello ${who}"})
    assert out["msg"] == "hello world"


def test_resolve_chain():
    cfg = {"a": 2, "b": "${a}", "c": "${b}"}
    out = resolve(cfg)
    assert out["c"] == 2


def test_resolve_unknown_path_raises():
    with pytest.raises(ResolveError):
        resolve({"x": "${nope.gone}"})


def test_resolve_div_by_zero_raises():
    with pytest.raises(ResolveError):
        resolve({"x": "${div:1,0}"})


def test_load_and_resolve_e2e(tmp_path):
    _write(
        tmp_path,
        "cfg.yaml",
        """
        data: {max_subband: 52, max_port: 32}
        model:
          decoder:
            out_dim: ${mul:${data.max_subband},${data.max_port}}
        training: {epochs: 10}
        """,
    )
    out = load_and_resolve(
        tmp_path / "cfg.yaml",
        overrides=["training.epochs=3", "data.max_port=16"],
    )
    assert out["training"]["epochs"] == 3
    assert out["data"]["max_port"] == 16
    assert out["model"]["decoder"]["out_dim"] == 52 * 16
