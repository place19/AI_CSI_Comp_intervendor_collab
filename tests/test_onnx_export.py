from pathlib import Path

import pytest

from csi_comp.export import VALID_SCOPES, export_to_onnx, verify_onnx_parity
from csi_comp.training import build_model, get_mode_spec

MAX_S, MAX_P = 6, 8


def _cfg():
    return {
        "data": {"layout": "cnn", "max_subband": MAX_S, "max_port": MAX_P},
        "model": {
            "encoder": {
                "blocks": [
                    {"name": "cnn_block", "channels": 4, "kernel": 3, "norm": "none"},
                    {"name": "linear_proj", "out_dim": 16, "activation": "relu"},
                    {"name": "activation", "activation": "tanh"},
                ],
            },
            "decoder": {
                "blocks": [
                    {"name": "linear_proj", "out_dim": 32, "activation": "relu"},
                    {"name": "reshape_head", "max_subband": MAX_S, "max_port": MAX_P},
                ]
            },
        },
        "quantizer": {"type": "uniform", "bits": 2, "value_range": [-1.0, 1.0], "grad": "ste"},
        "training": {"mode": "joint"},
        "loss": {"terms": [{"name": "one_minus_sgcs", "weight": 1.0}]},
    }


@pytest.mark.parametrize("scope", list(VALID_SCOPES))
def test_export_each_scope_and_verify(scope, tmp_path):
    cfg = _cfg()
    spec = get_mode_spec("joint")
    ae, _, _ = build_model(cfg, spec)

    out = tmp_path / f"{scope}.onnx"
    export_to_onnx(ae, cfg, scope=scope, out_path=out)
    assert out.exists()
    # parity (allow modest tolerance — BN with default running stats, float32 ops)
    diff = verify_onnx_parity(out, ae, cfg, scope=scope, atol=1e-3)
    assert diff < 1e-3


def test_export_unknown_scope_raises(tmp_path):
    cfg = _cfg()
    spec = get_mode_spec("joint")
    ae, _, _ = build_model(cfg, spec)
    with pytest.raises(ValueError):
        export_to_onnx(ae, cfg, scope="weird", out_path=tmp_path / "x.onnx")
