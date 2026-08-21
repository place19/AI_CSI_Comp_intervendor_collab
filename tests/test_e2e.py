"""End-to-end smoke tests that exercise the actual CLI scripts via subprocess.

These are slower (a few seconds each) — they ensure the train/test/pretrained-checkpoint/export
pipelines hang together correctly on CPU with synthetic data.
"""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
PY = sys.executable


def _run(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd, cwd=str(cwd), check=True, capture_output=True, text=True,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )


@pytest.fixture(scope="module")
def workdir(tmp_path_factory):
    """A temp working dir with npz data + a tiny config pointing at it."""
    import numpy as np

    d = tmp_path_factory.mktemp("e2e")
    # 1) Fabricate a small npz dataset directly (no synthetic generator script).
    data_dir = d / "data" / "npz"
    data_dir.mkdir(parents=True)

    def _write(path, n):
        rng = np.random.default_rng(int(path.stat().st_size if path.exists() else n))
        D = rng.integers(-128, 128, size=(n, 6, 10, 2), dtype=np.int8)
        Z = rng.standard_normal((n, 16)).astype(np.float32)
        Zq = rng.standard_normal((n, 16)).astype(np.float32)
        np.savez(path, D=D, Z=Z, Zq=Zq)

    _write(data_dir / "train.npz", 8)
    _write(data_dir / "val.npz", 4)

    # 2) Write a tiny config pinned to that data
    cfg = {
        "experiment": {
            "name": "e2e_run",
            "seed": 0,
            "device": "cpu",
            "mlflow": {
                "tracking_uri": f"file:{d / 'mlruns'}",
                "experiment_name": "e2e",
                "log_every_n_iters": 1,
            },
        },
        "data": {
            "format": "npz",
            "train_path": str(data_dir / "train.npz"),
            "val_path": str(data_dir / "val.npz"),
            "max_subband": 6,
            "max_port": 10,
            "layout": "cnn",
            "batch_size": 4,
            "num_workers": 0,
        },
        "model": {
            "encoder": {
                "blocks": [
                    {"name": "cnn_block", "channels": 4, "kernel": 3},
                    {"name": "linear_proj", "out_dim": 16, "activation": "relu"},
                    {"name": "activation", "activation": "tanh"},
                ],
            },
            "decoder": {
                "blocks": [
                    {"name": "linear_proj", "out_dim": 32, "activation": "relu"},
                    {"name": "reshape_head", "max_subband": 6, "max_port": 10},
                ],
                "pretrained_path": None,
            },
        },
        "quantizer": {
            "type": "uniform", "bits": 2, "value_range": [-1.0, 1.0],
            "unit_spaced": True, "grad": "ste",
        },
        "training": {
            "mode": "joint", "epochs": 1,
            "optimizer": {"name": "adam", "lr": 1.0e-2},
            "val_every_n_epochs": 1,
            "best_metric": {"name": "sgcs", "mode": "max"},
        },
        "loss": {"terms": [{"name": "one_minus_sgcs", "weight": 1.0}]},
    }
    cfg_path = d / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return d


def test_train_writes_checkpoints_and_mlflow(workdir):
    # --no-timestamp keeps the legacy outputs/<name>/ path stable so the rest
    # of the e2e cases can refer to it by name. Default behaviour appends a
    # _YYYYMMDD_HHMMSS suffix.
    _run([PY, str(REPO / "scripts" / "train.py"),
          "--config", str(workdir / "cfg.yaml"),
          "--out-root", str(workdir / "outputs"),
          "--no-timestamp"], cwd=workdir)
    out = workdir / "outputs" / "e2e_run"
    assert (out / "latest.pt").exists()
    assert (out / "best.pt").exists()
    assert (out / "config.resolved.yaml").exists()
    # MLflow files
    assert (workdir / "mlruns").exists()


def test_test_runs_after_train(workdir):
    result = _run([PY, str(REPO / "scripts" / "test.py"),
                   "--checkpoint", str(workdir / "outputs" / "e2e_run" / "best.pt")],
                  cwd=workdir)
    assert "val/sgcs" in result.stdout or "val/loss/total" in result.stdout


def test_cross_checkpoint_test(workdir):
    """Cross-checkpoint mode: load encoder and decoder from separate checkpoints.

    Uses the same joint checkpoint for both sides so no extra training is
    needed; the goal is to exercise the config-merge, quantizer-compat check,
    and component-selective load_checkpoint code paths end-to-end.
    """
    best = workdir / "outputs" / "e2e_run" / "best.pt"
    result = _run([PY, str(REPO / "scripts" / "test.py"),
                   "--encoder-checkpoint", str(best),
                   "--decoder-checkpoint", str(best)],
                  cwd=workdir)
    assert "val/sgcs" in result.stdout


def test_infer_default_save_skips_original(workdir):
    """Default --save excludes 'original'; all other items get written."""
    import json
    import numpy as np

    out_dir = workdir / "infer_default"
    _run([PY, str(REPO / "scripts" / "infer.py"),
          "--checkpoint", str(workdir / "outputs" / "e2e_run" / "best.pt"),
          "--out", str(out_dir)], cwd=workdir)
    # default save list
    for f in ("recon.npy", "latent.npy", "quant_latent.npy", "mask.npy", "sgcs_per_sample.npy"):
        assert (out_dir / f).exists(), f"missing {f}"
    assert not (out_dir / "original.npy").exists()
    meta = json.loads((out_dir / "meta.json").read_text())
    assert "original" not in meta["saved"]
    assert meta["n_samples"] == 4
    # Load directly — no NpzFile indirection.
    recon = np.load(out_dir / "recon.npy")
    assert recon.shape == (4, 6, 10, 2)
    sgcs = np.load(out_dir / "sgcs_per_sample.npy")
    assert sgcs.shape == (4,)


def test_infer_save_all_includes_original(workdir):
    """`--save all` adds original.npy to the dumped set."""
    out_dir = workdir / "infer_all"
    _run([PY, str(REPO / "scripts" / "infer.py"),
          "--checkpoint", str(workdir / "outputs" / "e2e_run" / "best.pt"),
          "--out", str(out_dir),
          "--save", "all"], cwd=workdir)
    assert (out_dir / "original.npy").exists()


def test_infer_explicit_subset(workdir):
    """`--save recon,sgcs_per_sample` writes exactly that subset."""
    out_dir = workdir / "infer_subset"
    _run([PY, str(REPO / "scripts" / "infer.py"),
          "--checkpoint", str(workdir / "outputs" / "e2e_run" / "best.pt"),
          "--out", str(out_dir),
          "--save", "recon,sgcs_per_sample"], cwd=workdir)
    assert (out_dir / "recon.npy").exists()
    assert (out_dir / "sgcs_per_sample.npy").exists()
    for missing in ("latent.npy", "quant_latent.npy", "mask.npy", "original.npy"):
        assert not (out_dir / missing).exists()


def test_infer_limit_caps_samples(workdir):
    """--limit stops after N samples even if more are available."""
    import numpy as np
    out_dir = workdir / "infer_limit"
    _run([PY, str(REPO / "scripts" / "infer.py"),
          "--checkpoint", str(workdir / "outputs" / "e2e_run" / "best.pt"),
          "--out", str(out_dir),
          "--save", "recon",
          "--limit", "2"], cwd=workdir)
    recon = np.load(out_dir / "recon.npy")
    assert recon.shape == (2, 6, 10, 2)


def test_infer_cross_checkpoint(workdir):
    """Cross-checkpoint infer: encoder and decoder from separate checkpoints.

    Uses the same joint checkpoint for both sides so no extra training is
    needed; exercises config-merge, quantizer-compat check, component-selective
    load_checkpoint, and output file writing end-to-end.
    """
    import json
    import numpy as np

    best = workdir / "outputs" / "e2e_run" / "best.pt"
    out_dir = workdir / "infer_cross"
    _run([PY, str(REPO / "scripts" / "infer.py"),
          "--encoder-checkpoint", str(best),
          "--decoder-checkpoint", str(best),
          "--out", str(out_dir)], cwd=workdir)

    for f in ("recon.npy", "latent.npy", "quant_latent.npy", "mask.npy", "sgcs_per_sample.npy"):
        assert (out_dir / f).exists(), f"missing {f}"
    assert not (out_dir / "original.npy").exists()

    meta = json.loads((out_dir / "meta.json").read_text())
    assert "encoder_checkpoint" in meta
    assert "decoder_checkpoint" in meta
    assert "checkpoint" not in meta
    assert meta["n_samples"] == 4

    recon = np.load(out_dir / "recon.npy")
    assert recon.shape == (4, 6, 10, 2)


def test_pretrained_checkpoint_continues_training(workdir):
    _run([PY, str(REPO / "scripts" / "train.py"),
          "--config", str(workdir / "outputs" / "e2e_run" / "config.resolved.yaml"),
          "--pretrained-checkpoint", str(workdir / "outputs" / "e2e_run" / "latest.pt"),
          "--set", "training.epochs=2",
          "--out-root", str(workdir / "outputs_pretrained"),
          "--no-timestamp"], cwd=workdir)
    out = workdir / "outputs_pretrained" / "e2e_run"
    assert (out / "latest.pt").exists()


def test_qat_finetune_produces_a_plain_float_checkpoint(workdir):
    """float run -> QAT fine-tune -> the result is consumable with zero QAT awareness.

    This is the whole point of the QAT design: `test.py` and `export_onnx.py` build a
    plain float model and load the QAT run's checkpoint strictly, because
    `float_state_dict` undid the fusion and dropped the observers on save.
    """
    import torch

    base = yaml.safe_load((workdir / "outputs" / "e2e_run" / "config.resolved.yaml").read_text())
    base["experiment"]["name"] = "e2e_qat"
    base["training"]["epochs"] = 2
    base["training"]["qat"] = {
        "enabled": True,
        "fold_bn": True,
        "quantize_input": True,
        "quantize_activations": True,
        "weight": {"bits": 8, "dtype": "qint8", "qscheme": "per_channel_symmetric"},
        "activation": {"bits": 8, "dtype": "quint8", "qscheme": "per_tensor_affine"},
        "freeze_observer_epoch": 1,
        "freeze_bn_epoch": 1,
    }
    qat_cfg = workdir / "qat_cfg.yaml"
    qat_cfg.write_text(yaml.safe_dump(base, sort_keys=False))

    _run([PY, str(REPO / "scripts" / "train.py"),
          "--config", str(qat_cfg),
          "--pretrained-checkpoint", str(workdir / "outputs" / "e2e_run" / "best.pt"),
          "--out-root", str(workdir / "outputs_qat"),
          "--no-timestamp"], cwd=workdir)
    out = workdir / "outputs_qat" / "e2e_qat"
    assert (out / "best.pt").exists()

    # The saved encoder must be byte-for-byte shaped like the float run's, and must
    # carry no observer/fake-quant keys.
    float_sd = torch.load(workdir / "outputs" / "e2e_run" / "best.pt",
                          map_location="cpu", weights_only=False)
    qat_sd = torch.load(out / "best.pt", map_location="cpu", weights_only=False)
    assert list(qat_sd["encoder"]) == list(float_sd["encoder"])
    assert not any(
        p in k.split(".")
        for k in qat_sd["encoder"]
        for p in ("weight_fake_quant", "activation_post_process")
    )
    # Observer state rides along under its own key, outside the model entries.
    assert qat_sd["qat_observers"]

    # Downstream scripts consume it unchanged.
    result = _run([PY, str(REPO / "scripts" / "test.py"),
                   "--checkpoint", str(out / "best.pt")], cwd=workdir)
    assert "val/sgcs" in result.stdout
    _run([PY, str(REPO / "scripts" / "export_onnx.py"),
          "--checkpoint", str(out / "best.pt"),
          "--scope", "encoder,full",
          "--out", str(out / "onnx")], cwd=workdir)
    assert (out / "onnx" / "encoder.onnx").exists()
    assert (out / "onnx" / "full.onnx").exists()


def test_export_onnx_all_scopes(workdir):
    result = _run([PY, str(REPO / "scripts" / "export_onnx.py"),
                   "--checkpoint", str(workdir / "outputs" / "e2e_run" / "best.pt"),
                   "--scope", "encoder,encoder_quant,decoder,full",
                   "--out", str(workdir / "outputs" / "e2e_run" / "onnx")],
                  cwd=workdir)
    onnx_dir = workdir / "outputs" / "e2e_run" / "onnx"
    for s in ("encoder", "encoder_quant", "decoder", "full"):
        assert (onnx_dir / f"{s}.onnx").exists(), f"missing {s}.onnx"
    assert "parity diff" in result.stdout


def test_inter_vendor_frozen_decoder_keeps_decoder_intact(workdir, tmp_path):
    """After training joint, build a frozen-decoder run pointing at the saved best.pt
    and confirm the decoder weights are unchanged after a training epoch."""
    import torch

    best_path = workdir / "outputs" / "e2e_run" / "best.pt"
    snap = torch.load(best_path, map_location="cpu", weights_only=False)
    dec_snapshot = {k: v.clone() for k, v in snap["decoder"].items()}

    # Build a separate frozen-decoder config that references the same decoder.
    cfg2_path = tmp_path / "frozen.yaml"
    cfg = yaml.safe_load((workdir / "cfg.yaml").read_text())
    cfg["experiment"]["name"] = "e2e_frozen"
    cfg["training"]["mode"] = "encoder_only_frozen_decoder"
    cfg["model"]["decoder"]["pretrained_path"] = str(best_path)
    cfg["training"]["epochs"] = 1
    cfg2_path.write_text(yaml.safe_dump(cfg))

    _run([PY, str(REPO / "scripts" / "train.py"),
          "--config", str(cfg2_path),
          "--out-root", str(workdir / "outputs"),
          "--no-timestamp"], cwd=workdir)

    new_ckpt = torch.load(workdir / "outputs" / "e2e_frozen" / "latest.pt",
                          map_location="cpu", weights_only=False)
    for k, v in new_ckpt["decoder"].items():
        assert torch.equal(v, dec_snapshot[k]), f"frozen decoder param {k} changed!"
