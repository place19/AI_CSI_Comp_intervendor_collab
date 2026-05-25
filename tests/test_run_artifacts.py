import json
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from csi_comp.analysis import build_note, profile_model
from csi_comp.data import NpzDataset, make_collate_fn
from csi_comp.losses.composite import WeightedSumLoss
from csi_comp.training import (
    Trainer,
    TrainerCallback,
    build_model,
    build_optimizer,
    get_mode_spec,
    seed_everything,
)
from csi_comp.training.checkpoint import (
    CheckpointCallback,
    load_checkpoint,
    save_checkpoint,
)
from csi_comp.training.mlflow_logger import MLflowCallback, MLflowLogger

MAX_S, MAX_P = 8, 12


def _cfg(mode: str = "joint"):
    return {
        "experiment": {"device": "cpu", "seed": 42},
        "data": {"layout": "cnn", "max_subband": MAX_S, "max_port": MAX_P},
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
                    {"name": "reshape_head", "max_subband": MAX_S, "max_port": MAX_P},
                ]
            },
        },
        "quantizer": {"type": "uniform", "bits": 2, "value_range": [-1.0, 1.0], "grad": "ste"},
        "training": {"mode": mode},
        "loss": {"terms": [{"name": "one_minus_sgcs", "weight": 1.0}]},
    }


def _loaders(syn):
    coll = make_collate_fn(MAX_S, MAX_P)
    return (
        DataLoader(NpzDataset(syn / "train.npz"), batch_size=4, collate_fn=coll),
        DataLoader(NpzDataset(syn / "val.npz"), batch_size=4, collate_fn=coll),
    )


def test_save_and_load_checkpoint(tmp_path):
    cfg = _cfg()
    spec = get_mode_spec("joint")
    ae, _, _ = build_model(cfg, spec)
    opt = build_optimizer(ae, {"name": "adam", "lr": 1e-3})

    # Twiddle one param so the save→load round-trip is detectable
    with torch.no_grad():
        for p in ae.parameters():
            p.add_(0.1)

    snapshot = {k: v.detach().clone() for k, v in ae.state_dict().items()}
    save_checkpoint(tmp_path / "ckpt.pt", ae, opt, None, epoch=3, global_step=42,
                    best_value=0.123, config=cfg)

    # Reset model and reload
    ae2, _, _ = build_model(cfg, spec)
    opt2 = build_optimizer(ae2, {"name": "adam", "lr": 1e-3})
    restored = load_checkpoint(tmp_path / "ckpt.pt", ae2, opt2, None)
    assert restored.epoch == 3
    assert restored.global_step == 42
    assert restored.best_value == pytest.approx(0.123)
    for k, v in ae2.state_dict().items():
        assert torch.equal(v, snapshot[k])


def test_checkpoint_callback_writes_files(npz_root, tmp_path):
    seed_everything(0)
    cfg = _cfg("joint")
    spec = get_mode_spec("joint")
    ae, _, _ = build_model(cfg, spec)
    train_loader, val_loader = _loaders(npz_root)
    loss_fn = WeightedSumLoss(cfg["loss"]["terms"], mode="joint")
    opt = build_optimizer(ae, {"name": "adam", "lr": 1e-2})

    out_dir = tmp_path / "outputs"
    cb = CheckpointCallback(out_dir=out_dir, config=cfg)
    trainer = Trainer(
        model=ae, optimizer=opt, loss_fn=loss_fn,
        train_loader=train_loader, val_loader=val_loader,
        mode_spec=spec, device=torch.device("cpu"),
        epochs=2, val_every_n_epochs=1, callbacks=[cb],
        best_metric={"name": "sgcs", "mode": "max"},
    )
    trainer.fit()

    assert (out_dir / "latest.pt").exists()
    assert (out_dir / "best.pt").exists()


def test_build_note_contains_table_and_config():
    cfg = _cfg("joint")
    spec = get_mode_spec("joint")
    ae, etr, dtr = build_model(cfg, spec)
    prof = profile_model(ae, etr, dtr)
    note = build_note(cfg, prof)
    assert "Encoder" in note
    assert "Decoder" in note
    assert "in_shape" in note
    assert "TOTAL" in note
    assert "Summary" in note       # top-level params/FLOPs summary table
    assert "max_subband" in note   # cfg made it into the note
    # Profile section must appear before Configuration section.
    assert note.index("Model profile") < note.index("Configuration")


def test_mlflow_logger_local_file_uri(tmp_path, make_npz):
    """Smoke: run a tiny fit with MLflowCallback against a file:// store."""
    seed_everything(0)
    syn = tmp_path / "syn"
    syn.mkdir()
    make_npz(syn / "train.npz", n=4, S=6, P=10, latent_dim=16, seed=0)
    make_npz(syn / "val.npz", n=2, S=6, P=10, latent_dim=16, seed=1)

    cfg = _cfg("joint")
    spec = get_mode_spec("joint")
    ae, etr, dtr = build_model(cfg, spec)
    train_loader, val_loader = _loaders(syn)
    loss_fn = WeightedSumLoss(cfg["loss"]["terms"], mode="joint")
    opt = build_optimizer(ae, {"name": "adam", "lr": 1e-2})

    mlruns = tmp_path / "mlruns"
    logger = MLflowLogger(
        tracking_uri=f"file:{mlruns}",
        experiment_name="test_exp",
        log_every_n_iters=1,
        run_name="test_run",
    )
    with logger:
        prof = profile_model(ae, etr, dtr)
        logger.set_note(build_note(cfg, prof))
        # Write the resolved config and log it as an artifact
        cfg_path = tmp_path / "cfg.yaml"
        import yaml
        cfg_path.write_text(yaml.safe_dump(cfg))
        logger.log_artifact(cfg_path)

        trainer = Trainer(
            model=ae, optimizer=opt, loss_fn=loss_fn,
            train_loader=train_loader, val_loader=val_loader,
            mode_spec=spec, device=torch.device("cpu"),
            epochs=1, val_every_n_epochs=1,
            callbacks=[MLflowCallback(logger)],
            best_metric={"name": "sgcs", "mode": "max"},
        )
        trainer.fit()

    # MLflow should have created a run directory under the experiment
    exp_dirs = list(mlruns.glob("*"))
    assert exp_dirs, "no experiment directory created"
