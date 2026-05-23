from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from csi_comp.data import NpzDataset, make_collate_fn
from csi_comp.losses.composite import WeightedSumLoss
from csi_comp.training import (
    Trainer,
    TrainerCallback,
    build_model,
    build_optimizer,
    build_scheduler,
    configure_device,
    get_mode_spec,
    seed_everything,
)
MAX_S, MAX_P = 8, 12


def _cfg(mode: str, with_latent_shape: tuple[int, ...] | None = None):
    cfg = {
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
                ],
            },
        },
        "quantizer": {"type": "uniform", "bits": 2, "value_range": [-1.0, 1.0], "grad": "ste"},
        "training": {"mode": mode},
        "loss": {"terms": [{"name": "one_minus_sgcs", "weight": 1.0}]},
    }
    if with_latent_shape is not None:
        cfg["model"]["decoder"]["latent_shape"] = list(with_latent_shape)
    return cfg


def _loaders(npz_root: Path, batch_size: int = 4, latent_key: str = "Zq"):
    train = NpzDataset(npz_root / "train.npz", latent_key=latent_key)
    val = NpzDataset(npz_root / "val.npz", latent_key=latent_key)
    coll = make_collate_fn(MAX_S, MAX_P)
    return (
        DataLoader(train, batch_size=batch_size, shuffle=False, collate_fn=coll),
        DataLoader(val, batch_size=batch_size, shuffle=False, collate_fn=coll),
    )


def test_seed_everything_repeatable():
    seed_everything(123)
    a = torch.randn(5)
    seed_everything(123)
    b = torch.randn(5)
    assert torch.equal(a, b)


def test_configure_device_cpu():
    assert configure_device({"device": "cpu"}).type == "cpu"


def test_build_model_joint(npz_root):
    cfg = _cfg("joint")
    spec = get_mode_spec("joint")
    ae, etr, dtr = build_model(cfg, spec)
    assert ae.encoder is not None
    assert ae.decoder is not None
    assert ae.quantizer is not None
    assert len(etr) >= 1 and len(dtr) >= 1
    assert all(p.requires_grad for p in ae.parameters())


def test_build_model_encoder_only_freezes_nothing_external():
    cfg = _cfg("encoder_only")
    spec = get_mode_spec("encoder_only")
    ae, etr, dtr = build_model(cfg, spec)
    assert ae.encoder is not None
    assert ae.decoder is None
    assert ae.quantizer is not None
    assert dtr == []


def test_build_model_decoder_only_requires_latent_shape():
    cfg = _cfg("decoder_only")
    spec = get_mode_spec("decoder_only")
    with pytest.raises(ValueError):
        build_model(cfg, spec)
    cfg2 = _cfg("decoder_only", with_latent_shape=(16,))
    ae, etr, dtr = build_model(cfg2, spec)
    assert ae.encoder is None
    assert ae.decoder is not None
    assert ae.quantizer is None


def test_trainer_joint_one_epoch_improves_loss(npz_root):
    seed_everything(0)
    cfg = _cfg("joint")
    spec = get_mode_spec("joint")
    ae, _, _ = build_model(cfg, spec)
    train_loader, val_loader = _loaders(npz_root)

    loss_fn = WeightedSumLoss(cfg["loss"]["terms"], mode="joint")
    opt = build_optimizer(ae, {"name": "adam", "lr": 1e-2})

    metrics_log: list[dict] = []

    class RecordCb(TrainerCallback):
        def on_train_step_end(self, trainer, step, metrics):
            metrics_log.append(metrics)

    trainer = Trainer(
        model=ae, optimizer=opt, loss_fn=loss_fn,
        train_loader=train_loader, val_loader=val_loader,
        mode_spec=spec, device=torch.device("cpu"),
        epochs=2, val_every_n_epochs=1,
        callbacks=[RecordCb()],
        best_metric={"name": "sgcs", "mode": "max"},
    )
    trainer.fit()

    assert len(metrics_log) > 0
    first_loss = metrics_log[0]["loss/total"]
    last_loss = metrics_log[-1]["loss/total"]
    assert last_loss <= first_loss + 0.5


def test_trainer_frozen_decoder_does_not_update_decoder(npz_root, tmp_path, make_npz):
    """End-to-end inter-vendor sanity: train decoder_only first, save it,
    then load it as a frozen decoder for encoder_only_frozen_decoder mode and
    verify the decoder's parameters remain untouched."""
    seed_everything(0)
    cfg = _cfg("decoder_only", with_latent_shape=(16,))
    spec = get_mode_spec("decoder_only")
    ae, _, _ = build_model(cfg, spec)
    syn = tmp_path / "with_lat"
    syn.mkdir()
    make_npz(syn / "train.npz", n=4, S=6, P=10, latent_dim=16, seed=42)
    make_npz(syn / "val.npz", n=2, S=6, P=10, latent_dim=16, seed=43)
    train_loader, val_loader = _loaders(syn)

    loss_fn = WeightedSumLoss(cfg["loss"]["terms"], mode="decoder_only")
    opt = build_optimizer(ae, {"name": "adam", "lr": 1e-2})
    trainer = Trainer(
        model=ae, optimizer=opt, loss_fn=loss_fn,
        train_loader=train_loader, val_loader=val_loader,
        mode_spec=spec, device=torch.device("cpu"),
        epochs=1, val_every_n_epochs=1,
        best_metric={"name": "sgcs", "mode": "max"},
    )
    trainer.fit()

    dec_path = tmp_path / "decoder.pt"
    torch.save({"decoder": ae.decoder.state_dict()}, dec_path)
    snapshot = {k: v.detach().clone() for k, v in ae.decoder.state_dict().items()}

    cfg2 = _cfg("encoder_only_frozen_decoder")
    cfg2["model"]["decoder"]["pretrained_path"] = str(dec_path)
    spec2 = get_mode_spec("encoder_only_frozen_decoder")
    ae2, _, _ = build_model(cfg2, spec2)
    assert all(not p.requires_grad for p in ae2.decoder.parameters())

    train_loader2, val_loader2 = _loaders(npz_root)
    loss_fn2 = WeightedSumLoss(cfg2["loss"]["terms"], mode="encoder_only_frozen_decoder")
    opt2 = build_optimizer(ae2, {"name": "adam", "lr": 1e-2})
    trainer2 = Trainer(
        model=ae2, optimizer=opt2, loss_fn=loss_fn2,
        train_loader=train_loader2, val_loader=val_loader2,
        mode_spec=spec2, device=torch.device("cpu"),
        epochs=1, val_every_n_epochs=1,
        best_metric={"name": "sgcs", "mode": "max"},
    )
    trainer2.fit()
    for k, v in ae2.decoder.state_dict().items():
        assert torch.equal(v, snapshot[k]), f"frozen decoder param {k} changed"


def test_build_scheduler_cosine():
    model = torch.nn.Linear(4, 4)
    opt = torch.optim.Adam(model.parameters())
    sch = build_scheduler(opt, {"name": "cosine", "T_max": 5})
    assert sch is not None
    assert build_scheduler(opt, None) is None
