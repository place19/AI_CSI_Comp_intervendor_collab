"""Mode-agnostic training loop with callback hooks."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..losses.composite import WeightedSumLoss
from ..models import Autoencoder
from ..models.latent_mask import (
    LatentMaskSpec,
    apply_latent_mask,
    apply_random_latent_mask,
)
from .amp import AmpSpec, autocast_ctx, build_grad_scaler
from .modes import ModeSpec


# ----- batch_to_io: how a raw batch becomes (pred_pack, target_pack) -----

def _batch_to_io(
    ae: Autoencoder,
    batch: Dict[str, torch.Tensor],
    mode_spec: ModeSpec,
    device: torch.device,
    mask_spec: Optional[LatentMaskSpec] = None,
    use_cuda_graphs: bool = False,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    real = batch["real"].to(device)
    imag = batch["imag"].to(device)
    mask = batch["mask"].to(device)
    # Reconstruction target may differ from encoder input by a bin-midpoint
    # dequantization offset (3GPP/HW int8 floor convention) and/or because the
    # encoder input is an augmented (UE-condition) dataset paired against the
    # clean target (PairedInputDataset; see data/paired.py). Fall back to the
    # raw input when the dataset doesn't provide a separate target.
    if "real_target" in batch and "imag_target" in batch:
        real_t = batch["real_target"].to(device)
        imag_t = batch["imag_target"].to(device)
    else:
        real_t, imag_t = real, imag
    precoder = torch.stack([real_t, imag_t], dim=-1)  # (B, S, P, 2)

    out: Dict[str, Any] = {}
    if mode_spec.needs_encoder:
        # Masking applies only when both encoder and decoder are present.
        if mask_spec is not None and mode_spec.needs_decoder:
            latent = ae.encoder(real, imag)
            # Clone before decoder: encoder/decoder CUDA Graph pools may overlap.
            if use_cuda_graphs:
                latent = latent.clone()
            if ae.quantizer is not None:
                rescaled = ae.quantizer.rescale_to_value_range(latent)
                q_latent = ae.quantizer(latent)
            else:
                rescaled = latent
                q_latent = latent
            if mask_spec.mode == "half":
                masked = apply_latent_mask(q_latent, mask_spec.mask_ratio)
                recon = ae.decoder(masked) if ae.decoder is not None else None
                out = {"latent": latent, "rescaled_latent": rescaled,
                       "quantized_latent": q_latent, "recon": recon}
            elif mask_spec.mode == "random":
                masked = apply_random_latent_mask(q_latent, mask_spec.mask_ratio)
                recon = ae.decoder(masked) if ae.decoder is not None else None
                out = {"latent": latent, "rescaled_latent": rescaled,
                       "quantized_latent": q_latent, "recon": recon}
            elif mask_spec.mode == "dual":
                recon_full = ae.decoder(q_latent) if ae.decoder is not None else None
                # Clone before second decoder run overwrites the first output buffer.
                if use_cuda_graphs and recon_full is not None:
                    recon_full = recon_full.clone()
                masked = apply_latent_mask(q_latent, mask_spec.mask_ratio)
                recon_half = ae.decoder(masked) if ae.decoder is not None else None
                out = {
                    "latent": latent,
                    "rescaled_latent": rescaled,
                    "quantized_latent": q_latent,
                    "recon": recon_full,
                    "recon_half": recon_half,
                }
            else:
                out = ae(real, imag)
        else:
            out = ae(real, imag)
    elif mode_spec.needs_decoder:
        # decoder_only: provided latent goes through decoder directly. The decoder
        # consumes the post-quant latent (Zq) at deployment; accept the legacy
        # `latent_target` (npz latent_key) or an exposed `latent_target_zq` / `_z`.
        dec_in_key = next(
            (k for k in ("latent_target", "latent_target_zq", "latent_target_z") if k in batch),
            None,
        )
        if dec_in_key is None:
            raise KeyError(
                "decoder_only mode needs a latent input: set data.dataset_args.expose_zq "
                "(lmdb/npz) or latent_key (npz) so the batch carries a latent target."
            )
        latent = batch[dec_in_key].to(device)
        recon = ae.decoder(latent) if ae.decoder is not None else None
        out = {"latent": latent, "quantized_latent": latent, "recon": recon}

    # mask never flows through the model anymore; it's only consumed by the
    # loss (and only by terms that need it, like one_minus_sgcs).
    target: Dict[str, Any] = {"precoder": precoder, "mask": mask}
    # Forward every latent target the dataset emitted; loss terms pick one via
    # `target_key` (e.g. mse_latent ↔ latent_target_z, mse_quantized_latent ↔
    # latent_target_zq). `latent_target` also doubles as the decoder_only input above.
    for k in ("latent_target", "latent_target_z", "latent_target_zq"):
        if k in batch:
            target[k] = batch[k].to(device)
    return out, target


def _to_fp32_pack(pack: Dict[str, Any]) -> Dict[str, Any]:
    """Cast floating-point tensors in a pred/target pack to fp32 (no copy if
    already fp32). Used at the autocast boundary before loss / metrics so
    those computations always run in fp32 regardless of AMP dtype."""
    out: Dict[str, Any] = {}
    for k, v in pack.items():
        if isinstance(v, torch.Tensor) and v.is_floating_point():
            out[k] = v.float()
        else:
            out[k] = v
    return out


# ----- Callback contract -----

class TrainerCallback:
    """Subclass and override the hooks you care about. All are no-ops by default."""

    def on_train_begin(self, trainer: "Trainer") -> None: ...
    def on_train_end(self, trainer: "Trainer") -> None: ...
    def on_epoch_begin(self, trainer: "Trainer", epoch: int) -> None: ...
    def on_epoch_end(
        self, trainer: "Trainer", epoch: int, train_metrics: Dict[str, float]
    ) -> None: ...
    def on_train_step_end(
        self, trainer: "Trainer", step: int, metrics: Dict[str, float]
    ) -> None: ...
    def on_val_end(
        self, trainer: "Trainer", epoch: int, val_metrics: Dict[str, float]
    ) -> None: ...

    def on_epoch_complete(
        self, trainer: "Trainer", epoch: int, train_metrics: Dict[str, float]
    ) -> None: ...


@dataclass
class Trainer:
    model: Autoencoder
    optimizer: torch.optim.Optimizer
    loss_fn: WeightedSumLoss
    train_loader: DataLoader
    val_loader: Optional[DataLoader]
    mode_spec: ModeSpec
    device: torch.device
    epochs: int
    val_every_n_epochs: int = 1
    scheduler: Optional[Any] = None
    callbacks: List[TrainerCallback] = field(default_factory=list)
    best_metric: dict[str, Any] = field(default_factory=lambda: {"name": "sgcs", "mode": "max"})
    amp_spec: Optional[AmpSpec] = None
    mask_spec: Optional[LatentMaskSpec] = None
    use_cuda_graphs: bool = False

    # Runtime state
    epoch: int = 0
    global_step: int = 0
    best_value: float = float("nan")
    # The current epoch's validation value for `best_metric` (NaN when validation
    # did not run this epoch). Distinct from `best_value` (best-so-far); used to
    # name the latest checkpoint after the epoch it actually describes.
    last_val_value: float = float("nan")

    def __post_init__(self):
        self.model.to(self.device)
        self.loss_fn.to(self.device)
        # initialize best_value to ±inf depending on direction
        if self.best_metric.get("mode", "max") == "max":
            self.best_value = float("-inf")
        else:
            self.best_value = float("inf")
        self.scaler = build_grad_scaler(self.amp_spec)

    # ----- public API -----

    def fit(self) -> None:
        self._dispatch("on_train_begin")
        for epoch in range(self.epoch, self.epochs):
            self.epoch = epoch
            self.last_val_value = float("nan")  # reset; set below iff validation runs
            self._dispatch("on_epoch_begin", epoch)
            train_metrics = self._train_one_epoch()
            self._dispatch("on_epoch_end", epoch, train_metrics)
            if self.val_loader is not None and (epoch + 1) % self.val_every_n_epochs == 0:
                val_metrics = self.validate()
                self.last_val_value = val_metrics.get(
                    f"val/{self.best_metric['name']}", float("nan")
                )
                self._update_best(val_metrics)
                self._dispatch("on_val_end", epoch, val_metrics)
            # Epoch-unit schedulers step here. Iter-unit ones already stepped inside the loop.
            if self.scheduler is not None and getattr(self.scheduler, "step_unit", "epoch") == "epoch":
                self.scheduler.step()
            # on_epoch_complete fires after validation, best update, and scheduler step so
            # that latest.pt captures the fully-updated state for clean resume.
            self._dispatch("on_epoch_complete", epoch, train_metrics)
        self._dispatch("on_train_end")

    # ----- internals -----

    def _train_one_epoch(self) -> Dict[str, float]:
        self.model.train()
        # Frozen submodules stay in eval()
        for name in self.mode_spec.frozen_inference:
            getattr(self.model, name).eval()

        totals: Dict[str, float] = {}
        n_samples: int = 0
        for batch in self.train_loader:
            # Model forward under autocast (when AMP enabled).
            with autocast_ctx(self.amp_spec):
                pred_pack, target_pack = _batch_to_io(
                    self.model, batch, self.mode_spec, self.device, self.mask_spec,
                    self.use_cuda_graphs,
                )
            # Loss runs OUTSIDE autocast — fp32 island. Cast any fp tensors back
            # to fp32 first so the loss never sees half/bf16 inputs.
            pred_pack = _to_fp32_pack(pred_pack)
            target_pack = _to_fp32_pack(target_pack)

            self.optimizer.zero_grad(set_to_none=True)
            total, per_term = self.loss_fn(pred_pack, target_pack)
            if self.scaler is not None:
                self.scaler.scale(total).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                total.backward()
                self.optimizer.step()
            # Iter-unit schedulers (warmup_cosine, etc.) step after each optimizer step.
            if self.scheduler is not None and getattr(self.scheduler, "step_unit", "epoch") == "iter":
                self.scheduler.step()
            self.global_step += 1
            step_metrics = self._collect_step_metrics(total, per_term, pred_pack, target_pack)
            self._dispatch("on_train_step_end", self.global_step, step_metrics)
            # Sample-count weighting: a small final batch (drop_last=false) must not
            # carry the same weight as a full one. An unweighted batch-mean-of-means
            # otherwise drifts from the true dataset mean. `lr` is an optimizer-step
            # metric (not a sample metric), so skip it here and report the final LR.
            bs = int(batch["real"].shape[0])
            for k, v in step_metrics.items():
                if k == "lr" or k.startswith("lr/"):
                    continue
                totals[k] = totals.get(k, 0.0) + v * bs
            n_samples += bs
        result = {k: v / max(n_samples, 1) for k, v in totals.items()}
        result["lr"] = float(self.optimizer.param_groups[0]["lr"])
        return result

    @torch.no_grad()
    def validate(self, prefix: str = "val") -> Dict[str, float]:
        self.model.eval()
        totals: Dict[str, float] = {}
        n_samples = 0
        for batch in self.val_loader:  # type: ignore[union-attr]
            with autocast_ctx(self.amp_spec):
                pred_pack, target_pack = _batch_to_io(
                    self.model, batch, self.mode_spec, self.device, self.mask_spec,
                    self.use_cuda_graphs,
                )
            # Clone to detach from CUDAGraph output buffers (reduce-overhead / max-autotune reuse them).
            if self.use_cuda_graphs:
                pred_pack = {k: v.clone() if isinstance(v, torch.Tensor) else v
                             for k, v in pred_pack.items()}
            # Loss + metrics in fp32, matching the training path.
            pred_pack = _to_fp32_pack(pred_pack)
            target_pack = _to_fp32_pack(target_pack)
            total, per_term = self.loss_fn(pred_pack, target_pack)
            m = self._collect_step_metrics(total, per_term, pred_pack, target_pack)
            # `lr` reflects the optimizer state, not the validation batch — drop
            # it so we don't log a meaningless prefixed lr metric.
            bs = int(batch["real"].shape[0])
            for k, v in m.items():
                if k == "lr" or k.startswith("lr/"):
                    continue
                totals[k] = totals.get(k, 0.0) + v * bs
            n_samples += bs
        return {f"{prefix}/{k}": v / max(n_samples, 1) for k, v in totals.items()}

    def _collect_step_metrics(
        self,
        total: torch.Tensor,
        per_term: Dict[str, torch.Tensor],
        pred_pack: Dict[str, Any],
        target_pack: Dict[str, Any],
    ) -> Dict[str, float]:
        m = {"loss/total": float(total.detach().cpu().item())}
        for k, v in per_term.items():
            m[f"loss/{k}"] = float(v.detach().cpu().item())
        # Surface the optimizer's current LR. The decay/no_decay split (see
        # builders._split_decay_no_decay) leaves both groups on the same LR
        # schedule, so a single curve is enough; if a future config wants per-group
        # LRs this is the place to switch back to logging every group.
        m["lr"] = float(self.optimizer.param_groups[0]["lr"])
        # Always log SGCS when the decoder is present, even if it's not the training loss.
        if pred_pack.get("recon") is not None:
            if "one_minus_sgcs" in per_term:
                # Loss term already computed the masked mean SGCS — reuse it.
                m["sgcs"] = 1.0 - m["loss/one_minus_sgcs"]
            else:
                from ..losses.sgcs import sgcs_per_subband
                recon = pred_pack["recon"]
                target = target_pack["precoder"]
                mask = target_pack.get("mask")
                if mask is not None:
                    mask4d = mask.unsqueeze(-1)
                    sgcs = sgcs_per_subband(target * mask4d, recon * mask4d)
                    sb_valid = mask.any(dim=-1).to(sgcs.dtype)
                    mean_sgcs = (sgcs * sb_valid).sum() / (sb_valid.sum() + 1e-12)
                else:
                    sgcs = sgcs_per_subband(target, recon)
                    mean_sgcs = sgcs.mean()
                m["sgcs"] = float(mean_sgcs.detach().cpu().item())
        return m

    def _update_best(self, val_metrics: Dict[str, float]) -> None:
        import warnings
        key = f"val/{self.best_metric['name']}"
        if key not in val_metrics:
            warnings.warn(
                f"best_metric key {key!r} not found in val_metrics "
                f"(available: {sorted(val_metrics)}). "
                f"best.pt will never be saved. "
                f"Set training.best_metric to a key that appears in your loss terms.",
                UserWarning,
                stacklevel=2,
            )
            return
        v = val_metrics[key]
        mode = self.best_metric.get("mode", "max")
        improved = (v > self.best_value) if mode == "max" else (v < self.best_value)
        if improved:
            self.best_value = v
            val_metrics["best/" + self.best_metric["name"]] = v

    def _dispatch(self, method: str, *args, **kwargs) -> None:
        for cb in self.callbacks:
            fn = getattr(cb, method, None)
            if fn is not None:
                fn(self, *args, **kwargs)
