# AI CSI Compression — Inter-Vendor Framework

Experimental PyTorch framework for **AI-based CSI compression** with support for **inter-vendor collaboration** (encoder and decoder trained independently or jointly, potentially by different vendors).

Built around a config-driven, registry-based block system so new encoders/decoders, losses, quantizers, and LR schedulers drop in without touching the training loop.

---

## Highlights

- **Inter-vendor training modes**: `joint`, `encoder_only`, `decoder_only`, `encoder_only_frozen_decoder`.
- **Registry-based blocks**: CNN, depthwise-separable conv, residual, average pooling, transformer (custom Q/K/V/O with selectable pre/post-LN), positional encoding (fixed sin/cos · learnable random · learnable sin/cos), reshape, linear projection, standalone activation, bounding head, reshape head, complex FFN head (per-branch real/imag projections). Add a new block with a single `@register("block", "name")` decorator.
- **Fixed-shape forward contract**: every block declares `out_shape` in `__init__` and has a single `forward(x) -> x` signature; the encoder/decoder builder propagates shapes statically. Padding masks live only on the data batch and are consumed by the loss — not threaded through the model.
- **Configurable quantizer**: uniform quantization with pluggable gradient strategies (STE / soft / hard). Bit-width, range, and level spacing all configurable.
- **Composite loss**: weighted sum of named loss terms (`one_minus_sgcs`, `mse_latent`, `mse_quantized_latent`, `dual_one_minus_sgcs`, …). Add new terms via the loss registry.
- **Latent masking** (`model.latent_mask`): zero out the trailing fraction of the quantized latent before the decoder to simulate partial-bandwidth scenarios. Four modes: `full` (disabled), `half` (fixed), `random` (per-sample augmentation), `dual` (decoder called twice; loss is a weighted sum of full- and half-latent reconstructions). Applies only when both encoder and decoder are present (`joint`, `encoder_only_frozen_decoder`); automatically skipped in `encoder_only` and `decoder_only` modes.
- **Schedulers**: PyTorch built-ins (`cosine`, `step`, `none`) plus a custom `warmup_cosine` (linear warmup → cosine annealing, iteration-unit).
- **Optimizer hygiene**: AdamW/Adam/SGD split params into decay and no-decay groups (LayerNorm + bias excluded by default, GPT-2/BERT convention).
- **Profiler with fusion awareness**: strict per-block FLOPs (every mul and add counted) and params on the fused inference model. Blocks declare `fusion_pairs: [(absorber, absorbee), ...]` (e.g. Conv2d ↔ BatchNorm2d) and the profiler drops absorbee FLOPs/params while forcing the absorber to be counted as biased — recursively through nested blocks.
- **Training UX**:
  - **Console**: always-on stdout progress (epoch/step/loss/SGCS/LR), throttled to a windowed mean.
  - **MLflow** (optional): windowed-mean train metrics, **epoch-indexed validation metrics**, per-run note with full config + per-block FLOPs/params/output shapes, resolved-config artifact. Checkpoints are NOT uploaded — they live only in `outputs/<run>/`.
- **Devices**: `cpu`, `cuda` (selected via `CUDA_VISIBLE_DEVICES`), `mps` (Apple Silicon). For `test.py` / `infer.py`, `gpu_index` embedded in the checkpoint config cannot be applied before `torch.load()` — pass `--device cuda --gpu-index N` on the CLI to select a specific GPU with these scripts.
- **Data formats**: raw int8 LMDB (`lmdb_raw`) and single-file `.npz` (`npz`). `D` (int8 CSI) is the only required array; `Z`/`Zq` (latent arrays) are optional and only needed for `decoder_only` mode or latent-space losses. Datasets emit a separate `real_target`/`imag_target` pair offset by `target_offset` (default `1/256`) to model the 3GPP/HW int8 floor-quantization bin midpoint.
- **DataLoader knobs from YAML**: `num_workers`, `pin_memory`, `prefetch_factor`, `persistent_workers`, `drop_last` plus per-split `train_loader` / `val_loader` overrides.
- **Mixed precision (AMP)**: `training.amp.{enabled, dtype}` wraps the forward in `torch.amp.autocast` for train + val. Two fp32 islands are baked in — loss computation and MHA softmax — to prevent backprop blow-ups seen under naive fp16. `GradScaler` is only constructed for cuda + fp16.
- **`torch.compile`** (optional): `training.compile.{enabled, mode, fullgraph}` compiles encoder and decoder separately (quantizer stays uncompiled — STE backward is hostile to graph capture). Dynamic shapes are always disabled (`dynamic=False`) since DataLoader batches are fixed-size and CUDA Graph-based modes require static shapes. Checkpoints save the underlying `_orig_mod` state_dict so disk files load cleanly into compiled or uncompiled inference builds.
- **ONNX export**: encoder alone, encoder + quantizer, decoder alone, or full autoencoder. Conv↔BN / Linear↔BN1d folds are applied in-place on a deep copy before export (driven by each block's `fusion_pairs` metadata), so the exported graph matches the profiler's "fused inference model" accounting. `--no-fuse` keeps the unfused graph for debugging. Encoder-facing scopes accept a single **`input`** tensor (CNN: `(1, 2, S, P)` channels `[imag, real]`; Transformer: `(1, S, 2P)` interleaved `[i₀, r₀, i₁, r₁, …]`), bypassing the LayoutAdapter in the exported graph. AMP fp32 casts around MHA softmax are omitted — inference is always fp32.
- **Reproducibility**: fixed seeds; config + resolved config + checkpoints serialized together.

---

## Installation

The project is developed against the `torch` conda environment with Python 3.11.

```bash
# 1) Create / activate a Python 3.10+ env (conda example shown)
conda create -n torch python=3.11
conda activate torch

# 2) Install Python dependencies
pip install -r requirements.txt

# 3) Editable-install the package (recommended)
pip install -e .
```

PyTorch installation: follow [pytorch.org](https://pytorch.org) for the right CUDA / MPS / CPU build for your machine. The framework runs on all three.

**Running scripts without `pip install -e .`**: `scripts/_common.py` prepends `<repo>/src` to `sys.path`, so `python scripts/train.py …` works on a fresh checkout as long as the third-party dependencies (`torch`, `numpy`, `pyyaml`, `mlflow`, `lmdb`, `onnx`, `onnxruntime`) are available. Editable-install is still recommended because `pytest` and notebooks need `csi_comp` importable from anywhere, not just from `scripts/`.

---

## Quick start

### 1. Supply data

The framework reads two on-disk formats:

- **`npz`** — a single `.npz` file. `D` (int8 `(N, S, P, 2)` CSI) is required;
  `Z` (float32 `(N, latent_dim)` pre-quant latent) and `Zq` (post-quant latent)
  are optional — needed only for `decoder_only` mode or latent-space losses.
  Loaded fully into RAM.
- **`lmdb_raw`** — a directory-style LMDB with keys `D{i:06d}` (required);
  `Z{i:06d}` / `Zq{i:06d}` are optional (enable with `with_latent: true`).
  Use this when the dataset is too large to keep resident.

Set `data.format` to `npz` or `lmdb_raw` and `data.train_path` / `data.val_path`
accordingly.

### 2. (Optional) Start an MLflow server

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

If you don't want MLflow, set `experiment.mlflow.enabled: false` in the config (or omit the `mlflow:` block). The console logger always runs.

### 3. Train

```bash
python scripts/train.py --config configs/examples/joint_cnn.yaml
```

Output:
```
[train] mode=joint epochs=30 steps/epoch=100 device=mps
[epoch 1/30] start
[epoch 1/30 step 50/100] loss/total=0.9158 sgcs=0.0842 lr=8.15e-04
[epoch 1/30 step 100/100] loss/total=0.9001 sgcs=0.0999 lr=7.75e-04
[epoch 1/30] train done in 2.4s | loss/total=0.9079 sgcs=0.0921 lr=7.95e-04
[epoch 1/30] val | val/loss/total=0.9009 val/sgcs=0.0991 * new best best/sgcs=0.0991
...
[train] done in 79.0s | best sgcs=0.1519
```

Checkpoints land in `outputs/<experiment.name>_<YYYYMMDD_HHMMSS>/{latest,best}.pt`.
The timestamp suffix (from local time at process start) is also used as the
MLflow `run_name`, so each invocation gets its own folder/run and prior
checkpoints are never overwritten. Pass `--no-timestamp` to fall back to the
plain `outputs/<experiment.name>/` path (handy for tests / stable links).

Both `best.pt` and `latest.pt` are **symlinks** to sibling files whose names
encode the run state — `best_e{epoch:03d}_{metric}{value:.4f}.pt` /
`latest_e{epoch:03d}_{metric}{value:.4f}.pt` (e.g. `best_e023_sgcs0.8421.pt`)
— so you can see at a glance which epoch and metric produced each checkpoint
without opening it. The stable names keep working for downstream scripts
(`test.py`, `infer.py`, `export_onnx.py`). On filesystems that
refuse symlinks the code falls back to a hardlink, then a copy (copy uses
double the disk).

### 4. Other entry points

```bash
# Continue training from a checkpoint (weights only; optimizer/scheduler start fresh)
python scripts/train.py \
  --config outputs/<run>/config.resolved.yaml \
  --pretrained-checkpoint outputs/<run>/latest.pt \
  --set training.epochs=60

# Continue with a different config (e.g. new loss, scheduler, or added blocks)
python scripts/train.py \
  --config configs/examples/new_config.yaml \
  --pretrained-checkpoint outputs/<run>/best.pt

# Allow architecture mismatch (new blocks start randomly initialized)
python scripts/train.py \
  --config configs/examples/new_config.yaml \
  --pretrained-checkpoint outputs/<run>/best.pt --no-strict

# Evaluate — single checkpoint
python scripts/test.py --checkpoint outputs/<run>/best.pt

# Evaluate — single checkpoint with a specific test dataset
python scripts/test.py --checkpoint outputs/<run>/best.pt \
  --data-path /path/to/test_data.npz

# Evaluate — cross-checkpoint: encoder and decoder from separate runs
python scripts/test.py \
  --encoder-checkpoint outputs/<enc_run>/best.pt \
  --decoder-checkpoint outputs/<dec_run>/best.pt

# Evaluate — cross-checkpoint with a specific test dataset
python scripts/test.py \
  --encoder-checkpoint outputs/<enc_run>/best.pt \
  --decoder-checkpoint outputs/<dec_run>/best.pt \
  --data-path /path/to/test_data.npz

# Export to ONNX (encoder, encoder_quant, decoder, full)
python scripts/export_onnx.py \
  --checkpoint outputs/<run>/best.pt \
  --scope encoder --out exports/

# Run inference on a dataset (outputs saved to <ckpt parent>/infer_<timestamp>/)
python scripts/infer.py \
  --checkpoint outputs/<run>/best.pt \
  --data-path /path/to/data.npz

# Inference — cross-checkpoint: encoder and decoder from separate runs
python scripts/infer.py \
  --encoder-checkpoint outputs/<enc_run>/best.pt \
  --decoder-checkpoint outputs/<dec_run>/best.pt \
  --data-path /path/to/data.npz
```

`test.py` and `infer.py` inherit `training.amp` and `training.compile` from the embedded checkpoint config. To evaluate in strict fp32 without compile, override them on the CLI:

```bash
python scripts/test.py --checkpoint outputs/<run>/best.pt \
  --set training.amp.enabled=false \
  --set training.compile.enabled=false

python scripts/infer.py --checkpoint outputs/<run>/best.pt \
  --set training.amp.enabled=false \
  --set training.compile.enabled=false
```

> `training.compile.enabled=false` is sufficient on its own — `mode` and `fullgraph` are ignored when compile is disabled.

---

## Configuration cheat sheet

YAML lives in `configs/examples/`. Key top-level sections:

```yaml
experiment:
  name: my_run
  seed: 42
  device: mps                        # cpu | cuda | mps
  log_every_n_iters: 50              # console + mlflow throttle
  mlflow:                            # optional
    enabled: true
    tracking_uri: http://127.0.0.1:5000
    experiment_name: my_experiments

data:
  format: npz                        # 'npz' (single file) or 'lmdb_raw' (directory)
  train_path: /path/to/train_dataset.npz
  val_path: /path/to/valid_dataset.npz
  max_subband: 13
  max_port: 32
  layout: cnn                        # 'cnn' or 'transformer'
  batch_size: 256
  num_workers: 0                     # default for both loaders
  # Optional DataLoader knobs (commented = use defaults):
  # pin_memory: false
  # prefetch_factor: 2               # only honoured when num_workers > 0
  # persistent_workers: false        # only honoured when num_workers > 0
  # drop_last: false                   # train.py (build_dataloaders): applies to both
  #                                    # train and val loaders as a shared default.
  #                                    # test.py/infer.py (build_val_loader): ignored for
  #                                    # the val loader so every sample is evaluated;
  #                                    # use val_loader.drop_last to override explicitly.
  # train_loader: { shuffle: true,  drop_last: false }   # per-split overrides
  # val_loader:   { shuffle: false, drop_last: false }
  dataset_args:
    # latent_key: null                # null (default) | 'Zq' | 'Z' — set only for
    #                                  decoder_only mode or latent-space losses
    # target_offset: 0.00390625      # = 1/256 (default). 3GPP/HW bin-midpoint offset
    #                                  applied to reconstruction target only. 0.0 to disable.

model:
  encoder:
    # build_encoder does NOT auto-append a head; the last block's output is the
    # latent that feeds the quantizer. End with a bounded activation
    # (e.g. {name: activation, activation: tanh}) to keep the latent inside
    # the quantizer's value_range.
    blocks:
      - { name: dw_sep_conv, out_channels: 16, kernel: 3 }
      - name: residual
        main_blocks:
          - { name: dw_sep_conv, out_channels: 16, kernel: 3 }
          - { name: dw_sep_conv, out_channels: 16, kernel: 3, use_act2: false }
        skip_blocks: []
        post_activation: relu
      - { name: linear_proj, out_dim: 64, activation: relu }
      - { name: activation, activation: tanh }
  decoder:
    # build_decoder does NOT auto-append a head; the last block must produce
    # (max_subband, max_port, 2). Use reshape_head (CNN-path) or
    # complex_ffn_head (transformer-path).
    blocks:
      - { name: linear_proj, out_dim: 256, activation: relu }
      - { name: linear_proj, out_dim: "${mul:${data.max_subband},${data.max_port}}", activation: relu }
      - { name: reshape_head, max_subband: "${data.max_subband}", max_port: "${data.max_port}" }
  # Latent masking (optional) — zeros trailing elements of the quantized latent
  # before the decoder to simulate partial-bandwidth / robustness scenarios.
  # Omit this block (or set mode: full) to disable.
  # latent_mask:
  #   mode: half          # full | half | dual | random
  #   mask_ratio: 0.5     # fraction zeroed (trailing); default 0.5. At least one
  #                       # element is always kept (even at mask_ratio=1.0).

quantizer:
  type: uniform
  bits: 2
  value_range: [-1.0, 1.0]
  unit_spaced: true
  grad: ste                          # ste | soft | hard

training:
  mode: joint                        # see "Inter-vendor modes" above
  epochs: 30
  optimizer: { name: adamw, lr: 1.0e-3, weight_decay: 0.01 }
  # Default: LayerNorm + bias excluded from weight decay (GPT-2/HuggingFace
  # convention). Set `optimizer.decay_norm_bias: true` to override.
  scheduler:
    name: warmup_cosine
    warmup_steps: 200
    min_lr: 1.0e-6
  val_every_n_epochs: 1
  best_metric: { name: sgcs, mode: max }
  # Mixed precision — forward under autocast; loss + MHA softmax stay fp32.
  amp:
    enabled: false
    # dtype: bf16                    # bf16 | fp16. Defaults: cuda→bf16, mps→fp16, cpu→bf16.
  # torch.compile on encoder/decoder (quantizer stays uncompiled). Checkpoints stay portable.
  compile:
    enabled: false
    # mode: default                  # default | reduce-overhead | max-autotune
    # fullgraph: false

loss:
  terms:
    - { name: one_minus_sgcs, weight: 1.0 }
    # For latent_mask.mode: dual, use dual_one_minus_sgcs instead and set
    # best_metric to loss/total (val/sgcs only tracks the full-latent path):
    # - name: dual_one_minus_sgcs
    #   weight: 1.0
    #   params: { full_weight: 0.5, half_weight: 0.5 }
    # training.best_metric: { name: loss/total, mode: min }
```

### CLI overrides

Any key can be overridden at the command line:

```bash
python scripts/train.py --config configs/examples/joint_cnn.yaml \
  --set training.epochs=50 \
  --set training.optimizer.lr=2.0e-4 \
  --set model.encoder.blocks[1].channels=64 \
  --set experiment.device=mps
```

Lists and dicts are accepted (parsed as YAML scalars):

```bash
--set quantizer.value_range='[-2.0,2.0]'
--set 'model.encoder.blocks[2]={name: cnn_block, channels: 64, kernel: 3}'
```

---

## Adding a new block

```python
# src/csi_comp/models/blocks/se_block.py
import torch.nn as nn
from ...registry import register
from .base import Block

@register("block", "se_block")
class SEBlock(Block):
    def __init__(self, in_shape, reduction: int = 4):
        super().__init__(in_shape)
        C = self.in_shape[0]
        self.fc1 = nn.Linear(C, C // reduction)
        self.fc2 = nn.Linear(C // reduction, C)
        self.out_shape = tuple(self.in_shape)

    def forward(self, x):
        s = x.mean(dim=(-2, -1))
        s = torch.sigmoid(self.fc2(torch.relu(self.fc1(s))))
        return x * s[:, :, None, None]
```

Register it by importing the module in `src/csi_comp/models/blocks/__init__.py`:

```python
from . import (
    activation, avg_pool, cnn, complex_ffn_head, dw_sep_conv, heads, mlp,
    positional_encoding, reshape, residual, se_block, transformer,
)
```

Now `{ name: se_block, reduction: 8 }` works in any YAML.

The same pattern (`@register("loss"|"quantizer"|"dataset"|"scheduler", "...")`) extends to losses, quantizers, datasets, and schedulers.

---

## Testing

```bash
pytest                # full suite (344 tests)
pytest tests/test_amp.py tests/test_compile.py tests/test_onnx_fuse.py -v
pytest tests/test_latent_mask.py -v   # latent masking unit + integration tests
```

---

## Repository layout

```
src/csi_comp/
  registry.py            # namespaced class registry
  config/                # YAML loader + expression resolver
  data/                  # npz + lmdb_raw readers, padding collate
  models/
    blocks/              # block implementations
    encoder.py decoder.py autoencoder.py
    latent_mask.py       # LatentMaskSpec + masking helpers
  quantization/          # UniformQuantizer + gradient strategies (STE / soft / hard)
  losses/                # SGCS, MSE latent, dual SGCS, composite WeightedSumLoss
  training/
    trainer.py           # mode-agnostic loop (AMP-aware)
    builders.py          # build_model / build_optimizer / build_scheduler
    schedulers/          # registered scheduler factories
    console_logger.py    # ConsoleCallback
    mlflow_logger.py     # MLflowCallback (optional)
    checkpoint.py        # latest/best checkpointing (compile-aware)
    amp.py               # AmpSpec + autocast helpers
    compile_utils.py     # torch.compile wrap/unwrap helpers
  analysis/              # FLOPs/params profiler + MLflow note
  export/
    onnx_export.py       # ONNX export at 4 scopes
    fuse.py              # Conv↔BN / Linear↔BN1d folding for inference

scripts/                 # train.py, test.py, export_onnx.py, infer.py
configs/                 # examples/
tests/                   # pytest suite (344 tests)
```

---

## License

Research / experimental code. License TBD.
