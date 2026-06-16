# AI CSI Compression — Inter-Vendor Framework

Experimental PyTorch framework for **AI-based CSI compression** with support for **inter-vendor collaboration** (encoder and decoder trained independently or jointly, potentially by different vendors).

Built around a config-driven, registry-based block system so new encoders/decoders, losses, quantizers, and LR schedulers drop in without touching the training loop.

---

## Highlights

- **Inter-vendor training modes**: `joint`, `encoder_only`, `decoder_only`, `encoder_only_frozen_decoder`.
- **Registry-based blocks**: CNN, depthwise-separable conv, residual, average pooling, transformer (custom Q/K/V/O with selectable pre/post-LN), positional encoding (fixed sin/cos · learnable random · learnable sin/cos), reshape, linear projection, standalone activation, bounding head, reshape head, complex FFN head (per-branch real/imag projections). Add a new block with a single `@register("block", "name")` decorator.
- **Fixed-shape forward contract**: every block declares `out_shape` in `__init__` and has a single `forward(x) -> x` signature; the encoder/decoder builder propagates shapes statically. Padding masks live only on the data batch and are consumed by the loss — not threaded through the model.
- **Configurable quantizer**: uniform quantization with **decoupled forward / backward axes**. The value sent to the decoder (`quant_forward`: `hard` snap | `soft` blend) and the gradient surrogate flowing to the encoder (`quant_backward`: `identity`/STE | `soft` | `none`) are chosen independently and combined via the straight-through identity `out = surrogate + (forward_value - surrogate).detach()`. `quantizer.grad` accepts a preset string (`ste` = hard+identity, `soft` = soft+soft, `hard` = hard+none — all backward compatible) or a mapping (`{forward: hard, backward: soft, temperature: 1.0}`) — e.g. hard forward + soft backward gives the decoder exact codes (no train/eval gap) while the encoder gets a smooth gradient. Bit-width, range, and level spacing all configurable. The forward/backward choice applies only during training — in `eval()` the quantizer always hard-snaps to the nearest level, so validation / `test.py` / `infer.py` reflect the deployed (hard) quantizer (a `soft` forward would otherwise emit continuous values at eval and overstate SGCS).
- **Composite loss**: weighted sum of named loss terms (`one_minus_sgcs`, `mse_latent`, `mse_rescaled_latent`, `mse_quantized_latent`, `cross_entropy_levels`, `dual_one_minus_sgcs`, …). The latent-space terms pick their teacher (Z / Zq) per term via `params.target_key`; `cross_entropy_levels` is the discrete-label sibling of `mse_quantized_latent` — it classifies each latent element into the teacher's codeword bin (cross-entropy over quantization levels) rather than regressing the code value. It has two label modes: **hard** (default, `soft_labels: false`) snaps the teacher to a one-hot bin index (typically against Zq), while **soft** (`soft_labels: true`) turns the pre-quant teacher Z into a full distribution over levels as a knowledge-distillation soft target — `teacher_temperature` sets the label sharpness independently of the student-logit `temperature`. Add new terms via the loss registry.
- **Latent masking** (`model.latent_mask`): zero out the trailing fraction of the quantized latent before the decoder to simulate partial-bandwidth scenarios. Four modes: `full` (disabled), `half` (fixed), `random` (per-sample augmentation), `dual` (decoder called twice; loss is a weighted sum of full- and half-latent reconstructions). Applies only when both encoder and decoder are present (`joint`, `encoder_only_frozen_decoder`); automatically skipped in `encoder_only` and `decoder_only` modes.
- **Schedulers**: PyTorch built-ins (`cosine`, `step`, `none`) plus a custom `warmup_cosine` (linear warmup → cosine annealing, iteration-unit).
- **Optimizer hygiene**: AdamW/Adam/SGD split params into decay and no-decay groups (LayerNorm + bias excluded by default, GPT-2/BERT convention).
- **Profiler with fusion awareness**: strict per-block FLOPs (every mul and add counted) and params on the fused inference model. Blocks declare `fusion_pairs: [(absorber, absorbee), ...]` (e.g. Conv2d ↔ BatchNorm2d) and the profiler drops absorbee FLOPs/params while forcing the absorber to be counted as biased — recursively through nested blocks.
- **Training UX**:
  - **Console**: always-on stdout progress (epoch/step/loss/SGCS/LR), throttled to a windowed mean.
  - **MLflow** (optional): windowed-mean train metrics, **epoch-indexed validation metrics**, per-run note with full config + per-block FLOPs/params/output shapes, resolved-config artifact. Checkpoints are NOT uploaded — they live only in `outputs/<run>/`.
- **Devices**: `cpu`, `cuda` (selected by assigning `CUDA_VISIBLE_DEVICES = gpu_index` before torch is imported, then addressing `cuda:0` in-process), `mps` (Apple Silicon). `gpu_index` is authoritative — it overrides any pre-existing `CUDA_VISIBLE_DEVICES` in the environment. For `test.py` / `infer.py`, `gpu_index` embedded in the checkpoint config cannot be applied before `torch.load()` — pass `--device cuda --gpu-index N` on the CLI to select a specific GPU with these scripts.
- **Data formats**: raw int8 LMDB (`lmdb_raw`) and single-file `.npz` (`npz`). `D` (int8 CSI) is the only required array; `Z`/`Zq` (latent arrays) are optional and only needed for `decoder_only` mode or latent-space losses. Datasets emit a separate `real_target`/`imag_target` pair offset by `target_offset` (default `1/256`) to model the 3GPP/HW int8 floor-quantization bin midpoint.
- **Augmented-input training** (`augmented CSI → target CSI`): set optional `data.aug_train_path` / `data.aug_val_path` to feed an augmented (UE-condition) dataset as the **encoder input** while the reconstruction target stays the clean CSI from `train_path` / `val_path`. Omit them for the default `target CSI → target CSI`. Index-aligned, same format + `dataset_args` (latent-related args like `latent_key` / `expose_z` / `expose_zq` are stripped for the augmented dataset — it only supplies the encoder input, so its file needs no latent arrays); only meaningful when the encoder is trained. `test.py` / `infer.py` honour the embedded `aug_val_path` and expose `--aug-data-path` to override it. For phase augmentation the augmented encoder input can be scaled per channel via `scale_real` / `scale_imag` (overriding the single `scale` for `D[...,0]` / `D[...,1]`); these apply to the augmented input only (the target keeps the plain `scale`) and accept a number or a little-endian float64 hex bit pattern — set them in `data.dataset_args` or, scoped to the aug build, `data.aug_dataset_args`.
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
  `Z{i:06d}` / `Zq{i:06d}` are optional (enable with `expose_z: true` /
  `expose_zq: true`). Use this when the dataset is too large to keep resident.

Set `data.format` to `npz` or `lmdb_raw` and `data.train_path` / `data.val_path`
accordingly.

To train an `augmented CSI → target CSI` autoencoder (modelling the degraded CSI
a UE actually sees), additionally set `data.aug_train_path` / `data.aug_val_path`
to index-aligned datasets of the same format. Their samples become the **encoder
input** while the reconstruction target stays the clean CSI from
`train_path` / `val_path`. Omit them for the default `target CSI → target CSI`.
See `configs/examples/joint_cnn_aug_input.yaml`.

For phase augmentation the augmented encoder input can be scaled per channel:
`scale_real` / `scale_imag` override the single `scale` for the real
(`D[..., 0]`) / imag (`D[..., 1]`) channel. They apply to the **augmented input
only** — the reconstruction target keeps the plain `scale` — and are a no-op
without `aug_*_path`. Put them in `data.dataset_args`, or scope them purely to
the aug build with `data.aug_dataset_args` (per-aug-dataset constructor
overrides). Each scale (and `scale` itself) accepts a number **or** a hex string
parsed as the little-endian IEEE-754 float64 bit pattern
(e.g. `1/128 == "0x000000000000803F"`), so an exact double survives without
decimal round-trip loss.

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
[epoch 1/30] train done in 2.4s | loss/total=0.9079 sgcs=0.0921 lr=7.75e-04
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
without opening it. `best`'s `{value}` is the best-so-far metric; `latest`'s is
**that epoch's** validation value (or an epoch-only name like `latest_e005.pt`
when validation didn't run that epoch). The stable names keep working for downstream scripts
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

# Evaluate with augmented encoder input (augmented CSI -> target CSI)
python scripts/test.py --checkpoint outputs/<run>/best.pt \
  --data-path /path/to/test_data.npz \
  --aug-data-path /path/to/test_data_augmented.npz

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

`test.py` reports a scale+phase-**aligned NMSE** alongside SGCS in three views (all from the same per-subband aligned NMSE). Since SGCS is invariant to global scale and phase, both target and reconstruction are unit-normed and rotated to zero port-0 phase per subband before the NMSE is taken — so it measures the residual error SGCS can't see. This is a test-time-only metric (off during training validation). With `--data-path` the prefix is `test/` instead of `val/`:

| metric | meaning |
|---|---|
| `val/nmse` | linear energy ratio (dataset mean) |
| `val/nmse_db_mean` | `10·log10` of that linear mean — dB **of the mean** |
| `val/nmse_db_persb` | per-subband dB averaged in dB space — mean **of the logs** |

The last two differ because `log(mean) != mean(log)` (10·log10 is concave, so `nmse_db_persb ≤ nmse_db_mean`).

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
  # Optional augmented encoder input (augmented CSI -> target CSI). Same format +
  # dataset_args, index-aligned with train/val. Omit for target CSI -> target CSI.
  # aug_train_path: /path/to/train_dataset_augmented.npz
  # aug_val_path: /path/to/valid_dataset_augmented.npz
  # Constructor overrides applied ONLY to the augmented-input dataset (merged over
  # dataset_args, latent args excluded). Use for the per-component phase-aug scales:
  # aug_dataset_args:
  #   scale_real: "0x..."              # see dataset_args.scale_real below
  #   scale_imag: "0x..."
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
    # scale: 0.0078125                # = 1/128 (default). int sample × scale. Number OR
    #                                  hex string (little-endian float64 bit pattern,
    #                                  e.g. "0x000000000000803F" == 1/128).
    # scale_real / scale_imag:        # OPTIONAL per-channel override of `scale` (real =
    #                                  D[...,0], imag = D[...,1]); number or hex string.
    #                                  Aug-input ONLY (stripped from target/non-aug build);
    #                                  a no-op without aug_*_path. Phase augmentation.
    # latent_key: null                # null (default) | 'Zq' | 'Z' — primary latent
    #                                  slot (decoder_only input); set for decoder_only.
    # expose_z: true                  # expose Z  as latent_target_z  (npz + lmdb_raw)
    # expose_zq: false                # expose Zq as latent_target_zq (npz + lmdb_raw)
    #                                  → loss terms pick a teacher via params.target_key
    # target_offset: 0.00390625      # = 1/256 (default). 3GPP/HW bin-midpoint offset
    #                                  applied to reconstruction target only. 0.0 to disable.

model:
  encoder:
    # build_encoder does NOT auto-append a head; the last block's output is the
    # latent that feeds the quantizer. End with a bounded activation
    # (e.g. {name: activation, activation: tanh}) to keep the latent inside
    # encoder_value_range (when set) or value_range (when not set).
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
  value_range: [-1.0, 1.0]          # decoder input range; quantization levels defined here
  unit_spaced: true
  grad: ste                          # preset (ste | soft | hard), OR a two-axis mapping:
  # grad: { forward: hard|soft, backward: identity|soft|none, temperature: 1.0 }
  #   ste = {hard, identity} | soft = {soft, soft} | hard = {hard, none}
  #   (train-time only; eval always hard-snaps)
  # encoder_value_range: [-1.0, 1.0] # optional; encoder output range (e.g. tanh → [-1,1],
  #                                   # sigmoid → [0,1]). A linear transform maps encoder output
  #                                   # → value_range before quantization. Omit when the same.

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
pytest                # full suite (421 tests)
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
  quantization/          # UniformQuantizer + decoupled forward (hard/soft) / backward (identity/soft/none) axes; soft_ops shared primitive
  losses/                # SGCS, MSE latent, cross-entropy over levels, dual SGCS, composite WeightedSumLoss
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
tests/                   # pytest suite (421 tests)
```

---

## License

Research / experimental code. License TBD.
