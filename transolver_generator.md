# Transolver Spatiotemporal Generator Plan

## Goal

Replace the current one-latent-per-simulation workflow with an explicit per-frame latent model:

```text
ASCOT frame X_t -> frame encoder E -> latent z_t -> physical decoder D -> reconstructed frame X_hat_t
z_t -> latent dynamics F -> z_hat_{t+1} -> decoder D -> predicted future frame
```

The encoder and decoder are trained jointly as an autoencoder. No latent labels are required. The latent is created by an explicit terminal slice bottleneck that stops before Transolver's normal deslice operation.

The first implementation is deterministic. Diffusion, flow matching, and Diffusion Forcing are future work after deterministic reconstruction and rollout are validated.

## Core design decision

A stock Transolver++ block performs:

```text
full mesh -> slice tokens -> attention -> deslice -> full mesh
```

The new frame encoder will use:

```text
full mesh
  -> ordinary mesh-to-mesh Transolver blocks
  -> terminal SliceBottleneck
  -> latent tokens z_t [B, G, D]
```

`SliceBottleneck` performs learned node-to-token aggregation and token-space processing, but does not deslice.

The decoder must not reuse encoder slice weights, because generated latents do not have a true pressure frame from which to compute those weights. Instead, static node queries formed from coordinates, magnetic field, time, and optional equilibrium context cross-attend to the latent tokens.

---

# 1. Files to add

```text
alpha_analysis/ai/frame_dataset.py
alpha_analysis/ai/transolver_autoencoder.py
alpha_analysis/ai/train_frame_autoencoder.py
alpha_analysis/ai/export_frame_latents.py
alpha_analysis/ai/latent_dynamics.py
alpha_analysis/ai/train_latent_dynamics.py
alpha_analysis/ai/evaluate_frame_autoencoder.py
alpha_analysis/ai/evaluate_latent_rollout.py

workflow/train_frame_autoencoder.sbatch
workflow/export_frame_latents.sbatch
workflow/train_latent_dynamics.sbatch
workflow/plot_frame_reconstructions.py
workflow/plot_latent_rollouts.py

tests/test_frame_dataset.py
tests/test_transolver_autoencoder.py
tests/test_latent_dynamics.py
```

Update `pyproject.toml` and `alpha_analysis/ai/README.md` with entry points and usage.

Do not modify the upstream checkout under `no_sync/`. Reuse upstream modules only through imports; implement the new no-deslice bottleneck and decoder in tracked repository code.

---

# 2. Explicit frame dataset

## 2.1 Sequence representation

Implement `AscotSequenceDataset`, one item per ASCOT simulation:

```python
{
    "folder": str,
    "coordinates": Tensor,   # [N, 3]
    "bfield": Tensor,        # [N, 3]
    "profiles": Tensor,      # [T, N, 2], channels=(prs_para, prs_perp)
    "times": Tensor,         # [T]
    "context": dict,         # R_lmn, Z_lmn and future static/source inputs
    "target": Tensor,        # scalar fast-ion loss
    "grid_shape": tuple,
}
```

Validate that `prs_para` and `prs_perp` have matching shapes and a leading time axis. Do not hard-code `T=10`.

Implement `AscotFrameDataset` as a view over a fixed set of simulation indices. One item corresponds to `(simulation, frame)` and returns:

```python
{
    "folder": str,
    "simulation_index": int,
    "frame_index": int,
    "time": Tensor,
    "coordinates": Tensor,   # [N, 3]
    "bfield": Tensor,        # [N, 3]
    "profile": Tensor,       # [N, 2]
    "context": dict,
    "target": Tensor,
    "grid_shape": tuple,
}
```

Split by simulation before creating frame views. No simulation may appear in more than one split.

## 2.2 Time

Use physical profile times if reliably available in HDF5. Otherwise use:

```python
time = frame_index / max(T - 1, 1)
```

Record the time convention in run metadata.

## 2.3 Normalization

Fit statistics on training simulations only.

Baseline:

1. signed `log1p` for `prs_para` and `prs_perp`;
2. per-channel standardization of transformed profiles;
3. standardization of `br`, `bphi`, `bz`;
4. stable coordinate normalization using known ranges or training-set statistics;
5. inverse transforms available for physical-space metrics.

Store statistics in checkpoints and latent-export metadata.

## 2.4 Node sampling

The new pipeline must not silently default to `max_nodes=16384`.

Use Python `None` as the default, exposed as `--max-nodes none` or by omitting the argument. Print original and used node counts at startup.

If subsampling is explicitly enabled:

- apply identical indices to coordinates, B field, encoder profile, and reconstruction target;
- use reproducible training sampling;
- use fixed validation subsets;
- use full-mesh validation by default when memory allows;
- save node indices in diagnostic exports.

## 2.5 Dataset tests

Verify:

- exact `[T,N,2]` construction and channel order;
- frame-to-sequence indexing;
- simulation-grouped train/validation split;
- train-only normalization statistics;
- consistent node subsampling;
- variable `T` support.

---

# 3. Frame autoencoder

## 3.1 Initial default architecture

```text
hidden_dim: 128
encoder mesh blocks: 2
encoder heads: 4 or 8
encoder internal slice_num: 32
explicit latent tokens G: 16
latent_dim: 64
bottleneck token blocks: 1
decoder cross-attention heads: 4
decoder mesh refinement blocks: 1
```

All values must be configurable.

## 3.2 FrameEncoder

Per-node input features:

```text
rho, theta, phi
br, bphi, bz
normalized prs_para_t, prs_perp_t
normalized time broadcast to nodes
```

Architecture:

```text
input MLP
 -> 2 Transolver_plus_block(last_layer=False)
 -> SliceBottleneck
 -> z_t [B,G,latent_dim]
```

Optional equilibrium context should be compressed separately and added as a context embedding. Do not repeat all `R_lmn/Z_lmn` coefficients at every node without compression.

Suggested interface:

```python
@dataclass
class EncoderOutput:
    latent: Tensor               # [B,G,D]
    pre_attention_tokens: Tensor # optional diagnostics
    slice_weights: Tensor        # optional diagnostics
    slice_norms: Tensor

class FrameEncoder(nn.Module):
    def forward(profile, coordinates, bfield, time, context=None, node_mask=None): ...
```

## 3.3 SliceBottleneck

Implement a tracked module based on the slicing portion of `Physics_Attention_1D_Eidetic`:

```text
node features [B,N,C]
 -> per-head node features [B,H,N,Dh]
 -> slice logits [B,H,N,G]
 -> masked softmax assignments
 -> occupancy-normalized weighted aggregation [B,H,G,Dh]
 -> token self-attention with residual
 -> token MLP with residual
 -> merge/project heads
 -> latent [B,G,D]
```

Requirements:

- deterministic softmax assignments for the initial baseline;
- learned positive temperature with lower bound;
- proper node-mask handling;
- learned slot embeddings to anchor token indices;
- return occupancy diagnostics;
- no deslice operation;
- no all-reduce across unrelated data-parallel samples.

Optional anti-collapse losses should be implemented but default to zero:

- occupancy-balance loss;
- token-variance floor;
- token decorrelation loss.

## 3.4 LatentToMeshDecoder

Build node queries only from information available without the target frame:

```text
coordinates + B field + time + compressed static context
 -> query MLP [B,N,C]
 -> cross-attention(query=node queries, key/value=latent tokens)
 -> residual node MLP
 -> optional Transolver_plus_block(last_layer=False)
 -> output MLP
 -> reconstructed [prs_para_t, prs_perp_t] [B,N,2]
```

Suggested interface:

```python
class LatentToMeshDecoder(nn.Module):
    def forward(latent, coordinates, bfield, time, context=None, node_mask=None): ...
```

The decoder must work with an externally supplied latent and must never read the true target pressure.

## 3.5 Autoencoder wrapper

```python
class TransolverFrameAutoencoder(nn.Module):
    def encode(...): ...
    def decode(...): ...
    def forward(...):
        encoded = self.encode(...)
        reconstruction = self.decode(encoded.latent, ...)
        return encoded, reconstruction
```

Train encoder and decoder jointly. The reconstruction loss creates the latent; there are no precomputed latent targets.

## 3.6 Losses

Initial loss:

```text
L_AE = masked Huber(reconstruction, target)
       + integral_weight * integrated-profile error
```

Start with `integral_weight=0.01`, configurable.

Report parallel and perpendicular pressure errors separately.

Optional later terms:

- structured-grid gradient loss;
- occupancy balance;
- token variance/decorrelation.

Do not calculate finite-difference gradient loss on a random node subset.

## 3.7 Training CLI

Implement `train-alpha-frame-autoencoder` with at least:

```text
--results-root
--save-dir
--train-fraction
--seed
--batch-size
--epochs
--lr
--weight-decay
--grad-clip
--max-nodes
--full-mesh-validation
--hidden-dim
--encoder-layers
--encoder-heads
--encoder-slice-num
--latent-tokens
--latent-dim
--decoder-hidden-dim
--decoder-heads
--decoder-layers
--field-loss
--integral-weight
--slice-balance-weight
--device
--dry-run
```

Save:

```text
best_field.pt
best_total.pt
last.pt
config.json
metrics.jsonl
train_folders.txt
val_folders.txt
```

Checkpoints must include model config, normalization statistics, split indices, optimizer state, and metrics.

## 3.8 Autoencoder evaluation

Report:

- transformed-space MAE/RMSE by channel;
- physical-space MAE/RMSE after inverse transform;
- integrated-pressure relative error;
- error by frame index/time;
- token variance and pairwise cosine similarity;
- effective rank/PCA spectrum;
- slice occupancy and dead-token count;
- node-subsampling sensitivity.

Plot ground truth, reconstruction, and difference for selected frames, plus latent heatmaps over time.

### Autoencoder acceptance gate

Proceed only when:

1. validation reconstruction beats a training-mean field baseline;
2. the decoder demonstrably uses the latent;
3. time-dependent spatial variation is reconstructed;
4. latent effective rank and token variance are reported and nontrivial;
5. at least one full-mesh validation case runs successfully.

---

# 4. Per-frame latent export

Implement `export-alpha-frame-latents`.

For every simulation, encode all frames deterministically and save:

```python
{
    "folder": str,
    "dataset_index": int,
    "split": str,
    "times": Tensor,       # [T]
    "latents": Tensor,     # [T,G,D]
    "slice_norms": Tensor, # optional
    "target": Tensor,
    "grid_shape": tuple,
    "node_count": int,
    "node_indices": Tensor | None,
}
```

Write `manifest.jsonl`, `metadata.json`, and split directories. Preserve the autoencoder split exactly.

Default export should use full mesh and deterministic evaluation. If an explicit node limit is required, record it in every relevant metadata file.

Add a repeatability test: repeated exports with identical settings must match within tolerance.

---

# 5. Deterministic latent dynamics

## 5.1 Baseline model

Start with a shared residual transition:

```text
z_t + time + dt + static context -> delta_z_t
z_hat_{t+1} = z_t + delta_z_t
```

Use only pre-simulation/static context. Do not use summary statistics of future pressure frames.

Inputs:

```text
z_t [B,G,D]
current time and dt
R_lmn/Z_lmn encoded to context tokens
available source/marker/simulation parameters
learned latent-slot embeddings
```

Architecture:

```text
latent projection
 + slot/time embeddings
 + cross-attention or concatenation with context tokens
 -> 2-4 transformer blocks over latent tokens
 -> residual delta_z [B,G,D]
```

## 5.2 Training stages

Freeze encoder and decoder initially.

### Stage A: one-step warm start

```text
L_1 = MSE(z_hat_{t+1}, z_{t+1})
```

Use all transitions from each training simulation.

### Stage B: three-step free rollout

Start from true `z_t`, then feed predictions back into the model for up to three steps.

```text
L_latent = sum_k MSE(z_hat_{t+k}, z_{t+k})
L_field  = sum_k Huber(D(z_hat_{t+k}), profile_{t+k})
L_dyn    = L_latent + decode_weight * L_field
```

Start with `decode_weight=0.1`, configurable.

### Stage C: full rollout

Start from true `z_0` and predict to the final frame. Report each horizon separately.

## 5.3 Baselines

Compare against:

- latent persistence: `z_hat_{t+1}=z_t`;
- field persistence;
- linear latent transition fitted on training data;
- optional direct `z_0 -> z_1...z_T` sequence predictor.

## 5.4 Dynamics evaluation

Report:

- latent error by horizon;
- decoded field MAE/RMSE by horizon and channel;
- integrated-pressure error by horizon;
- latent norm and token-variance drift;
- comparison against persistence and linear baselines;
- free-rollout plots for actual low-, median-, and high-loss validation cases.

### Dynamics acceptance gate

Proceed to stochastic modeling only when free rollout beats persistence on held-out simulations in decoded physical-field metrics over multiple horizons.

---

# 6. Fast-ion-loss readout

Add an optional sequence head:

```text
z_0...z_T -> time embeddings -> transformer/attention pooling -> scalar loss
```

Evaluate separately on:

1. true encoded latent sequences;
2. free-rollout latent sequences.

After frozen autoencoder and dynamics training works, optionally fine-tune encoder, decoder, dynamics, and scalar head jointly with a smaller learning rate for the pretrained components:

```text
L = reconstruction loss + rollout field loss + scalar loss
```

Save separate best checkpoints for field rollout and scalar prediction so scalar supervision cannot silently collapse the representation again.

---

# 7. Deferred probabilistic phase

Do not implement Diffusion Forcing in the first Codex pass.

After deterministic rollout is validated, model:

```text
p(z_1,...,z_T | z_0, static context)
```

Suggested order:

1. conditional Gaussian residual model;
2. conditional flow matching;
3. Diffusion Forcing over `[time, latent-token]` tokens.

Repeat-seed ASCOT cases are needed before interpreting generated variance as physical or Monte Carlo variability.

---

# 8. Tests and implementation constraints

Required tests:

- frame dataset shapes and split isolation;
- no-deslice bottleneck output shape `[B,G,D]`;
- masks exclude padded nodes from slicing and losses;
- gradients flow through encoder and decoder;
- decoder reconstructs from externally supplied latent without target pressure;
- deterministic evaluation is repeatable;
- tiny synthetic autoencoder can overfit a few frames;
- latent export matches direct encoder output;
- dynamics rollout shape and gradient tests;
- persistence-baseline evaluation.

Fail loudly on unexpected profile ranks, mismatched grids, nonfinite statistics, empty splits, invalid gradient loss on subsampled nodes, and checkpoint/config mismatches.

Preserve the current scalar workflow files. The new pipeline must coexist until validated.

---

# 9. Recommended grouped commits for Codex

## Commit 1: Dataset

- frame/sequence dataset;
- normalization and split persistence;
- node-sampling policy;
- tests and dry run.

## Commit 2: Autoencoder modules

- `SliceBottleneck`;
- frame encoder;
- latent-to-mesh decoder;
- wrapper and unit tests;
- tiny synthetic overfit test.

## Commit 3: Autoencoder workflows

- training CLI;
- checkpointing and metrics;
- evaluation/plots;
- Slurm script.

## Commit 4: Latent export

- deterministic export;
- manifest/metadata;
- repeatability tests;
- Slurm script.

## Commit 5: Dynamics

- latent dataset loader;
- residual transition model;
- one-step and rollout training;
- decoded-field loss;
- persistence and linear baselines;
- tests and Slurm script.

## Commit 6: Scalar head and documentation

- optional sequence scalar head;
- evaluation on true versus rolled-out latents;
- README and commands;
- no diffusion implementation.

---

# 10. Definition of done

The first implementation is complete when:

1. time is explicit and splits are by simulation;
2. the autoencoder contains a tracked terminal no-deslice bottleneck;
3. encoder and decoder train jointly without latent labels;
4. decoder queries use no true target pressure;
5. deterministic `[T,G,D]` latent export works;
6. deterministic dynamics can execute a full free rollout;
7. rollout evaluation includes decoded physical-field metrics and persistence baselines;
8. no future profile statistics enter dynamics conditioning;
9. existing scalar Transolver workflows remain operational;
10. tests and CPU smoke runs pass;
11. Perlmutter scripts are present but production runs are not required from Codex;
12. probabilistic generation remains a documented future phase.

## Questions Codex should resolve from the actual data

Document answers in the README/run metadata rather than guessing:

- whether physical frame times and units are stored;
- whether all simulations share grid shape and frame count;
- whether pressure fields can be negative;
- what integration weights or cell volumes are available;
- which source/marker parameters are available before ASCOT;
- whether full-mesh training fits on one Perlmutter GPU;
- whether direct import of upstream `Transolver_plus_block` is stable.
