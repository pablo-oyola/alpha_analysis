# Transolver Latent Modeling Summary And Future Directions

Date: 2026-06-26  
Workspace: `/global/cfs/projectdirs/m5300/rchurchi/alpha_analysis`

## Core Context

The trained Transolver++ model in `rchurchi/alpha_analysis` predicts global
fast ion loss from profile-grid inputs. The original input per mesh node was:

```text
rho/theta/phi coordinates       3
prs_para time/profile channels  10
prs_perp time/profile channels  10
br, bphi, bz                    3
----------------------------------
total                          26
```

The 10 time/profile points were not treated as a temporal sequence. They were
flattened into feature channels on each grid node.

The original Transolver was trained end to end:

```text
[B, N, 26] node features
  -> Transolver++ model
  -> per-node scalar values
  -> uniform masked mean over nodes
  -> global scalar prediction
  -> MSE loss against ASCOT global fast ion loss
```

The masked mean weights were not learned. Valid nodes had weight `1.0`; padded
nodes had weight `0.0`.

## What Was Implemented

### 1. Internal Token Export

Implemented:

- [`alpha_analysis/ai/export_transolver_slice_tokens.py`](alpha_analysis/ai/export_transolver_slice_tokens.py)
- [`workflow/export_transolver_slice_tokens.sbatch`](workflow/export_transolver_slice_tokens.sbatch)

This exports internal Transolver++ tokens for the train/validation set:

```text
slice_tokens
out_slice_tokens
slice_norms
```

Each sample writes one `.pt` file plus a manifest and metadata file.

Full export used later:

```text
runs/transolver_alpha/53562942/best_slice_tokens
```

Exported token shape:

```text
[layers, heads, slices, dim_head] = [4, 8, 32, 16]
```

### 2. Latent Scalar Head And Conditional Generator

Implemented:

- [`alpha_analysis/ai/train_latent_generator.py`](alpha_analysis/ai/train_latent_generator.py)
- [`workflow/train_latent_generator.sbatch`](workflow/train_latent_generator.sbatch)

The training has two stages.

Stage 1:

```text
real extracted Transolver latent -> scalar_head -> ASCOT fraction_lost
```

Stage 2:

```text
condition + noise -> generator -> generated latent
generated latent -> frozen scalar_head -> scalar loss
```

Generator loss:

```text
loss =
  latent_weight * latent_mse
  + scalar_weight * scalar_mse
  + consistency_weight * scalar_head_consistency
```

Default condition vector:

```text
R_lmn coefficients
Z_lmn coefficients
prs_para summary stats
prs_perp summary stats
br/bphi/bz summary stats
```

The completed run used:

```text
token_kind = out_slice_tokens
layers = last
num_tokens = 32
latent_dim = 128
condition_dim = 4584
train_records = 462
val_records = 116
```

Output directory:

```text
runs/transolver_alpha/53562942/best_slice_tokens/latent_generator_54945515
```

## Key Results

### Original Transolver Validation

From `fraction_lost_validation.csv`:

```text
n = 116
R2  = 0.9401
MAE = 0.01327
MSE = 0.0003287
```

### Scalar Head On Real Extracted Latents

Plot:

- [`scalar_head_predicted_vs_ground_truth.png`](runs/transolver_alpha/53562942/best_slice_tokens/latent_generator_54945515/scalar_head_predicted_vs_ground_truth.png)

Metrics:

```text
Train:
  n = 462
  MAE = 0.00793
  RMSE = 0.01052
  R2 = 0.9819

Validation:
  n = 116
  MAE = 0.01024
  RMSE = 0.01305
  R2 = 0.9690
```

Interpretation:

```text
The extracted Transolver latents contain strong scalar-loss information.
The original Transolver uniform-mean readout was not the best possible scalar readout from those features.
```

### Conditional Generator

Best generator checkpoint:

- [`generator_best.pt`](runs/transolver_alpha/53562942/best_slice_tokens/latent_generator_54945515/generator_best.pt)

Best epoch by total validation loss:

```text
epoch = 21
train_latent_mse = 0.02782
val_latent_mse   = 0.23345
val_mae          = 0.01361
```

Final epoch 300:

```text
train_latent_mse = 0.00460
val_latent_mse   = 0.25762
val_mae          = 0.01446
```

Interpretation:

```text
The generator learns training latents strongly.
Validation behavior peaks early.
Use generator_best.pt, not last.pt.
```

## Important Discovery: Last-Layer `out_slice_tokens` Collapse Across Slices

A visualization made the generated and ground-truth latents appear constant
along the slice dimension. Direct tensor inspection confirmed this was real.

For a representative sample:

```text
slice_tokens:
  layer 0: mean channel std over slices = 0.5658
  layer 1: mean channel std over slices = 0.0326
  layer 2: mean channel std over slices = 0.0021
  layer 3: mean channel std over slices = 0.0020

out_slice_tokens:
  layer 0: mean channel std over slices = 0.1177
  layer 1: mean channel std over slices = 0.00015
  layer 2: mean channel std over slices = ~1e-10
  layer 3: mean channel std over slices = ~1e-10
```

Conclusion:

```text
late-layer out_slice_tokens are effectively a global broadcast vector repeated across slices.
```

This makes sense because the original task was:

```text
full 10-frame field -> one global scalar loss
```

The model had no need to preserve slice-resolved structure in the final
post-attention slice tokens.

## Main Interpretation

The real extracted latents are useful for scalar prediction, but the first
chosen target:

```text
last-layer out_slice_tokens
```

is not a good slice-resolved latent for generation. It behaves more like one
global context vector repeated over 32 slices.

The scalar head got higher validation R2 than the original Transolver because:

1. It uses the Transolver as a learned feature extractor.
2. It reads the latent channels directly with a nonlinear readout.
3. The original model's scalar readout was constrained to:

```text
per-node scalar -> uniform mean over nodes
```

## Recommended Future Directions

### 1. Re-export And Train On `slice_tokens`

Next target should be:

```text
--token-kind slice_tokens --layers all
```

or at minimum:

```text
--token-kind slice_tokens --layers last
```

Reason:

```text
slice_tokens retain slice structure much better than late out_slice_tokens.
```

### 2. Train Scalar Head On Several Candidate Latents

Compare validation metrics for:

```text
last-layer out_slice_tokens
last-layer slice_tokens
all-layer slice_tokens
layer-0 out_slice_tokens
all-layer slice_tokens + out_slice_tokens
```

The goal is to identify the smallest latent that:

```text
preserves scalar prediction
and
retains meaningful slice variation
```

### 3. Check Latent Slice Variance Before Training A Generator

Before generator training, compute:

```text
std across slices
cosine similarity between slices
effective rank / PCA spectrum
```

This should prevent training on another collapsed latent target.

### 4. Add Checkpoint Selection By Scalar-Facing Metrics

Current generator best checkpoint is selected by total validation loss, dominated
by latent reconstruction. Add checkpointing for:

```text
best val_mae
best val_scalar_mse
best val_latent_mse
```

This will distinguish:

```text
best physical latent reconstruction
vs
best scalar-predictive generated latent
```

### 5. Consider A Temporal Prediction Objective

If the real goal is a latent evolution model, scalar-only supervision is too
compressive. A better objective would be:

```text
field_t0 -> future latent sequence
```

or:

```text
field_t -> field_t+1 latent transition
```

This would force the latent to retain spatial and temporal structure.

### 6. Consider A Purpose-Built Encoder

A future architecture could be:

```text
encoder: field or condition -> latent tokens
dynamics/generator: latent_t -> latent_t+1 or latent sequence
readout: latent sequence -> scalar global fast ion loss
```

This would expose the bottleneck intentionally rather than relying on a late
internal tensor from a scalar-only Transolver.

## Presentation-Ready Artifact List

Code:

- [`alpha_analysis/ai/export_transolver_slice_tokens.py`](alpha_analysis/ai/export_transolver_slice_tokens.py)
- [`alpha_analysis/ai/train_latent_generator.py`](alpha_analysis/ai/train_latent_generator.py)
- [`workflow/export_transolver_slice_tokens.sbatch`](workflow/export_transolver_slice_tokens.sbatch)
- [`workflow/train_latent_generator.sbatch`](workflow/train_latent_generator.sbatch)

Run artifacts:

- [`workflow/logs/train_latent_generator_54945515.log`](workflow/logs/train_latent_generator_54945515.log)
- [`runs/transolver_alpha/53562942/best_slice_tokens/latent_generator_54945515/config.json`](runs/transolver_alpha/53562942/best_slice_tokens/latent_generator_54945515/config.json)
- [`runs/transolver_alpha/53562942/best_slice_tokens/latent_generator_54945515/metrics.jsonl`](runs/transolver_alpha/53562942/best_slice_tokens/latent_generator_54945515/metrics.jsonl)

Plots:

- [`scalar_head_predicted_vs_ground_truth.png`](runs/transolver_alpha/53562942/best_slice_tokens/latent_generator_54945515/scalar_head_predicted_vs_ground_truth.png)
- [`generated_vs_ground_truth_latents_val_examples.png`](runs/transolver_alpha/53562942/best_slice_tokens/latent_generator_54945515/generated_vs_ground_truth_latents_val_examples.png)

Metrics:

- [`scalar_head_eval_metrics.json`](runs/transolver_alpha/53562942/best_slice_tokens/latent_generator_54945515/scalar_head_eval_metrics.json)
- [`generated_vs_ground_truth_latents_val_examples.json`](runs/transolver_alpha/53562942/best_slice_tokens/latent_generator_54945515/generated_vs_ground_truth_latents_val_examples.json)

