# Transolver Latent Modeling Thread

Date range: 2026-06-23 to 2026-06-26  
Workspace: `/global/cfs/projectdirs/m5300/rchurchi/alpha_analysis`

This document records the technical thread about the trained Transolver++ model,
its internal eidetic slice tokens, latent token export, scalar readout training,
conditional latent generation, and interpretation of the results. It omits hidden
system/developer instructions and raw command output except where the output
matters for the technical record.

## 1. Initial Question: Can the Eidetic Physical States Be Extracted?

User asked whether, for the Transolver model trained in `rchurchi/alpha_analysis`,
it was possible to extract the part of the trained model that creates the
eidetic physical states as an encoder for other contexts, or whether the model
was too entangled. The user also asked whether the eidetic states were formed
using the full temporal evolution or frame by frame.

### Code Inspected

Relevant files:

- [`alpha_analysis/ai/train_transolver.py`](alpha_analysis/ai/train_transolver.py)
- [`alpha_analysis/ai/dataloader.py`](alpha_analysis/ai/dataloader.py)
- [`no_sync/transolver_plus-src/models/Transolver_plus.py`](no_sync/transolver_plus-src/models/Transolver_plus.py)
- Saved run configs under [`runs/transolver_alpha`](runs/transolver_alpha)

### Findings

The trained Transolver++ model can be instrumented to extract its internal
eidetic state construction, but those internal states are not a clean standalone
encoder in the usual sense. They are task-specific intermediate tensors trained
only to support scalar global fast ion loss prediction.

The important module is:

```python
Physics_Attention_1D_Eidetic
```

in [`no_sync/transolver_plus-src/models/Transolver_plus.py`](no_sync/transolver_plus-src/models/Transolver_plus.py).

Within each attention block:

```text
node hidden states
  -> in_project_x
  -> in_project_slice + gumbel/softmax slice assignment
  -> slice_token
  -> self-attention among slice tokens
  -> out_slice_token
  -> scatter back to node hidden states
```

So the internal tokens can be captured mechanically, but their semantic meaning
is entangled with the scalar prediction task.

## 2. Temporal Handling in the Original Transolver Training

The original Transolver training did not use an explicit temporal axis. The 10
time/profile points were flattened into feature channels on each grid node.

The per-node input features were:

```text
rho/theta/phi coordinates       3
prs_para time/profile channels  10
prs_perp time/profile channels  10
br, bphi, bz                    3
----------------------------------
total                          26
```

The saved model configs confirm:

```json
"space_dim": 26
```

So the model input was:

```text
[num_nodes, 26]
```

There was no frame-by-frame processing, no temporal attention, no recurrence,
and no autoregressive evolution model. The full 10-point profile history was
presented as static channels per mesh node.

## 3. How the Original Transolver Was Trained

The original model was trained end to end.

There was no separate Transolver encoder solve followed by a separate MLP. The
training path was:

```text
x: [B, N, 26]
  -> TransolverPlusModel
  -> per-node scalar values [B, N]
  -> uniform masked mean over nodes
  -> global scalar prediction
  -> MSE loss against global fast ion loss
```

The relevant code is in [`alpha_analysis/ai/train_transolver.py`](alpha_analysis/ai/train_transolver.py):

```python
node_values = model((x, pos, None)).squeeze(-1)
weights = mask.float()
prediction = (node_values * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
```

The weights are not learned. They are just the binary validity mask converted
to float. Valid sampled nodes get weight `1.0`; padded nodes get `0.0`.

For normal single-sample training with `max_nodes=16384`, the scalar prediction
was simply the equal mean of the sampled node outputs.

The final node-level output head is inside the standard Transolver++ model:

```text
final Transolver block:
  LayerNorm(hidden_dim)
  -> Linear(hidden_dim, out_dim=1)
```

That per-node scalar field is not directly supervised. Only its average is
supervised against the global scalar target.

## 4. Proposed Low-Risk Route: Distill Internal Tokens

The user wanted to try a low-risk route before training a new encoder:

```text
full 10-frame field
  -> trained Transolver
  -> capture internal slice tokens
```

Then train downstream models in the extracted latent space.

The assistant recommended first exporting both:

- `slice_tokens`: compressed per-slice summaries before attention
- `out_slice_tokens`: slice tokens after slice-to-slice attention

The initial suggestion was to start with the last-layer `out_slice_tokens`,
because they are closest to what the scalar-loss model actually uses.

## 5. Token Export Code Implemented

The user asked for code to take in the trained Transolver model, run the entire
train/validation dataset through it, and capture the internal slice tokens.

### Files Added

- [`alpha_analysis/ai/export_transolver_slice_tokens.py`](alpha_analysis/ai/export_transolver_slice_tokens.py)
- [`workflow/export_transolver_slice_tokens.sbatch`](workflow/export_transolver_slice_tokens.sbatch)

### File Modified

- [`pyproject.toml`](pyproject.toml)

Added console entry:

```toml
export-alpha-transolver-slice-tokens = "alpha_analysis.ai.export_transolver_slice_tokens:main"
```

### Export Behavior

The exporter:

1. Loads a trained Transolver checkpoint and its saved config.
2. Rebuilds the same train/validation split.
3. Patches `Physics_Attention_1D_Eidetic.forward` to capture:
   - `slice_tokens`
   - `out_slice_tokens`
   - `slice_norms`
4. Runs each train/val sample through the model.
5. Writes one `.pt` file per sample plus:
   - `metadata.json`
   - `manifest.jsonl`

The exporter defaults to deterministic softmax slice assignment rather than
Gumbel sampling, for stable teacher latents.

### Exported Payload Fields

Each sample `.pt` contains:

```text
split
split_index
dataset_index
folder
target
prediction
grid_shape
node_count
original_node_count
node_indices
attention_module_names
slice_tokens
out_slice_tokens
slice_norms
```

### Smoke Test Result

A two-sample smoke export produced:

```text
slice_tokens:     [4, 8, 32, 16]
out_slice_tokens: [4, 8, 32, 16]
slice_norms:      [4, 8, 32]
```

Meaning:

```text
4 layers
8 heads
32 slices
16 dimensions per head
```

Full export location used later:

```text
runs/transolver_alpha/53562942/best_slice_tokens
```

## 6. Interpreting `slice_tokens` vs `out_slice_tokens`

The user asked how to think about `slice_tokens` versus `out_slice_tokens`.

The answer:

```text
mesh/node hidden states
  -> soft assignment of nodes to learned slices
  -> slice_tokens
  -> attention between slices
  -> out_slice_tokens
  -> scatter back to mesh/node hidden states
```

So:

```text
slice_token[g] = weighted average of node hidden states assigned to slice g
out_slice_token[g] = attention(slice_token[g], all other slice_tokens)
```

`slice_tokens` are closer to an encoder bottleneck.  
`out_slice_tokens` are contextualized latents after slice-to-slice communication.

At that point, we expected `out_slice_tokens` to be useful for scalar prediction.
Later diagnostics showed that late-layer `out_slice_tokens` had collapsed along
the slice dimension.

## 7. Latent Diffusion / Generator Concept

The user wanted a latent generator:

```text
condition -> generated Transolver latent tokens -> scalar loss
```

Possible conditions discussed:

1. First frame only.
2. Equilibrium data:
   - VMEC/DESC `R_lmn`
   - VMEC/DESC `Z_lmn`
   - pressure representation
3. Hybrid:
   - equilibrium tokens plus compressed first-frame tokens

The first-frame mesh was judged too large and redundant for an initial model.
The recommended first conditioning path was equilibrium and summary data.

## 8. Scalar Head Concept

The assistant introduced the term "scalar head" to mean:

```text
exported latent tokens -> scalar global fast ion loss
```

This scalar head is not the original Transolver head. It is a new readout model
trained directly on extracted latents.

The purpose was diagnostic:

```text
real teacher latents -> scalar head -> ASCOT target
```

If this works, then the extracted latents contain loss-relevant information.

## 9. Label-Preserving Latent Augmentation Discussion

The user proposed:

```text
condition c_i
  -> generate many synthetic latent samples z_i_j
  -> train scalar head with all z_i_j assigned to same scalar y_i
```

The assistant cautioned that this is label-preserving latent augmentation, not
new physical supervision. It can help as regularization if the generated
variation is nuisance variation, but it can also teach the scalar head to ignore
directions that might be physically meaningful.

The recommended first version was:

```text
Train scalar_head(z_i) -> y_i

Then train generator(c_i, noise) -> z_hat_i

loss =
  latent_reconstruction(z_hat_i, z_i)
  + lambda_scalar * scalar_loss(scalar_head(z_hat_i), y_i)
  + lambda_consistency * consistency(scalar_head(z_hat_i), scalar_head(z_i))
```

## 10. Latent Generator Code Implemented

The user asked to implement that first version.

### Files Added

- [`alpha_analysis/ai/train_latent_generator.py`](alpha_analysis/ai/train_latent_generator.py)
- [`workflow/train_latent_generator.sbatch`](workflow/train_latent_generator.sbatch)

### File Modified

- [`pyproject.toml`](pyproject.toml)

Added console entry:

```toml
train-alpha-latent-generator = "alpha_analysis.ai.train_latent_generator:main"
```

### Default Setup

The script defaults to:

```text
token_kind: out_slice_tokens
layers: last
scalar_target: target
```

The latent is reshaped:

```text
last-layer out_slice_tokens:
  original: [heads, slices, dim_head] = [8, 32, 16]
  used:     [slices, heads * dim_head] = [32, 128]
```

### Condition Vector

The condition vector includes:

```text
R_lmn coefficients
Z_lmn coefficients
prs_para summary stats
prs_perp summary stats
br/bphi/bz summary stats
```

For the run below:

```text
condition_dim = 4584
```

### Model Architecture

#### `TokenScalarHead`

Input:

```text
latent tokens [B, 32, 128]
```

Architecture:

```text
Linear token projection
learned CLS token
learned position embeddings
TransformerEncoder
MLP readout
scalar output
```

#### `ConditionalLatentGenerator`

Input:

```text
condition [B, 4584]
noise [B, 64]
```

Output:

```text
generated latent [B, 32, 128]
```

Architecture:

```text
condition + noise
  -> MLP condition encoder
  -> context vector

learned latent queries [32 tokens]
  + learned position embeddings
  + broadcast context vector
  -> TransformerEncoder
  -> output projection
  -> generated latent tokens
```

This is a stochastic conditional regressor, not a full diffusion model yet.

### Training Stages

Stage 1:

```text
scalar_head(real Transolver latent) -> ASCOT fraction_lost
```

Stage 2:

```text
generator(condition + noise) -> generated latent
frozen scalar_head(generated latent) -> scalar
```

Generator loss:

```text
loss =
  latent_weight * latent_mse
  + scalar_weight * scalar_mse
  + consistency_weight * scalar_head_consistency
```

Default loss weights:

```text
latent_weight = 1.0
scalar_weight = 0.1
consistency_weight = 0.1
```

## 11. Latent Generator Run 54945515

The user ran:

```text
workflow/logs/train_latent_generator_54945515.log
```

### Run Status

Slurm reported:

```text
Job:       54945515
State:     COMPLETED
ExitCode:  0:0
Elapsed:   01:51:46
```

The run completed:

```text
scalar epochs:    200 / 200
generator epochs: 300 / 300
```

### Run Config

Output directory:

```text
runs/transolver_alpha/53562942/best_slice_tokens/latent_generator_54945515
```

Files:

- [`config.json`](runs/transolver_alpha/53562942/best_slice_tokens/latent_generator_54945515/config.json)
- [`metrics.jsonl`](runs/transolver_alpha/53562942/best_slice_tokens/latent_generator_54945515/metrics.jsonl)
- [`scalar_head_best.pt`](runs/transolver_alpha/53562942/best_slice_tokens/latent_generator_54945515/scalar_head_best.pt)
- [`generator_best.pt`](runs/transolver_alpha/53562942/best_slice_tokens/latent_generator_54945515/generator_best.pt)
- [`last.pt`](runs/transolver_alpha/53562942/best_slice_tokens/latent_generator_54945515/last.pt)

Configuration:

```text
train_records = 462
val_records   = 116
token_kind    = out_slice_tokens
layers        = last
num_tokens    = 32
latent_dim    = 128
condition_dim = 4584
batch_size    = 16
scalar_epochs = 200
generator_epochs = 300
```

### Best Scalar Head Metrics

Best scalar head checkpoint:

```text
scalar_head_best.pt
epoch = 165
train_loss = 0.023167
train_mae  = 0.008876
train_rmse = 0.011612
val_loss   = 0.025695
val_mae    = 0.009767
val_rmse   = 0.012213
```

### Best Generator Metrics

Best generator checkpoint by total validation loss:

```text
generator_best.pt
epoch = 21
train_loss = 0.031711
train_latent_mse = 0.027819
train_scalar_mse = 0.026457
train_consistency_mse = 0.012462
train_mae = 0.009347
val_loss = 0.245342
val_latent_mse = 0.233448
val_scalar_mse = 0.068250
val_consistency_mse = 0.050689
val_mae = 0.013609
```

Final generator checkpoint at epoch 300:

```text
last.pt
train_loss = 0.006267
train_latent_mse = 0.004605
train_scalar_mse = 0.012539
train_consistency_mse = 0.004079
train_mae = 0.006547
val_loss = 0.274965
val_latent_mse = 0.257618
val_scalar_mse = 0.094339
val_consistency_mse = 0.079127
val_mae = 0.014458
```

Interpretation:

```text
The scalar head is healthy.
The generator learns the training latents strongly.
Validation peaks early, so generator_best.pt is preferable to last.pt.
```

## 12. Scalar Head Prediction Plot

The user asked for a plot of predicted versus ground-truth global fast ion loss
as predicted by the frozen scalar head.

Generated files:

- [`scalar_head_predicted_vs_ground_truth.png`](runs/transolver_alpha/53562942/best_slice_tokens/latent_generator_54945515/scalar_head_predicted_vs_ground_truth.png)
- [`scalar_head_predicted_vs_ground_truth.csv`](runs/transolver_alpha/53562942/best_slice_tokens/latent_generator_54945515/scalar_head_predicted_vs_ground_truth.csv)
- [`scalar_head_eval_metrics.json`](runs/transolver_alpha/53562942/best_slice_tokens/latent_generator_54945515/scalar_head_eval_metrics.json)

Exact scalar-head evaluation on real exported latents:

```text
Train:
  n = 462
  MAE = 0.007928
  RMSE = 0.010521
  bias = 0.001391
  R2 = 0.981880

Validation:
  n = 116
  MAE = 0.010242
  RMSE = 0.013045
  bias = 0.000553
  R2 = 0.968972

All:
  n = 578
  MAE = 0.008393
  RMSE = 0.011074
  bias = 0.001223
  R2 = 0.979519
```

The assistant clarified that the `0.982` number was train R2. The relevant
validation comparison is:

```text
Original Transolver validation:
  R2  = 0.9401
  MAE = 0.01327
  MSE = 0.0003287

Scalar head on real Transolver latents, validation:
  R2  = 0.9690
  MAE = 0.01024
  MSE = 0.0001702
```

Interpretation:

```text
The trained Transolver learned useful latent features.
The original final readout was constrained to per-node scalar output followed by uniform averaging.
The second-stage scalar head learned a stronger nonlinear readout from the extracted latent features.
```

## 13. Generated Latent vs Ground Truth Visualization

The user requested a visual representation of generated latent versus ground
truth for a couple of examples.

Generated files:

- [`generated_vs_ground_truth_latents_val_examples.png`](runs/transolver_alpha/53562942/best_slice_tokens/latent_generator_54945515/generated_vs_ground_truth_latents_val_examples.png)
- [`generated_vs_ground_truth_latents_val_examples.json`](runs/transolver_alpha/53562942/best_slice_tokens/latent_generator_54945515/generated_vs_ground_truth_latents_val_examples.json)

The plot shows two validation examples:

```text
row 1: low-loss validation case
row 2: high-loss validation case
```

For each row:

```text
ground-truth latent
generated latent, averaged over 64 noise draws
generated minus ground truth
```

The plotted latents were normalized last-layer `out_slice_tokens` with shape:

```text
[32 slices x 128 channels]
```

Metrics from the examples:

```text
Low-loss case: G1600_00529
target = 0.0156
scalar(real latent) = 0.0436
scalar(generated latent) = 0.1008
latent MSE = 0.1460
latent MAE = 0.3552
cosine = 0.9961

High-loss case: G1600_00249
target = 0.3504
scalar(real latent) = 0.3210
scalar(generated latent) = 0.3240
latent MSE = 0.6252
latent MAE = 0.7356
cosine = 0.1187
```

Interpretation at the time:

```text
The low-loss example matches the latent direction well but shifts scalar prediction upward.
The high-loss generated latent gets the scalar prediction close, but the latent structure is not close to the teacher latent.
```

## 14. Discovery: Last-Layer `out_slice_tokens` Collapse Across Slices

The user noticed that the latents in the visualization appeared constant along
the slice dimension.

The assistant inspected saved payloads directly and confirmed:

```text
This is not a plotting artifact.
The last-layer out_slice_tokens are effectively identical across the 32 slice rows.
```

Example standard deviations across slices:

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

Interpretation:

```text
late-layer out_slice_tokens collapsed to a global broadcast-like vector.
```

The current scalar-only task:

```text
full 10-frame field -> one scalar global fast ion loss
```

does not force late attention outputs to preserve slice-resolved structure.
The model can solve the task with global context.

However, the pre-attention `slice_tokens`, especially in earlier layers, retain
substantially more slice variation.

## 15. Would a Temporal Forecasting Objective Help?

The user asked whether training the Transolver to predict time variation, such
as:

```text
initial frame -> next 9 timesteps
```

or:

```text
field at t -> field at t+1, autoregressively
```

would make the eidetic slices in the output tokens vary more.

The answer was: probably yes, but not guaranteed.

Reasoning:

```text
The current objective is scalar-only, so late latents can collapse to global context.
A time-forecasting objective would require spatial and temporal structure, so it would likely preserve more slice-specific information.
```

Caveat:

```text
Even with forecasting, late out_slice_tokens can still become global/broadcast states.
Slice-specific information may live in:
  - pre-attention slice_tokens
  - residual node states
  - earlier-layer out_slice_tokens
  - decoder-side states
```

Recommendation:

```text
Design the latent extraction point intentionally.
Do not assume final out_slice_tokens are the best physics latent.
```

## 16. Key Conclusions From the Thread

1. The original Transolver was trained end to end on scalar global fast ion loss.
2. The 10 temporal profile points were flattened into node feature channels.
3. The final scalar readout was per-node scalar output plus uniform mean pooling.
4. Internal Transolver slice tokens can be captured.
5. A token export script was implemented and smoke-tested.
6. A latent scalar head and conditional generator training script was implemented.
7. The scalar head trained on real extracted latents improved validation R2 from
   about `0.940` to about `0.969`.
8. The conditional generator learned train latents but overfit, with best validation
   behavior early.
9. Last-layer `out_slice_tokens` are not good slice-resolved latents because they
   collapse across slices.
10. Future latent-generator work should use `slice_tokens`, likely all layers, or
    a newly trained latent encoder/dynamics objective.

## 17. Main Artifacts

### Token Export

- [`alpha_analysis/ai/export_transolver_slice_tokens.py`](alpha_analysis/ai/export_transolver_slice_tokens.py)
- [`workflow/export_transolver_slice_tokens.sbatch`](workflow/export_transolver_slice_tokens.sbatch)
- [`runs/transolver_alpha/53562942/best_slice_tokens/manifest.jsonl`](runs/transolver_alpha/53562942/best_slice_tokens/manifest.jsonl)
- [`runs/transolver_alpha/53562942/best_slice_tokens/metadata.json`](runs/transolver_alpha/53562942/best_slice_tokens/metadata.json)

### Latent Training

- [`alpha_analysis/ai/train_latent_generator.py`](alpha_analysis/ai/train_latent_generator.py)
- [`workflow/train_latent_generator.sbatch`](workflow/train_latent_generator.sbatch)
- [`workflow/logs/train_latent_generator_54945515.log`](workflow/logs/train_latent_generator_54945515.log)
- [`runs/transolver_alpha/53562942/best_slice_tokens/latent_generator_54945515/config.json`](runs/transolver_alpha/53562942/best_slice_tokens/latent_generator_54945515/config.json)
- [`runs/transolver_alpha/53562942/best_slice_tokens/latent_generator_54945515/metrics.jsonl`](runs/transolver_alpha/53562942/best_slice_tokens/latent_generator_54945515/metrics.jsonl)
- [`runs/transolver_alpha/53562942/best_slice_tokens/latent_generator_54945515/scalar_head_best.pt`](runs/transolver_alpha/53562942/best_slice_tokens/latent_generator_54945515/scalar_head_best.pt)
- [`runs/transolver_alpha/53562942/best_slice_tokens/latent_generator_54945515/generator_best.pt`](runs/transolver_alpha/53562942/best_slice_tokens/latent_generator_54945515/generator_best.pt)

### Plots And Evaluation Files

- [`runs/transolver_alpha/53562942/best_slice_tokens/latent_generator_54945515/scalar_head_predicted_vs_ground_truth.png`](runs/transolver_alpha/53562942/best_slice_tokens/latent_generator_54945515/scalar_head_predicted_vs_ground_truth.png)
- [`runs/transolver_alpha/53562942/best_slice_tokens/latent_generator_54945515/scalar_head_predicted_vs_ground_truth.csv`](runs/transolver_alpha/53562942/best_slice_tokens/latent_generator_54945515/scalar_head_predicted_vs_ground_truth.csv)
- [`runs/transolver_alpha/53562942/best_slice_tokens/latent_generator_54945515/scalar_head_eval_metrics.json`](runs/transolver_alpha/53562942/best_slice_tokens/latent_generator_54945515/scalar_head_eval_metrics.json)
- [`runs/transolver_alpha/53562942/best_slice_tokens/latent_generator_54945515/generated_vs_ground_truth_latents_val_examples.png`](runs/transolver_alpha/53562942/best_slice_tokens/latent_generator_54945515/generated_vs_ground_truth_latents_val_examples.png)
- [`runs/transolver_alpha/53562942/best_slice_tokens/latent_generator_54945515/generated_vs_ground_truth_latents_val_examples.json`](runs/transolver_alpha/53562942/best_slice_tokens/latent_generator_54945515/generated_vs_ground_truth_latents_val_examples.json)

