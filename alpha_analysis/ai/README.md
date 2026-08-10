# AI for Ascot5

This repository is the starting point for the AI for Ascot5 (https://github.com/ascot4fusion/ascot5) project. This will serve as the central place for project code, notes, and supporting documentation as the work takes shape.

# Current status
- Data loaders for generated dataset
- Helper for sampling `br`, `bphi`, and `bz` from `ascot_results.h5` onto the
  `analysis_results.h5/profiles` grid


## Per-frame Transolver generator

The deterministic generator treats each pressure frame explicitly. The historical G1600 cohort is fixed to the first 578 complete simulations: seed 0 and `--train-fraction 0.8` reproduce the prior 462/116 train/validation split. Folders after that boundary are labeled `later`; they are encoded and predicted but never used to train models or fit normalization statistics.

```bash
train-alpha-frame-autoencoder --results-root /global/cfs/cdirs/m5300/results/G1600 --save-dir runs/frame_autoencoder/run --training-sample-count 578
export-alpha-frame-latents --checkpoint runs/frame_autoencoder/run/best_field.pt --results-root /global/cfs/cdirs/m5300/results/G1600 --output-dir runs/frame_autoencoder/run/latents
train-alpha-latent-dynamics --latent-dir runs/frame_autoencoder/run/latents --save-dir runs/frame_autoencoder/run/dynamics --stage one-step
```

Latent export defaults to the full mesh. Use `--max-nodes` only when necessary; the deterministic indices and limit are saved with each export. Train dynamics in order (`one-step`, `rollout3`, then `full`) and compare validation and later-cohort rollouts against persistence before attempting probabilistic models.
