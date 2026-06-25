#!/bin/bash
#SBATCH -A m5300
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH -t 04:00:00
#SBATCH -J train-latent-generator
#SBATCH -o /global/cfs/projectdirs/m5300/rchurchi/alpha_analysis/workflow/logs/train_latent_generator_%j.log
#SBATCH -e /global/cfs/projectdirs/m5300/rchurchi/alpha_analysis/workflow/logs/train_latent_generator_%j.log

set -euo pipefail

REPO_ROOT=/global/cfs/projectdirs/m5300/rchurchi/alpha_analysis
TOKEN_DIR="${TOKEN_DIR:-${REPO_ROOT}/runs/transolver_alpha/53562942/best_slice_tokens}"
OUTPUT_DIR="${OUTPUT_DIR:-${TOKEN_DIR}/latent_generator_${SLURM_JOB_ID}}"

mkdir -p "${REPO_ROOT}/workflow/logs" "${OUTPUT_DIR}"
cd "${REPO_ROOT}"

source "${REPO_ROOT}/modules"

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID

args=(
  --token-dir "${TOKEN_DIR}"
  --output-dir "${OUTPUT_DIR}"
  --device cuda
)

if [[ "${OVERWRITE:-false}" == "true" ]]; then
  args+=(--overwrite)
fi

if [[ -n "${BATCH_SIZE:-}" ]]; then
  args+=(--batch-size "${BATCH_SIZE}")
fi

if [[ -n "${SCALAR_EPOCHS:-}" ]]; then
  args+=(--scalar-epochs "${SCALAR_EPOCHS}")
fi

if [[ -n "${GENERATOR_EPOCHS:-}" ]]; then
  args+=(--generator-epochs "${GENERATOR_EPOCHS}")
fi

srun python -u -m alpha_analysis.ai.train_latent_generator "${args[@]}" "$@"
