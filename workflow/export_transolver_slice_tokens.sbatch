#!/bin/bash
#SBATCH -A m5300
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH -t 00:30:00
#SBATCH -J export-transolver-tokens
#SBATCH -o /global/cfs/projectdirs/m5300/rchurchi/alpha_analysis/workflow/logs/export_transolver_slice_tokens_%j.log
#SBATCH -e /global/cfs/projectdirs/m5300/rchurchi/alpha_analysis/workflow/logs/export_transolver_slice_tokens_%j.log

set -euo pipefail

REPO_ROOT=/global/cfs/projectdirs/m5300/rchurchi/alpha_analysis
RUN_DIR="${RUN_DIR:-${REPO_ROOT}/runs/transolver_alpha/53562942}"
CHECKPOINT="${CHECKPOINT:-best.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-${RUN_DIR}/best_slice_tokens}"

mkdir -p "${REPO_ROOT}/workflow/logs" "${OUTPUT_DIR}"
cd "${REPO_ROOT}"

source "${REPO_ROOT}/modules"

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID

args=(
  --run-dir "${RUN_DIR}"
  --checkpoint "${CHECKPOINT}"
  --output-dir "${OUTPUT_DIR}"
  --device cuda
)

if [[ "${OVERWRITE:-false}" == "true" ]]; then
  args+=(--overwrite)
fi

if [[ "${NEW_ONLY:-false}" == "true" ]]; then
  args+=(--new-only)
fi

if [[ -n "${NEW_SPLIT_NAME:-}" ]]; then
  args+=(--new-split-name "${NEW_SPLIT_NAME}")
fi

if [[ "${FULL_MESH:-false}" == "true" ]]; then
  args+=(--full-mesh)
elif [[ -n "${MAX_NODES:-}" ]]; then
  args+=(--max-nodes "${MAX_NODES}")
fi

if [[ -n "${TOKEN_DTYPE:-}" ]]; then
  args+=(--token-dtype "${TOKEN_DTYPE}")
fi

srun python -u -m alpha_analysis.ai.export_transolver_slice_tokens "${args[@]}" "$@"
