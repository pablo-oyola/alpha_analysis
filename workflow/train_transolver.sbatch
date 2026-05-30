#!/bin/bash
#SBATCH -A m4505
#SBATCH -C gpu
#SBATCH -q premium
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH -t 04:00:00
#SBATCH -J train-transolver
#SBATCH -o /global/cfs/projectdirs/m5300/rchurchi/alpha_analysis/workflow/logs/train_transolver_%j.log
#SBATCH -e /global/cfs/projectdirs/m5300/rchurchi/alpha_analysis/workflow/logs/train_transolver_%j.log

set -euo pipefail

REPO_ROOT=/global/cfs/projectdirs/m5300/rchurchi/alpha_analysis
RESULTS_ROOT=/global/cfs/cdirs/m5300/results/G1600
SAVE_DIR="${REPO_ROOT}/runs/transolver_alpha/${SLURM_JOB_ID}"

mkdir -p "${REPO_ROOT}/workflow/logs" "${SAVE_DIR}"
cd "${REPO_ROOT}"

source "${REPO_ROOT}/modules"

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID

python - <<'PY'
import torch
from models.Transolver_plus import Model

print(f"torch {torch.__version__}")
print(f"cuda_available {torch.cuda.is_available()}")
print(f"cuda_device_count {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"cuda_device_0 {torch.cuda.get_device_name(0)}")
print(f"transolver_model {Model.__name__}")
PY

srun python -u -m alpha_analysis.ai.train_transolver \
  --results-root "${RESULTS_ROOT}" \
  --save-dir "${SAVE_DIR}" \
  --device cuda \
  --epochs 400
  "$@"
