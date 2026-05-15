#!/bin/bash
#SBATCH -A m5300
#SBATCH -C cpu
#SBATCH -q debug
#SBATCH -N 4
#SBATCH --ntasks-per-node=12
#SBATCH --cpus-per-task=8
#SBATCH -t 00:30:00
#SBATCH -J export-bfield
#SBATCH -o /global/cfs/projectdirs/m5300/rchurchi/alpha_analysis/workflow/logs/export_bfield_%j.log
#SBATCH -e /global/cfs/projectdirs/m5300/rchurchi/alpha_analysis/workflow/logs/export_bfield_%j.log

set -euo pipefail

REPO_ROOT=/global/cfs/projectdirs/m5300/rchurchi/alpha_analysis
RESULTS_ROOT=/global/cfs/projectdirs/m5300/results

mkdir -p "${REPO_ROOT}/workflow/logs"
cd "${REPO_ROOT}"

source "${REPO_ROOT}/modules"

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export JAX_PLATFORMS=cpu
export JAX_PLATFORM_NAME=cpu
export JAX_SKIP_CUDA_CONSTRAINTS_CHECK=1
export CUDA_VISIBLE_DEVICES=""

python - <<'PY'
import mpi4py
print(f"mpi4py {mpi4py.__version__}")
PY

srun python "${REPO_ROOT}/workflow/export_bfield_results.py" \
  --mpi \
  --results-root "${RESULTS_ROOT}" \
  "$@"
