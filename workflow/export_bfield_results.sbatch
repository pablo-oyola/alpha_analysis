#!/bin/bash
#SBATCH -A m5300
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH -t 24:00:00
#SBATCH -J export-bfield
#SBATCH -o /global/cfs/projectdirs/m5300/rchurchi/alpha_analysis/workflow/logs/export_bfield_%j.out
#SBATCH -e /global/cfs/projectdirs/m5300/rchurchi/alpha_analysis/workflow/logs/export_bfield_%j.err

set -euo pipefail

REPO_ROOT=/global/cfs/projectdirs/m5300/rchurchi/alpha_analysis
RESULTS_ROOT=/global/cfs/projectdirs/m5300/results

mkdir -p "${REPO_ROOT}/workflow/logs"
cd "${REPO_ROOT}"

source "${REPO_ROOT}/modules"

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

python "${REPO_ROOT}/workflow/export_bfield_results.py" \
  --results-root "${RESULTS_ROOT}" \
  "$@"
