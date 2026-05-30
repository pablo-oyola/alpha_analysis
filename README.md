# alpha_analysis

Utilities to analyze alpha particle loss simulations, based on ASCOT5, DESC/VMEC equilibria, and the AFSI library.

This repository provides:

- Analysis of alpha losses in stellarators.
- Poincaré plot generation and field resolution studies.
- Workflows intended for later integration into ML/AI pipelines.

## Installation

ASCOT is mandatory. You must either already have a working ASCOT (a5py + compiled libascot) on your PYTHONPATH, or install it with the helper script before installing `alpha_analysis`.

### Fast path

```bash
bash ./tools/install_ascot.sh
pip install -e .
```

### Transolver++ training

The Transolver++ model code is kept out of git under `no_sync/` and installed
into the active Python environment with:

```bash
bash ./tools/install_transolver_plus.sh
pip install -e .
```

On NERSC, load this repo's module setup first so the `alpha_analysis` conda env
is active:

```bash
source ./modules
bash ./tools/install_transolver_plus.sh
pip install -e .
```

Run a one-batch smoke test:

```bash
train-alpha-transolver \
  --results-root /global/cfs/cdirs/m5300/results/G1600 \
  --max-samples 2 \
  --max-nodes 1024 \
  --dry-run
```

Start a small training run:

```bash
train-alpha-transolver \
  --results-root /global/cfs/cdirs/m5300/results/G1600 \
  --save-dir runs/transolver_alpha \
  --epochs 20 \
  --batch-size 1 \
  --max-nodes 16384
```

### Supplying your own ASCOT

If you already have ASCOT built:

```bash
export PYTHONPATH=/path/to/existing/ascot5:$PYTHONPATH
pip install -e .
```

### Customizing clone/build

Environment variables:

```bash
export ASCOT_REPO=https://github.com/ascot4fusion/ascot5.git  # override repository
export ASCOT_BRANCH=main                                     # checkout a ref
export ASCOT_DEST=/custom/ascot5-src                         # clone destination
export CC=clang                                              # compiler for libascot
bash ./tools/install_ascot.sh
pip install -e .
```

Skip the helper entirely if ASCOT is already installed:

```bash
export ASCOT_SKIP=true
bash ./tools/install_ascot.sh
pip install -e .
```

## Mandatory dependency behavior

`a5py` is not installed automatically by the current packaging metadata. The supported path is:

```bash
bash ./tools/install_ascot.sh
pip install -e .
```

If the ASCOT build picks the wrong compiler wrapper and produces a non-shared
`libascot.so`, rerun the helper with an explicit compiler, for example
`CC=gcc bash ./tools/install_ascot.sh`.
