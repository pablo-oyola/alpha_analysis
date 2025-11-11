# alpha_analysis

Utilities to analyze alpha particle loss simulations, based on ASCOT5, DESC/VMEC equilibria, and the AFSI library.

This repository provides:

- Analysis of alpha losses in stellarators.
- Poincaré plot generation and field resolution studies.
- Workflows intended for later integration into ML/AI pipelines.

## Installation

ASCOT is mandatory. You must either already have a working ASCOT (a5py + compiled libascot) on your PYTHONPATH, or allow the installer to fetch and build it automatically. By default, `setup.py` will attempt to clone and build if `a5py` is missing.

### Fast path (automatic fetch & build)

```bash
pip install -e .  # will clone ascot5 into ./no_sync/ascot5-src and build libascot if a5py absent
```

### Supplying your own ASCOT

If you already have ASCOT built:

```bash
export PYTHONPATH=/path/to/existing/ascot5:$PYTHONPATH
pip install -e .  # installer detects a5py and skips clone/build
```

### Customizing clone/build

Environment variables:

```bash
export ASCOT_REPO=https://github.com/ascot4fusion/ascot5.git  # override repository
export ASCOT_BRANCH=main                                     # checkout a ref
export ASCOT_DEST=/custom/ascot5-src                         # clone destination
export CC=clang                                              # compiler for libascot
pip install -e .
```

Skip automatic installation (only if you guarantee availability):

```bash
export ASCOT_SKIP=true
pip install -e .
```

### Console script (post-install)

After installing `alpha_analysis`, you can (re)install ASCOT via helper:

```bash
alpha-analysis-install-ascot --repo https://github.com/ascot4fusion/ascot5.git --branch main --dest ./no_sync/ascot5-src --cc gcc
```

## Mandatory dependency behavior

If `a5py` is missing and `ASCOT_SKIP` is not set, the installer clones and builds ASCOT. If build fails, installation aborts with an error. Provide `ASCOT_SKIP=true` only when ASCOT is available already.
