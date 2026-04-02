#!/usr/bin/env bash
# install_ascot.sh
# Clone and install ASCOT5 Python interface (a5py) from a Git repository.
# Usage:
#   bash ./tools/install_ascot.sh
# Example:
#   bash ./tools/install_ascot.sh

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  echo "Do not source this script." >&2
  echo "Run it instead:" >&2
  echo "  bash ${BASH_SOURCE[0]}" >&2
  return 1
fi

set -euo pipefail

WORKDIR="$(cd "$(dirname "$0")/.." && pwd)"
REPO_URL="${ASCOT_REPO:-https://github.com/ascot4fusion/ascot5.git}"
BRANCH="${ASCOT_BRANCH:-main}"
TARGET_DIR="${ASCOT_DEST:-${WORKDIR}/no_sync/ascot5-src}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CC_BIN="${CC:-gcc}"

if [[ $# -gt 0 ]]; then
  echo "Unexpected positional arguments: $*" >&2
  echo "Use environment variables ASCOT_REPO, ASCOT_BRANCH, or ASCOT_DEST to override defaults." >&2
  exit 1
fi

if [[ "${ASCOT_SKIP:-false}" == "true" ]]; then
  echo "Skipping ASCOT install because ASCOT_SKIP=true."
  exit 0
fi

if "${PYTHON_BIN}" - <<'PY' >/dev/null 2>&1
import a5py
PY
then
  echo "Found existing a5py installation in ${PYTHON_BIN}; skipping ASCOT install."
  exit 0
fi

mkdir -p "${TARGET_DIR}"

echo "Cloning ${REPO_URL} into ${TARGET_DIR}..."
if [[ -d "${TARGET_DIR}/.git" ]]; then
  echo "Existing checkout found; fetching updates..."
  git -C "${TARGET_DIR}" remote set-url origin "${REPO_URL}"
  git -C "${TARGET_DIR}" fetch --all --tags --prune
else
  git clone "${REPO_URL}" "${TARGET_DIR}"
fi

echo "Checking out branch ${BRANCH}..."
git -C "${TARGET_DIR}" checkout "${BRANCH}"
git -C "${TARGET_DIR}" pull --ff-only || true

echo "Installing a5py in editable mode..."
"${PYTHON_BIN}" -m pip install -U pip setuptools wheel
"${PYTHON_BIN}" -m pip install -e "${TARGET_DIR}"

echo "Building libascot with CC=${CC_BIN}..."
make -C "${TARGET_DIR}" clean
make -C "${TARGET_DIR}" libascot CC="${CC_BIN}"

echo "Generating ascot2py.py against the rebuilt libascot..."
make -C "${TARGET_DIR}" ascot2py.py CC="${CC_BIN}"

echo "Verifying that build/libascot.so is a shared library..."
if ! file "${TARGET_DIR}/build/libascot.so" | grep -q "shared object"; then
  echo "ERROR: ${TARGET_DIR}/build/libascot.so was not built as a shared library." >&2
  file "${TARGET_DIR}/build/libascot.so" >&2 || true
  echo "Try rerunning with an explicit compiler, for example:" >&2
  echo "  CC=gcc bash ${BASH_SOURCE[0]}" >&2
  exit 1
fi

echo "ASCOT5 (a5py) installation completed."
