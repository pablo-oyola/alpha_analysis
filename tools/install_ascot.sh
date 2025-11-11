#!/usr/bin/env bash
set -euo pipefail

# install_ascot.sh
# Clone and install ASCOT5 Python interface (a5py) from a Git repository.
# Usage:
#   ./tools/install_ascot.sh <git-url> [branch]
# Example:
#   ./tools/install_ascot.sh https://github.com/ASCOT5/ascot5.git main

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <git-url> [branch]" >&2
  exit 1
fi

REPO_URL="$1"
BRANCH="${2:-}"

WORKDIR="$(cd "$(dirname "$0")/.." && pwd)"
TARGET_DIR="${WORKDIR}/no_sync/ascot5-src"
mkdir -p "${TARGET_DIR}"

echo "Cloning ${REPO_URL} into ${TARGET_DIR}..."
if [[ -d "${TARGET_DIR}/.git" ]]; then
  echo "Existing checkout found; fetching updates..."
  git -C "${TARGET_DIR}" remote set-url origin "${REPO_URL}"
  git -C "${TARGET_DIR}" fetch --all --tags --prune
else
  git clone "${REPO_URL}" "${TARGET_DIR}"
fi

if [[ -n "${BRANCH}" ]]; then
  echo "Checking out branch ${BRANCH}..."
  git -C "${TARGET_DIR}" checkout "${BRANCH}"
  git -C "${TARGET_DIR}" pull --ff-only || true
fi

echo "Installing a5py in editable mode..."
python -m pip install -U pip setuptools wheel
python -m pip install -e "${TARGET_DIR}"

echo "ASCOT5 (a5py) installation completed."
