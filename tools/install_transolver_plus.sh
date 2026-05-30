#!/usr/bin/env bash
# install_transolver_plus.sh
# Clone and install the Transolver++ model code into the active Python environment.
# Usage:
#   bash ./tools/install_transolver_plus.sh

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  echo "Do not source this script." >&2
  echo "Run it instead:" >&2
  echo "  bash ${BASH_SOURCE[0]}" >&2
  return 1
fi

set -euo pipefail

WORKDIR="$(cd "$(dirname "$0")/.." && pwd)"
REPO_URL="${TRANSOLVER_PLUS_REPO:-https://github.com/thuml/Transolver_plus.git}"
BRANCH="${TRANSOLVER_PLUS_BRANCH:-main}"
TARGET_DIR="${TRANSOLVER_PLUS_DEST:-${WORKDIR}/no_sync/transolver_plus-src}"
PYTHON_BIN="${PYTHON_BIN:-python}"
TORCHVISION_SPEC="${TRANSOLVER_PLUS_TORCHVISION_SPEC:-torchvision==0.25.0}"

if [[ $# -gt 0 ]]; then
  echo "Unexpected positional arguments: $*" >&2
  echo "Use TRANSOLVER_PLUS_REPO, TRANSOLVER_PLUS_BRANCH, TRANSOLVER_PLUS_DEST," >&2
  echo "TRANSOLVER_PLUS_TORCHVISION_SPEC, or PYTHON_BIN to override defaults." >&2
  exit 1
fi

if [[ "${TRANSOLVER_PLUS_SKIP:-false}" == "true" ]]; then
  echo "Skipping Transolver++ install because TRANSOLVER_PLUS_SKIP=true."
  exit 0
fi

mkdir -p "$(dirname "${TARGET_DIR}")"

echo "Cloning ${REPO_URL} into ${TARGET_DIR}..."
if [[ -d "${TARGET_DIR}/.git" ]]; then
  echo "Existing checkout found; fetching updates..."
  git -C "${TARGET_DIR}" remote set-url origin "${REPO_URL}"
  git -C "${TARGET_DIR}" fetch --all --tags --prune
else
  git clone "${REPO_URL}" "${TARGET_DIR}"
fi

echo "Checking out ${BRANCH}..."
git -C "${TARGET_DIR}" checkout "${BRANCH}"
git -C "${TARGET_DIR}" pull --ff-only || true

if [[ ! -f "${TARGET_DIR}/pyproject.toml" ]] \
  || grep -q "transolver-plus-upstream" "${TARGET_DIR}/pyproject.toml"; then
  echo "Writing editable-install packaging shim..."
  cat > "${TARGET_DIR}/pyproject.toml" <<'PYPROJECT'
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "transolver-plus-upstream"
version = "0.0.0"
description = "Editable install shim for thuml/Transolver_plus."
requires-python = ">=3.10"
dependencies = [
]

[tool.setuptools.packages.find]
where = ["."]
include = ["models*", "dataset*"]
namespaces = true
PYPROJECT
fi

echo "Installing Transolver++ runtime dependencies and editable checkout..."
"${PYTHON_BIN}" -m pip install -U pip setuptools wheel
"${PYTHON_BIN}" -m pip install einops
"${PYTHON_BIN}" -m pip install huggingface_hub safetensors
"${PYTHON_BIN}" -m pip install --no-deps timm
if ! "${PYTHON_BIN}" - <<'PY' >/dev/null 2>&1
import torchvision
PY
then
  "${PYTHON_BIN}" -m pip install --no-deps "${TORCHVISION_SPEC}"
fi
"${PYTHON_BIN}" -m pip install --no-deps -e "${TARGET_DIR}"

echo "Verifying Transolver++ import..."
"${PYTHON_BIN}" - <<'PY'
from models.Transolver_plus import Model

model = Model(n_layers=1, n_hidden=32, n_head=4, space_dim=5, fun_dim=0, out_dim=1)
print(f"Imported Transolver++ Model with {sum(p.numel() for p in model.parameters())} parameters.")
PY

echo "Transolver++ installation completed."
