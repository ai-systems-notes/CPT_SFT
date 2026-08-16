#!/usr/bin/env bash
set -euo pipefail

CPT_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CPT_VENV="${CPT_REPO_ROOT}/.venv-cpt"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required: https://docs.astral.sh/uv/" >&2
  exit 1
fi
if [[ ! -x "${CPT_VENV}/bin/python" ]]; then
  uv venv --python 3.12 "${CPT_VENV}"
fi

uv pip install --python "${CPT_VENV}/bin/python" \
  torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python "${CPT_VENV}/bin/python" \
  -r "${CPT_REPO_ROOT}/cpt_training/requirements.txt"

"${CPT_VENV}/bin/python" -c \
  'import torch, transformers; print("torch", torch.__version__); print("transformers", transformers.__version__)'
