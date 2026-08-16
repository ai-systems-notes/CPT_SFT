#!/usr/bin/env bash
set -euo pipefail

CPT_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CPT_VENV="${CPT_REPO_ROOT}/.venv-unsloth"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required: https://docs.astral.sh/uv/" >&2
  exit 1
fi
if [[ ! -x "${CPT_VENV}/bin/python" ]]; then
  uv venv --python 3.12 "${CPT_VENV}"
fi

uv pip sync --python "${CPT_VENV}/bin/python" \
  "${CPT_REPO_ROOT}/cpt_training/requirements-unsloth.txt"

"${CPT_VENV}/bin/python" -c \
  'import unsloth, torch, transformers; print("torch", torch.__version__); print("transformers", transformers.__version__); print("unsloth", unsloth.__version__)'
