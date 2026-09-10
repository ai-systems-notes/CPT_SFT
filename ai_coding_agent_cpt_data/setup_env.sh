#!/usr/bin/env bash
set -euo pipefail

CPT_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CPT_VENV="${CPT_REPO_ROOT}/.venv-data"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required: https://docs.astral.sh/uv/" >&2
  exit 1
fi
if [[ ! -x "${CPT_VENV}/bin/python" ]]; then
  uv venv --python 3.12 "${CPT_VENV}"
fi

uv pip install --python "${CPT_VENV}/bin/python" \
  -r "${CPT_REPO_ROOT}/ai_coding_agent_cpt_data/requirements.txt"

"${CPT_VENV}/bin/python" -c \
  'import bs4, requests, trafilatura; print("data collection environment: ok")'
