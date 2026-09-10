#!/usr/bin/env bash
set -euo pipefail

CPT_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CPT_RESULT_DIR="${1:-${CPT_REPO_ROOT}/cpt_training/results/current/qa}"

exec "${CPT_REPO_ROOT}/.venv-cpt/bin/python" \
  "${CPT_REPO_ROOT}/ai_coding_agent_cpt_data/scripts/run_gemini_judge.py" \
  --result-dir "${CPT_RESULT_DIR}"
