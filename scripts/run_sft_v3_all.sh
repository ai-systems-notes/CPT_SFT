#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${REPO_ROOT}/.venv-cpt/bin/python"
DATASET="${REPO_ROOT}/ai_coding_agent_cpt_data/SFT_Dataset/sft_v3_train.jsonl"
OUTPUT_ROOT="${REPO_ROOT}/checkpoints/sft_v3"

[[ -x "${PYTHON_BIN}" ]] || { echo "Python not found: ${PYTHON_BIN}" >&2; exit 1; }
[[ -f "${DATASET}" ]] || { echo "SFT v3 dataset not found: ${DATASET}" >&2; exit 1; }

"${PYTHON_BIN}" "${REPO_ROOT}/ai_coding_agent_cpt_data/scripts/verify_sft_v3_dataset.py"

exec "${SCRIPT_DIR}/run_sft_all.sh" \
  --output-root "${OUTPUT_ROOT}" \
  --sft-dataset "${DATASET}" \
  --prompt-style v3 \
  "$@"
