#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DATASET="${REPO_ROOT}/ai_coding_agent_cpt_data/SFT_Dataset/sft_general_complete_train.jsonl"
OUTPUT_ROOT="${REPO_ROOT}/checkpoints/sft_complete"

[[ -f "${DATASET}" ]] || { echo "Complete SFT dataset not found: ${DATASET}" >&2; exit 1; }

exec "${SCRIPT_DIR}/run_sft_all.sh" \
  --output-root "${OUTPUT_ROOT}" \
  --sft-dataset "${DATASET}" \
  --prompt-style complete \
  "$@"
