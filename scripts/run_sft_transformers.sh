#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${REPO_ROOT}/.venv-cpt/bin/python"
MODEL_PATH="${REPO_ROOT}/checkpoints/transformers/current"
OUTPUT_DIR=""

if (( $# > 0 )) && [[ "$1" != -* ]]; then MODEL_PATH="$1"; shift; fi
if (( $# > 0 )) && [[ "$1" != -* ]]; then OUTPUT_DIR="$1"; shift; fi
SMOKE=false
for arg in "$@"; do [[ "$arg" == "--smoke-test" ]] && SMOKE=true; done
if [[ -z "${OUTPUT_DIR}" ]]; then
  if [[ "${SMOKE}" == true ]]; then
    OUTPUT_DIR="${REPO_ROOT}/checkpoints/sft/transformers_cpt/smoke_test"
  else
    OUTPUT_DIR="${REPO_ROOT}/checkpoints/sft/transformers_cpt/current"
  fi
fi
[[ -x "${PYTHON_BIN}" ]] || { echo "Python not found: ${PYTHON_BIN}" >&2; exit 1; }
[[ -d "${MODEL_PATH}" ]] || { echo "CPT checkpoint not found: ${MODEL_PATH}" >&2; exit 1; }

exec "${PYTHON_BIN}" "${REPO_ROOT}/cpt_training/scripts/train_sft.py" \
  --model-path "${MODEL_PATH}" \
  --sft-dataset "${REPO_ROOT}/ai_coding_agent_cpt_data/SFT_Dataset/sft_general_train.jsonl" \
  --output-dir "${OUTPUT_DIR}" \
  --epochs 3 --learning-rate 2e-4 --batch-size 4 --grad-accum 4 --seed 42 \
  "$@"
