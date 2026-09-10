#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${REPO_ROOT}/.venv-cpt/bin/python"
OUTPUT_DIR="${REPO_ROOT}/cpt_training/results/current/qa_complete_sft"
MAX_NEW_TOKENS=256
BATCH_SIZE=2

while (( $# > 0 )); do
  case "$1" in
    --output-dir)
      [[ $# -ge 2 ]] || { echo "--output-dir requires PATH" >&2; exit 2; }
      OUTPUT_DIR="$(realpath -m "$2")"
      shift 2
      ;;
    --max-new-tokens)
      [[ $# -ge 2 ]] || { echo "--max-new-tokens requires N" >&2; exit 2; }
      MAX_NEW_TOKENS="$2"
      shift 2
      ;;
    --batch-size)
      [[ $# -ge 2 ]] || { echo "--batch-size requires N" >&2; exit 2; }
      BATCH_SIZE="$2"
      shift 2
      ;;
    -h|--help)
      echo "Usage: $0 [--output-dir PATH] [--max-new-tokens N] [--batch-size N]"
      exit 0
      ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

REQUIRED=(
  "${PYTHON_BIN}"
  "${REPO_ROOT}/checkpoints/sft_complete/base/current"
  "${REPO_ROOT}/checkpoints/sft_complete/transformers_cpt/current"
  "${REPO_ROOT}/checkpoints/sft_complete/unsloth_cpt/current"
)
for path in "${REQUIRED[@]}"; do
  [[ -e "${path}" ]] || { echo "Required path not found: ${path}" >&2; exit 1; }
done

mkdir -p "${OUTPUT_DIR}"
export HF_HOME="${REPO_ROOT}/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false

"${PYTHON_BIN}" "${REPO_ROOT}/cpt_training/scripts/prepare_qa_generation.py" \
  --input "${REPO_ROOT}/ai_coding_agent_cpt_data/QA_Dataset/eval_qa_combined.jsonl" \
  --fixes "${REPO_ROOT}/cpt_training/configs/qa_generation_fixes.json" \
  --output "${OUTPUT_DIR}/qa_input.jsonl" \
  --metadata "${OUTPUT_DIR}/qa_input_metadata.json"

"${PYTHON_BIN}" "${REPO_ROOT}/cpt_training/scripts/generate_qa_answers.py" \
  --dataset "${OUTPUT_DIR}/qa_input.jsonl" \
  --model "base_complete_sft=${REPO_ROOT}/checkpoints/sft_complete/base/current" \
  --model "transformers_cpt_complete_sft=${REPO_ROOT}/checkpoints/sft_complete/transformers_cpt/current" \
  --model "unsloth_cpt_complete_sft=${REPO_ROOT}/checkpoints/sft_complete/unsloth_cpt/current" \
  --require-model-count 3 \
  --prompt-style complete \
  --batch-size "${BATCH_SIZE}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --output-dir "${OUTPUT_DIR}" 2>&1 | tee "${OUTPUT_DIR}/run.log"

"${PYTHON_BIN}" "${REPO_ROOT}/cpt_training/scripts/evaluate_qa.py" \
  --result-dir "${OUTPUT_DIR}"

"${PYTHON_BIN}" "${REPO_ROOT}/cpt_training/scripts/audit_qa_structure.py" \
  --qa "${OUTPUT_DIR}/qa_input.jsonl" \
  --answers "${OUTPUT_DIR}/answers_combined.jsonl" \
  --output "${OUTPUT_DIR}/qa_structure_audit.json"

echo "Result: ${OUTPUT_DIR}"
