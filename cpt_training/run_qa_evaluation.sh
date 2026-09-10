#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
export HF_HOME="${REPO_ROOT}/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false

MODE="${1:-full}"
if [[ "${MODE}" == "-h" || "${MODE}" == "--help" ]]; then
  echo "Usage: $0 [smoke|full] [--model-set 3|6|12] [--dataset PATH] [--output-dir PATH] [--max-new-tokens N] [--batch-size N]"
  exit 0
fi
shift || true

MODEL_SET=12
DATASET="${REPO_ROOT}/ai_coding_agent_cpt_data/QA_Dataset/eval_qa_combined.jsonl"
OUTPUT_DIR="${REPO_ROOT}/cpt_training/results/current/qa"
MAX_NEW_TOKENS=""
BATCH_SIZE=2

while (( $# > 0 )); do
  case "$1" in
    --model-set) [[ $# -ge 2 ]] || exit 2; MODEL_SET="$2"; shift 2 ;;
    --dataset) [[ $# -ge 2 ]] || exit 2; DATASET="$(realpath -m "$2")"; shift 2 ;;
    --output-dir) [[ $# -ge 2 ]] || exit 2; OUTPUT_DIR="$(realpath -m "$2")"; shift 2 ;;
    --max-new-tokens) [[ $# -ge 2 ]] || exit 2; MAX_NEW_TOKENS="$2"; shift 2 ;;
    --batch-size) [[ $# -ge 2 ]] || exit 2; BATCH_SIZE="$2"; shift 2 ;;
    -h|--help)
      echo "Usage: $0 [smoke|full] [--model-set 3|6|12] [--dataset PATH] [--output-dir PATH] [--max-new-tokens N] [--batch-size N]"
      exit 0
      ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

[[ "${MODE}" == "smoke" || "${MODE}" == "full" ]] || { echo "Mode must be smoke or full" >&2; exit 2; }
[[ "${MODEL_SET}" == "3" || "${MODEL_SET}" == "6" || "${MODEL_SET}" == "12" ]] || { echo "--model-set must be 3, 6, or 12" >&2; exit 2; }

LIMIT_ARGS=()
if [[ "${MODE}" == "smoke" ]]; then
  MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"
  LIMIT_ARGS=(--limit 3)
else
  MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
fi

PYTHON_BIN="${REPO_ROOT}/.venv-cpt/bin/python"
TRANSFORMERS_CPT="${REPO_ROOT}/checkpoints/transformers/current"
UNSLOTH_CPT="${REPO_ROOT}/checkpoints/unsloth/current"
SFT_V1="${REPO_ROOT}/checkpoints/sft"
SFT_V2="${REPO_ROOT}/checkpoints/sft_complete"
SFT_V3="${REPO_ROOT}/checkpoints/sft_v3"

REQUIRED=("${PYTHON_BIN}" "${DATASET}" "${TRANSFORMERS_CPT}" "${UNSLOTH_CPT}")
if [[ "${MODEL_SET}" == "6" || "${MODEL_SET}" == "12" ]]; then
  REQUIRED+=("${SFT_V1}/base/current" "${SFT_V1}/transformers_cpt/current" "${SFT_V1}/unsloth_cpt/current")
fi
if [[ "${MODEL_SET}" == "12" ]]; then
  REQUIRED+=(
    "${SFT_V2}/base/current" "${SFT_V2}/transformers_cpt/current" "${SFT_V2}/unsloth_cpt/current"
    "${SFT_V3}/base/current" "${SFT_V3}/transformers_cpt/current" "${SFT_V3}/unsloth_cpt/current"
  )
fi
for path in "${REQUIRED[@]}"; do
  [[ -e "${path}" ]] || { echo "Required path not found: ${path}" >&2; exit 1; }
done

mkdir -p "${OUTPUT_DIR}"
STALE_FILES=(
  answers_combined.jsonl metadata.json qa_automatic_metrics.json qa_automatic_metrics.md qa_structure_audit.json
  gemini_candidate_map.jsonl gemini_judge_api_events.jsonl gemini_judge_input_blind.jsonl
  gemini_judge_metadata.json gemini_judge_results.jsonl gemini_judge_results.errors.jsonl gemini_judge_summary.md
  answers_base.jsonl answers_transformers_cpt.jsonl answers_unsloth_cpt.jsonl
  answers_base_sft.jsonl answers_transformers_cpt_sft.jsonl answers_unsloth_cpt_sft.jsonl
  answers_base_sft_v1.jsonl answers_transformers_cpt_sft_v1.jsonl answers_unsloth_cpt_sft_v1.jsonl
  answers_base_sft_v2.jsonl answers_transformers_cpt_sft_v2.jsonl answers_unsloth_cpt_sft_v2.jsonl
  answers_base_sft_v3.jsonl answers_transformers_cpt_sft_v3.jsonl answers_unsloth_cpt_sft_v3.jsonl
)
for name in "${STALE_FILES[@]}"; do rm -f "${OUTPUT_DIR}/${name}"; done

"${PYTHON_BIN}" ai_coding_agent_cpt_data/scripts/generate_qa_dataset.py
"${PYTHON_BIN}" cpt_training/scripts/prepare_qa_generation.py   --input "${DATASET}"   --output "${OUTPUT_DIR}/qa_input.jsonl"   --metadata "${OUTPUT_DIR}/qa_input_metadata.json"

MODEL_ARGS=(
  --model base=Qwen/Qwen3-0.6B-Base
  --model "transformers_cpt=${TRANSFORMERS_CPT}"
  --model "unsloth_cpt=${UNSLOTH_CPT}"
)
if [[ "${MODEL_SET}" == "6" || "${MODEL_SET}" == "12" ]]; then
  MODEL_ARGS+=(
    --model "base_sft_v1=${SFT_V1}/base/current"
    --model "transformers_cpt_sft_v1=${SFT_V1}/transformers_cpt/current"
    --model "unsloth_cpt_sft_v1=${SFT_V1}/unsloth_cpt/current"
  )
fi
if [[ "${MODEL_SET}" == "12" ]]; then
  MODEL_ARGS+=(
    --model "base_sft_v2=${SFT_V2}/base/current"
    --model "transformers_cpt_sft_v2=${SFT_V2}/transformers_cpt/current"
    --model "unsloth_cpt_sft_v2=${SFT_V2}/unsloth_cpt/current"
    --model "base_sft_v3=${SFT_V3}/base/current"
    --model "transformers_cpt_sft_v3=${SFT_V3}/transformers_cpt/current"
    --model "unsloth_cpt_sft_v3=${SFT_V3}/unsloth_cpt/current"
  )
fi

"${PYTHON_BIN}" cpt_training/scripts/generate_qa_answers.py   --dataset "${OUTPUT_DIR}/qa_input.jsonl"   "${MODEL_ARGS[@]}"   --require-model-count "${MODEL_SET}"   --prompt-style v3   --revision da87bfb608c14b7cf20ba1ce41287e8de496c0cd   --batch-size "${BATCH_SIZE}"   --max-new-tokens "${MAX_NEW_TOKENS}"   "${LIMIT_ARGS[@]}"   --output-dir "${OUTPUT_DIR}" 2>&1 | tee "${OUTPUT_DIR}/run.log"

"${PYTHON_BIN}" cpt_training/scripts/evaluate_qa.py --result-dir "${OUTPUT_DIR}"
"${PYTHON_BIN}" cpt_training/scripts/audit_qa_structure.py   --qa "${OUTPUT_DIR}/qa_input.jsonl"   --answers "${OUTPUT_DIR}/answers_combined.jsonl"   --output "${OUTPUT_DIR}/qa_structure_audit.json"

echo "Result: ${OUTPUT_DIR}"
