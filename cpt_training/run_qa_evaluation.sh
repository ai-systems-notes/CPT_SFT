#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
export HF_HOME="${REPO_ROOT}/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  echo "Usage: $0 [smoke|full] [--model-set 3|6] [--output-dir PATH] [checkpoint options]"
  exit 0
fi
MODE="${1:-full}"
shift || true
MODEL_SET=6
OUTPUT_DIR=""
TRANSFORMERS_CPT=""
UNSLOTH_CPT=""
BASE_SFT=""
TRANSFORMERS_CPT_SFT=""
UNSLOTH_CPT_SFT=""
MAX_NEW_TOKENS=""
BATCH_SIZE=2

while (( $# > 0 )); do
  case "$1" in
    --model-set) [[ $# -ge 2 ]] || exit 2; MODEL_SET="$2"; shift 2 ;;
    --output-dir) [[ $# -ge 2 ]] || exit 2; OUTPUT_DIR="$(realpath -m "$2")"; shift 2 ;;
    --transformers-checkpoint) [[ $# -ge 2 ]] || exit 2; TRANSFORMERS_CPT="$(realpath -m "$2")"; shift 2 ;;
    --unsloth-checkpoint) [[ $# -ge 2 ]] || exit 2; UNSLOTH_CPT="$(realpath -m "$2")"; shift 2 ;;
    --base-sft-checkpoint) [[ $# -ge 2 ]] || exit 2; BASE_SFT="$(realpath -m "$2")"; shift 2 ;;
    --transformers-cpt-sft-checkpoint) [[ $# -ge 2 ]] || exit 2; TRANSFORMERS_CPT_SFT="$(realpath -m "$2")"; shift 2 ;;
    --unsloth-cpt-sft-checkpoint) [[ $# -ge 2 ]] || exit 2; UNSLOTH_CPT_SFT="$(realpath -m "$2")"; shift 2 ;;
    --max-new-tokens) [[ $# -ge 2 ]] || exit 2; MAX_NEW_TOKENS="$2"; shift 2 ;;
    --batch-size) [[ $# -ge 2 ]] || exit 2; BATCH_SIZE="$2"; shift 2 ;;
    -h|--help)
      echo "Usage: $0 [smoke|full] [--model-set 3|6] [--output-dir PATH] [checkpoint options]"
      exit 0
      ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

[[ "${MODE}" == "smoke" || "${MODE}" == "full" ]] || { echo "Mode must be smoke or full" >&2; exit 2; }
[[ "${MODEL_SET}" == "3" || "${MODEL_SET}" == "6" ]] || { echo "--model-set must be 3 or 6" >&2; exit 2; }

OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/cpt_training/results/current/qa}"
TRANSFORMERS_CPT="${TRANSFORMERS_CPT:-${REPO_ROOT}/checkpoints/transformers/current}"
UNSLOTH_CPT="${UNSLOTH_CPT:-${REPO_ROOT}/checkpoints/unsloth/current}"
BASE_SFT="${BASE_SFT:-${REPO_ROOT}/checkpoints/sft/base/current}"
TRANSFORMERS_CPT_SFT="${TRANSFORMERS_CPT_SFT:-${REPO_ROOT}/checkpoints/sft/transformers_cpt/current}"
UNSLOTH_CPT_SFT="${UNSLOTH_CPT_SFT:-${REPO_ROOT}/checkpoints/sft/unsloth_cpt/current}"
LIMIT_ARGS=()
if [[ "${MODE}" == "smoke" ]]; then
  MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32}"
  LIMIT_ARGS=(--limit 3)
else
  MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
fi

REQUIRED=(
  "${REPO_ROOT}/.venv-cpt/bin/python"
  "${TRANSFORMERS_CPT}"
  "${UNSLOTH_CPT}"
)
if [[ "${MODEL_SET}" == "6" ]]; then
  REQUIRED+=("${BASE_SFT}" "${TRANSFORMERS_CPT_SFT}" "${UNSLOTH_CPT_SFT}")
fi
for path in "${REQUIRED[@]}"; do
  [[ -e "${path}" ]] || { echo "Required path not found: ${path}" >&2; exit 1; }
done

mkdir -p "${OUTPUT_DIR}"
rm -f \
  "${OUTPUT_DIR}/answers_combined.jsonl" \
  "${OUTPUT_DIR}/answers_base.jsonl" \
  "${OUTPUT_DIR}/answers_transformers_cpt.jsonl" \
  "${OUTPUT_DIR}/answers_unsloth_cpt.jsonl" \
  "${OUTPUT_DIR}/answers_base_sft.jsonl" \
  "${OUTPUT_DIR}/answers_transformers_cpt_sft.jsonl" \
  "${OUTPUT_DIR}/answers_unsloth_cpt_sft.jsonl" \
  "${OUTPUT_DIR}/metadata.json" \
  "${OUTPUT_DIR}/gemini_judge_input_blind.jsonl" \
  "${OUTPUT_DIR}/gemini_candidate_map.jsonl" \
  "${OUTPUT_DIR}/gemini_judge_metadata.json" \
  "${OUTPUT_DIR}/gemini_judge_results.jsonl" \
  "${OUTPUT_DIR}/gemini_judge_summary.md" \
  "${OUTPUT_DIR}/gemini_judge_results.errors.jsonl" \
  "${OUTPUT_DIR}/gemini_judge_api_events.jsonl"

"${REPO_ROOT}/.venv-cpt/bin/python" cpt_training/scripts/prepare_qa_generation.py \
  --input ai_coding_agent_cpt_data/QA_Dataset/eval_qa_combined.jsonl \
  --fixes cpt_training/configs/qa_generation_fixes.json \
  --output "${OUTPUT_DIR}/qa_input.jsonl" \
  --metadata "${OUTPUT_DIR}/qa_input_metadata.json"

MODEL_ARGS=(
  --model base=Qwen/Qwen3-0.6B-Base
  --model "transformers_cpt=${TRANSFORMERS_CPT}"
  --model "unsloth_cpt=${UNSLOTH_CPT}"
)
if [[ "${MODEL_SET}" == "6" ]]; then
  MODEL_ARGS+=(
    --model "base_sft=${BASE_SFT}"
    --model "transformers_cpt_sft=${TRANSFORMERS_CPT_SFT}"
    --model "unsloth_cpt_sft=${UNSLOTH_CPT_SFT}"
  )
fi

"${REPO_ROOT}/.venv-cpt/bin/python" cpt_training/scripts/generate_qa_answers.py \
  --dataset "${OUTPUT_DIR}/qa_input.jsonl" \
  "${MODEL_ARGS[@]}" \
  --require-model-count "${MODEL_SET}" \
  --revision da87bfb608c14b7cf20ba1ce41287e8de496c0cd \
  --batch-size "${BATCH_SIZE}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  "${LIMIT_ARGS[@]}" \
  --output-dir "${OUTPUT_DIR}" 2>&1 | tee "${OUTPUT_DIR}/run.log"

"${REPO_ROOT}/.venv-cpt/bin/python" cpt_training/scripts/evaluate_qa.py \
  --result-dir "${OUTPUT_DIR}"

"${REPO_ROOT}/.venv-cpt/bin/python" cpt_training/scripts/prepare_gemini_judge_input.py \
  --input "${OUTPUT_DIR}/answers_combined.jsonl" \
  --judge-output "${OUTPUT_DIR}/gemini_judge_input_blind.jsonl" \
  --map-output "${OUTPUT_DIR}/gemini_candidate_map.jsonl" \
  --metadata-output "${OUTPUT_DIR}/gemini_judge_metadata.json"

echo "Result: ${OUTPUT_DIR}"
