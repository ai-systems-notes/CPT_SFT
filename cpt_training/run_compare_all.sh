#!/usr/bin/env bash
set -euo pipefail

CPT_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CPT_MODE="${1:-smoke}"
shift || true
CPT_CONFIRM_FULL=false
CPT_RESULT_DIR=""
CPT_CHECKPOINT_ROOT=""

while (( $# > 0 )); do
  case "$1" in
    --confirm-full)
      CPT_CONFIRM_FULL=true
      shift
      ;;
    --output-dir)
      [[ $# -ge 2 ]] || { echo "--output-dir requires PATH" >&2; exit 2; }
      CPT_RESULT_DIR="$(realpath -m "$2")"
      shift 2
      ;;
    --checkpoint-root)
      [[ $# -ge 2 ]] || { echo "--checkpoint-root requires PATH" >&2; exit 2; }
      CPT_CHECKPOINT_ROOT="$(realpath -m "$2")"
      shift 2
      ;;
    -h|--help)
      echo "Usage: $0 {smoke|pilot|full} [--confirm-full] [--output-dir PATH] [--checkpoint-root PATH]"
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ "${CPT_MODE}" != "smoke" && "${CPT_MODE}" != "pilot" && "${CPT_MODE}" != "full" ]]; then
  echo "Usage: $0 {smoke|pilot|full} [--confirm-full] [--output-dir PATH] [--checkpoint-root PATH]" >&2
  exit 2
fi
if [[ "${CPT_MODE}" == "full" && "${CPT_CONFIRM_FULL}" != true ]]; then
  echo "Full training requires: $0 full --confirm-full" >&2
  exit 2
fi

CPT_RESULT_DIR="${CPT_RESULT_DIR:-${CPT_REPO_ROOT}/cpt_training/results/current/compare_${CPT_MODE}}"
CPT_CHECKPOINT_ROOT="${CPT_CHECKPOINT_ROOT:-${CPT_REPO_ROOT}/checkpoints}"
CPT_CONFIG="${CPT_REPO_ROOT}/cpt_training/configs/qwen3_0.6b_seq1024.json"
mkdir -p "${CPT_RESULT_DIR}"

for CPT_REQUIRED in \
  "${CPT_REPO_ROOT}/.venv-cpt/bin/python" \
  "${CPT_REPO_ROOT}/.venv-unsloth/bin/python" \
  "${CPT_REPO_ROOT}/artifacts/datasets/cpt_v1/packed/seq1024/train.pt" \
  "${CPT_REPO_ROOT}/artifacts/datasets/cpt_v1/packed/seq1024/test.pt"; do
  if [[ ! -e "${CPT_REQUIRED}" ]]; then
    echo "Required path not found: ${CPT_REQUIRED}" >&2
    exit 1
  fi
done

export HF_HOME="${CPT_REPO_ROOT}/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false
export UNSLOTH_DISABLE_STATISTICS=1

nvidia-smi --query-gpu=timestamp,name,memory.total,memory.used,driver_version \
  --format=csv,noheader > "${CPT_RESULT_DIR}/gpu_before.csv"

CPT_TRANSFORMERS_ARGS=(
  "${CPT_REPO_ROOT}/cpt_training/scripts/train_transformers_cpt.py"
  --config "${CPT_CONFIG}" --mode "${CPT_MODE}"
  --output "${CPT_RESULT_DIR}/transformers_metrics.json"
)
CPT_UNSLOTH_ARGS=(
  "${CPT_REPO_ROOT}/cpt_training/scripts/train_unsloth_cpt.py"
  --config "${CPT_CONFIG}" --mode "${CPT_MODE}"
  --output "${CPT_RESULT_DIR}/unsloth_metrics.json"
)
if [[ "${CPT_MODE}" == "full" ]]; then
  CPT_TRANSFORMERS_ARGS+=(--checkpoint-dir "${CPT_CHECKPOINT_ROOT}/transformers/current")
  CPT_UNSLOTH_ARGS+=(--checkpoint-dir "${CPT_CHECKPOINT_ROOT}/unsloth/current")
fi

echo "Output: ${CPT_RESULT_DIR}"
echo "[1/3] Transformers/PyTorch: ${CPT_MODE}"
"${CPT_REPO_ROOT}/.venv-cpt/bin/python" "${CPT_TRANSFORMERS_ARGS[@]}" \
  2>&1 | tee "${CPT_RESULT_DIR}/transformers.log"

nvidia-smi --query-gpu=timestamp,name,memory.total,memory.used,driver_version \
  --format=csv,noheader > "${CPT_RESULT_DIR}/gpu_between.csv"

echo "[2/3] Unsloth: ${CPT_MODE}"
"${CPT_REPO_ROOT}/.venv-unsloth/bin/python" "${CPT_UNSLOTH_ARGS[@]}" \
  2>&1 | tee "${CPT_RESULT_DIR}/unsloth.log"

echo "[3/3] Compare metrics"
"${CPT_REPO_ROOT}/.venv-cpt/bin/python" \
  "${CPT_REPO_ROOT}/cpt_training/scripts/compare_results.py" \
  --transformers "${CPT_RESULT_DIR}/transformers_metrics.json" \
  --unsloth "${CPT_RESULT_DIR}/unsloth_metrics.json" \
  --output-dir "${CPT_RESULT_DIR}"

echo "Result: ${CPT_RESULT_DIR}/comparison.md"
