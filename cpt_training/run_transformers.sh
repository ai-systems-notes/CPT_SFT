#!/usr/bin/env bash
set -euo pipefail

CPT_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CPT_PYTHON="${CPT_REPO_ROOT}/.venv-cpt/bin/python"
CPT_MODE="${1:-smoke}"
shift || true
CPT_RESULT_DIR=""
CPT_CHECKPOINT_DIR=""

while (( $# > 0 )); do
  case "$1" in
    --output-dir)
      [[ $# -ge 2 ]] || { echo "--output-dir requires PATH" >&2; exit 2; }
      CPT_RESULT_DIR="$(realpath -m "$2")"
      shift 2
      ;;
    --checkpoint-dir)
      [[ $# -ge 2 ]] || { echo "--checkpoint-dir requires PATH" >&2; exit 2; }
      CPT_CHECKPOINT_DIR="$(realpath -m "$2")"
      shift 2
      ;;
    -h|--help)
      echo "Usage: $0 {smoke|pilot|full} [--output-dir PATH] [--checkpoint-dir PATH]"
      exit 0
      ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ "${CPT_MODE}" != "smoke" && "${CPT_MODE}" != "pilot" && "${CPT_MODE}" != "full" ]]; then
  echo "Usage: $0 {smoke|pilot|full} [options]" >&2
  exit 2
fi
if [[ ! -x "${CPT_PYTHON}" ]]; then
  echo "Python environment not found: ${CPT_PYTHON}" >&2
  exit 1
fi

export HF_HOME="${CPT_REPO_ROOT}/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false
CPT_RESULT_DIR="${CPT_RESULT_DIR:-${CPT_REPO_ROOT}/cpt_training/results/current/transformers_${CPT_MODE}}"
CPT_CHECKPOINT_DIR="${CPT_CHECKPOINT_DIR:-${CPT_REPO_ROOT}/checkpoints/transformers/current}"
mkdir -p "${CPT_RESULT_DIR}"

CPT_ARGS=(
  "${CPT_REPO_ROOT}/cpt_training/scripts/train_transformers_cpt.py"
  --config "${CPT_REPO_ROOT}/cpt_training/configs/qwen3_0.6b_seq1024.json"
  --mode "${CPT_MODE}"
  --output "${CPT_RESULT_DIR}/metrics.json"
)
if [[ "${CPT_MODE}" == "full" ]]; then
  CPT_ARGS+=(--checkpoint-dir "${CPT_CHECKPOINT_DIR}")
fi

echo "Mode: ${CPT_MODE}"
echo "Output: ${CPT_RESULT_DIR}"
"${CPT_PYTHON}" "${CPT_ARGS[@]}" 2>&1 | tee "${CPT_RESULT_DIR}/run.log"
