#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUTPUT_ROOT="${REPO_ROOT}/checkpoints/sft"
COMMON_ARGS=()

while (( $# > 0 )); do
  case "$1" in
    --output-root)
      [[ $# -ge 2 ]] || { echo "--output-root requires PATH" >&2; exit 2; }
      OUTPUT_ROOT="$(realpath -m "$2")"
      shift 2
      ;;
    --output-dir|--model-path)
      echo "$1 is ambiguous for run_sft_all.sh; use --output-root or an individual script" >&2
      exit 2
      ;;
    -h|--help)
      echo "Usage: $0 [--output-root PATH] [--smoke-test | --confirm-full] [train_sft options]"
      exit 0
      ;;
    *) COMMON_ARGS+=("$1"); shift ;;
  esac
done

SMOKE=false
for arg in "${COMMON_ARGS[@]}"; do [[ "$arg" == "--smoke-test" ]] && SMOKE=true; done
LEAF="current"
[[ "${SMOKE}" == true ]] && LEAF="smoke_test"

echo "[1/3] Base-SFT"
"${SCRIPT_DIR}/run_sft_base.sh" --output-dir "${OUTPUT_ROOT}/base/${LEAF}" "${COMMON_ARGS[@]}"
echo "[2/3] Transformers-CPT-SFT"
"${SCRIPT_DIR}/run_sft_transformers.sh" --output-dir "${OUTPUT_ROOT}/transformers_cpt/${LEAF}" "${COMMON_ARGS[@]}"
echo "[3/3] Unsloth-CPT-SFT"
"${SCRIPT_DIR}/run_sft_unsloth.sh" --output-dir "${OUTPUT_ROOT}/unsloth_cpt/${LEAF}" "${COMMON_ARGS[@]}"
echo "Completed all three SFT runs under ${OUTPUT_ROOT}"
