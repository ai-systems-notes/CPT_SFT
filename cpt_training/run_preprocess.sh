#!/usr/bin/env bash
set -euo pipefail

CPT_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CPT_PYTHON="${CPT_REPO_ROOT}/.venv-cpt/bin/python"
CPT_PREPROCESS_SCRIPT="${CPT_REPO_ROOT}/ai_coding_agent_cpt_data/scripts/preprocess_cpt_data.py"

if [[ ! -x "${CPT_PYTHON}" ]]; then
  echo "Python environment not found: ${CPT_PYTHON}" >&2
  exit 1
fi

export HF_HOME="${CPT_REPO_ROOT}/.cache/huggingface"

"${CPT_PYTHON}" "${CPT_PREPROCESS_SCRIPT}" \
  --input "${CPT_REPO_ROOT}/ai_coding_agent_cpt_data/data/scraped_documents.jsonl" \
  --qa-file "${CPT_REPO_ROOT}/ai_coding_agent_cpt_data/QA_Dataset/eval_qa_combined.jsonl" \
  --output-dir "${CPT_REPO_ROOT}/artifacts/datasets/cpt_v1" \
  --public-stats-output "${CPT_REPO_ROOT}/cpt_training/results/dataset_cpt_v1_stats.json" \
  --model "Qwen/Qwen3-0.6B-Base" \
  --sequence-length 1024 \
  --test-ratio 0.25 \
  --min-chars 800 \
  --seed 20260815 \
  --allow-download \
  --overwrite
