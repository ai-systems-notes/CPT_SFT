# LoRA-SFT and six-condition QA generation

Compact artifacts from the completed RTX 4070 run.

## Status

- Three PEFT LoRA-SFT runs: complete.
- Six conditions × 200 QA rows: complete, 1,200 answers.
- Formatting and lexical metrics: complete.
- Full primary-source reference audit: not performed.
- External LLM-as-a-Judge: not performed; no judge score is included.

## Files

- `answers_base.jsonl`: 200 Base-model answers.
- `answers_transformers_cpt.jsonl`: 200 Transformers-CPT answers.
- `answers_unsloth_cpt.jsonl`: 200 Unsloth-CPT answers.
- `answers_base_sft.jsonl`: 200 Base + LoRA-SFT answers.
- `answers_transformers_cpt_sft.jsonl`: 200 Transformers-CPT + LoRA-SFT answers.
- `answers_unsloth_cpt_sft.jsonl`: 200 Unsloth-CPT + LoRA-SFT answers.
- `answers_combined.jsonl`: the same 1,200 answers grouped into 200 QA rows.
- `qa_generation_metadata.json`: model sources, adapter metadata, decoding settings, runtime, and environment.
- `qa_automatic_metrics.json` / `.md`: non-empty rate, answer length, exact match, and keyword recall.
- `qa_structure_audit.json`: exact-question duplication, continuation markers, and pairwise exact-answer matches.
- `sft_*_metrics.json`: loss, runtime, throughput, memory, and parameter counts.
- `sft_*_metadata.json`: source model, dataset hash, environment, prompt, and hyperparameters.

## Interpretation

The metrics show increased short-answer format adherence after LoRA-SFT. They do not establish factual accuracy. `keyword_recall` is substring overlap against provisional keywords.

The QA input contains 200 rows but 83 exact unique question strings. Results are row-weighted and must not be reported as a 200-independent-question benchmark.

Model checkpoints and LoRA adapters are excluded from Git. Use the root README commands to recreate them.
