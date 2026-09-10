# Source-grounded AI Coding Agent QA Dataset

This directory contains the canonical QA set used for the Base/CPT/SFT ablation.
The existing filenames are stable and are overwritten by `scripts/generate_qa_dataset.py`.

## Files

- `eval_qa_claude.jsonl`: 35 Claude Code questions.
- `eval_qa_codex.jsonl`: 35 Codex/ChatGPT questions.
- `eval_qa_combined.jsonl`: all 70 questions.

## Evaluation splits

- `sft_seen` (30): the exact 30 domain-specific SFT v3 training items. This split measures direct SFT memorization and must not be presented as held-out performance.
- `heldout` (40): distinct questions and facts from other collected official documentation pages. This split measures whether CPT knowledge remains retrievable after SFT.

The two product categories and two evaluation roles must be reported separately.

## Grounding and validation

Every row contains the official source URL, source-document SHA-256, required keywords, a source-derived evidence excerpt, and an evidence summary. The generator refuses to overwrite the dataset unless:

- all source URLs exist in the collected official corpus;
- all required evidence tokens occur in the claimed source document;
- normalized questions and answers are unique;
- the fixed counts are 30 `sft_seen` and 40 `heldout` rows.

The generator performs structural and lexical source checks. Manual semantic review remains required before making factual benchmark claims.

## Regenerate

```bash
python3 ai_coding_agent_cpt_data/scripts/generate_qa_dataset.py
```

The older 200-row provisional dataset was replaced because it contained only 83 unique question strings and many unverified references.
