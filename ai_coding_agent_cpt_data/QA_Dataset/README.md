# Provisional AI Coding Agent QA Generation Set

This directory contains the 200 QA rows used for the six-condition generation run in this repository.

## Files

- `eval_qa_claude.jsonl`: 100 Claude/Anthropic-related rows.
- `eval_qa_codex.jsonl`: 100 Codex/OpenAI-related rows.
- `eval_qa_combined.jsonl`: all 200 rows.

## Schema

```json
{
  "id": "claude_001",
  "category": "claude_code",
  "topic": "...",
  "question": "...",
  "answer": "...",
  "keywords": ["..."],
  "source_url": "https://..."
}
```

## Current validation status

The model outputs have been generated, but the reference answers and source URLs have not completed a full item-by-item primary-source audit. This dataset is therefore provisional.

An exact-question audit found:

- 200 rows;
- 83 unique question strings;
- 30 duplicate-question groups;
- 147 rows belonging to duplicate-question groups.

The rows reproduce the completed experiment and are intentionally not rewritten after generation. They must not be described as 200 independent benchmark questions.

- `answer` is a draft reference, not guaranteed ground truth.
- `keywords` support lexical-overlap diagnostics only.
- Exact match and keyword recall do not establish factual correctness.
- No external LLM-as-a-Judge score is published.

The QA rows are excluded from CPT and SFT training. The preprocessing metadata records matched QA source URLs for auditing.

