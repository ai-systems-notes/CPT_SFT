# SFT datasets

This directory keeps each SFT experiment as a separate, reproducible dataset.
Do not compare runs trained from different files as if only the model condition changed.

## SFT v3 (current experiment)

SFT v3 tests whether a small amount of domain instruction data helps expose knowledge
learned during CPT without returning to the overly short 100-character behavior of v1.

- 200 unique rows: 150 general technical, 20 ambiguity handling, 15 Claude Code, and 15 Codex
- Answers: one to three sentences, at most 300 Python characters; observed maximum is recorded in metadata
- Domain facts: source URL, source-document hash, exact tokens, and related evaluation IDs are recorded
- Leakage audit: declared related IDs are excluded, then independent evaluation rows are checked by fixed thresholds
- Held-out file: a conservative candidate set, not a claim of semantic correctness; manual review remains required

Files:

- `sft_v3_train.jsonl`: canonical Alpaca training data
- `sft_v3_train_chat.jsonl`: content-equivalent chat format
- `sft_v3_item_metadata.jsonl`: row/category/topic mapping and row hashes
- `sft_v3_domain_sources.jsonl`: provenance for the 30 domain rows
- `sft_v3_leakage_report.json`: similarity audit against the existing evaluation set
- `sft_v3_metadata.json`: counts, prompt, statistics, and artifact hashes
- `../QA_Dataset/eval_qa_v3_heldout.jsonl`: conservative held-out candidates

Regenerate and verify:

```bash
python3 ai_coding_agent_cpt_data/scripts/generate_sft_v3_dataset.py
python3 ai_coding_agent_cpt_data/scripts/verify_sft_v3_dataset.py
```

The verifier proves schema, parity, normalization, hashes, claimed-source token
support, and configured similarity thresholds. It does not replace manual factual review.

## Pure general instruction SFT (v1)

- **Date Updated**: 2026-08-16
- **Total Unique Samples**: 200 Q&A items
- **SHA-256 Digest**: `559c6121bda08fc81fe5e005906ccd7008d7902c7481b0b46dfe983d335b6f56`
- **Template Placeholder Count**: **0** (Zero `#1~#100` numeric placeholders)
- **Automated domain-term audit**: **0 forbidden-term hits** (not a proof of every possible semantic overlap)
- **Answer Formatting**: All answers strictly direct, 1 concise sentence (**100 characters or fewer** in Python `len()`, no newlines).

### Categories & Distribution (200 Total)
- **Git & Version Control**: 40 unique items
- **Linux CLI & Shell Operations**: 40 unique items
- **Python Programming**: 40 unique items
- **Docker, SQL & Web Development**: 40 unique items
- **Networking, Security & DevOps**: 40 unique items

### Formats
- `sft_general_train.jsonl`: Alpaca format (`instruction`, `input`, `output`)
- `sft_general_train_chat.jsonl`: ChatML format (`messages`: `user` / `assistant`)
