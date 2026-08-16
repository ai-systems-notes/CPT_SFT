# Pure General Instruction SFT Dataset (Canonical)

- **Date Updated**: 2026-08-16
- **Total Unique Samples**: 200 Q&A items
- **SHA-256 Digest**: `559c6121bda08fc81fe5e005906ccd7008d7902c7481b0b46dfe983d335b6f56`
- **Template Placeholder Count**: **0** (Zero `#1~#100` numeric placeholders)
- **Automated domain-term audit**: **0 forbidden-term hits** (not a proof of every possible semantic overlap)
- **Answer Formatting**: All answers strictly direct, 1 concise sentence (**100 characters or fewer** in Python `len()`, no newlines).

## Categories & Distribution (200 Total)
- **Git & Version Control**: 40 unique items
- **Linux CLI & Shell Operations**: 40 unique items
- **Python Programming**: 40 unique items
- **Docker, SQL & Web Development**: 40 unique items
- **Networking, Security & DevOps**: 40 unique items

## Formats
- `sft_general_train.jsonl`: Alpaca format (`instruction`, `input`, `output`)
- `sft_general_train_chat.jsonl`: ChatML format (`messages`: `user` / `assistant`)
