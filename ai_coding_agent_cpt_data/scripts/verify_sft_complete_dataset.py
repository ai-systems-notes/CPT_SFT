#!/usr/bin/env python3
"""Verify parity and conservative expansion of the complete SFT variant."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SHORT = (
    REPO_ROOT / "ai_coding_agent_cpt_data/SFT_Dataset/sft_general_train.jsonl"
)
DEFAULT_COMPLETE = (
    REPO_ROOT
    / "ai_coding_agent_cpt_data/SFT_Dataset/sft_general_complete_train.jsonl"
)
DEFAULT_CHAT = (
    REPO_ROOT
    / "ai_coding_agent_cpt_data/SFT_Dataset/sft_general_complete_train_chat.jsonl"
)

RISKY_ADDITION_PATTERNS = (
    r"\b(?:always|guarantee(?:s|d)?|ensure(?:s|d)?|prevent(?:s|ed)?)\b",
    r"\b(?:securely|security|safe(?:ly)?|data loss|overheat(?:ing)?)\b",
    r"\b(?:without requiring|no special|all future|immediately)\b",
    r"\b(?:unauthori[sz]ed|credential(?:s)?|resource leak(?:s)?|crash(?:ing|es)?)\b",
    r"\b(?:independent copy|returns? one row|successful(?:ly)?)\b",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            rows.append(value)
    return rows


def code_spans(text: str) -> list[str]:
    return re.findall(r"`([^`]+)`", text)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--short", type=Path, default=DEFAULT_SHORT)
    parser.add_argument("--complete", type=Path, default=DEFAULT_COMPLETE)
    parser.add_argument("--chat", type=Path, default=DEFAULT_CHAT)
    args = parser.parse_args()

    short_rows = read_jsonl(args.short.resolve())
    complete_rows = read_jsonl(args.complete.resolve())
    chat_rows = read_jsonl(args.chat.resolve())
    if not (len(short_rows) == len(complete_rows) == len(chat_rows) == 200):
        raise ValueError(
            f"Expected 200 rows each, got {len(short_rows)}, "
            f"{len(complete_rows)}, {len(chat_rows)}"
        )

    outputs: list[str] = []
    lengths: list[int] = []
    for index, (short, complete, chat) in enumerate(
        zip(short_rows, complete_rows, chat_rows), 1
    ):
        if set(complete) != {"instruction", "input", "output"}:
            raise ValueError(f"Invalid complete schema at row {index}")
        if complete["instruction"] != short["instruction"] or complete["input"] != short["input"]:
            raise ValueError(f"Instruction/input changed at row {index}")
        output = str(complete["output"])
        if not output.startswith(str(short["output"]) + " "):
            raise ValueError(f"Trusted short answer is not preserved at row {index}")
        if not 100 <= len(output) <= 300:
            raise ValueError(f"Output length {len(output)} is outside 100..300 at row {index}")
        if "\n" in output or "Question:" in output or "Answer:" in output:
            raise ValueError(f"QA continuation marker/newline at row {index}")
        addition = output[len(str(short["output"])) + 1 :]
        original_spans = set(code_spans(str(short["output"])))
        if any(not any(span in original_span for original_span in original_spans) for span in code_spans(addition)):
            raise ValueError(f"New backticked technical token at row {index}")
        for pattern in RISKY_ADDITION_PATTERNS:
            if re.search(pattern, addition, flags=re.I):
                raise ValueError(
                    f"Overclaim-risk phrase in addition at row {index}: {pattern}"
                )
        expected_chat = {
            "messages": [
                {"role": "user", "content": complete["instruction"]},
                {"role": "assistant", "content": output},
            ]
        }
        if chat != expected_chat:
            raise ValueError(f"Chat parity mismatch at row {index}")
        outputs.append(output)
        lengths.append(len(output))

    if len(outputs) != len(set(outputs)):
        raise ValueError("Complete outputs are not unique")
    print("[PASS] complete SFT dataset")
    print(f"rows={len(outputs)}")
    print(f"chars=min:{min(lengths)} mean:{sum(lengths)/len(lengths):.1f} max:{max(lengths)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
