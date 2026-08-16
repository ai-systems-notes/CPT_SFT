#!/usr/bin/env python3
"""Validate the canonical general-purpose SFT dataset and benchmark separation."""

from __future__ import annotations

import hashlib
import json
import re
import sys
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SFT_DIR = REPO_ROOT / "ai_coding_agent_cpt_data/SFT_Dataset"
TEST_FILE = REPO_ROOT / "ai_coding_agent_cpt_data/QA_Dataset/eval_qa_combined.jsonl"
ALPACA_FILE = SFT_DIR / "sft_general_train.jsonl"
CHAT_FILE = SFT_DIR / "sft_general_train_chat.jsonl"

FORBIDDEN_TERMS = (
    "claude",
    "codex",
    "anthropic",
    "openai",
    "chatgpt",
    "gpt",
    "coding agent",
    "worktree",
    ".claude",
    "agents.md",
    "mcp",
    "model context protocol",
    "teleport",
    "appserver",
    "code-review",
    "security-guidance",
    "appshots",
    "remote control",
    "claude-code",
    "agent sdk",
)
JACCARD_THRESHOLD = 0.35
SEQUENCE_THRESHOLD = 0.80


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            rows.append(value)
    return rows


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(re.findall(r"[a-z0-9]+", text))


def token_set(text: str) -> set[str]:
    return set(normalize_text(text).split())


def jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def has_multiple_prose_sentences(text: str) -> bool:
    prose = re.sub(r"`[^`]*`", "", text)
    prose = re.sub(r"(?<!\w)\.[A-Za-z][\w.-]*", "", prose)
    body = prose[:-1] if prose and prose[-1] in ".!?" else prose
    return bool(re.search(r"[!?]|(?<!\d)\.(?!\d)", body))


def main() -> int:
    print("=" * 60)
    print(" CANONICAL SFT DATASET VERIFICATION & LEAKAGE AUDIT")
    print("=" * 60)
    errors: list[str] = []

    for path in (ALPACA_FILE, CHAT_FILE, TEST_FILE):
        if not path.is_file():
            errors.append(f"Missing file: {path.relative_to(REPO_ROOT)}")
    if errors:
        for error in errors:
            print(f"[FAIL] {error}")
        return 1

    try:
        alpaca_rows = read_jsonl(ALPACA_FILE)
        chat_rows = read_jsonl(CHAT_FILE)
        test_rows = read_jsonl(TEST_FILE)
    except (OSError, ValueError) as exc:
        print(f"[FAIL] {exc}")
        return 1

    print(f"Alpaca SFT rows: {len(alpaca_rows)}")
    print(f"Chat SFT rows: {len(chat_rows)}")
    print(f"Benchmark rows: {len(test_rows)}")
    if len(alpaca_rows) != 200:
        errors.append(f"Alpaca row count is {len(alpaca_rows)}, expected 200")
    if len(chat_rows) != 200:
        errors.append(f"Chat row count is {len(chat_rows)}, expected 200")
    if len(test_rows) != 200:
        errors.append(f"Benchmark row count is {len(test_rows)}, expected 200")

    normalized_instructions: set[str] = set()
    normalized_outputs: set[str] = set()
    for index, row in enumerate(alpaca_rows, 1):
        if set(row) != {"instruction", "input", "output"}:
            errors.append(f"Alpaca row {index} has invalid keys: {sorted(row)}")
            continue
        instruction = row["instruction"]
        input_text = row["input"]
        output = row["output"]
        if not isinstance(instruction, str) or not instruction.strip():
            errors.append(f"Alpaca row {index} has an empty/non-string instruction")
            continue
        if input_text != "":
            errors.append(f"Alpaca row {index} input must be an empty string")
        if not isinstance(output, str) or not output.strip():
            errors.append(f"Alpaca row {index} has an empty/non-string output")
            continue

        normalized_instruction = normalize_text(instruction)
        normalized_output = normalize_text(output)
        if normalized_instruction in normalized_instructions:
            errors.append(f"Alpaca row {index} duplicates a normalized instruction")
        if normalized_output in normalized_outputs:
            errors.append(f"Alpaca row {index} duplicates a normalized output")
        normalized_instructions.add(normalized_instruction)
        normalized_outputs.add(normalized_output)

        if len(output) > 100:
            errors.append(f"Alpaca row {index} output length is {len(output)}, expected <= 100")
        if "\n" in output or "\r" in output:
            errors.append(f"Alpaca row {index} output contains a newline")
        if not re.search(r"[.!?]$", output):
            errors.append(f"Alpaca row {index} output lacks terminal punctuation")
        elif has_multiple_prose_sentences(output):
            errors.append(f"Alpaca row {index} output contains multiple prose sentences")
        if re.search(r"(?:task|pattern|concept)?\s*#\d+", f"{instruction} {output}", re.I):
            errors.append(f"Alpaca row {index} contains a numbered placeholder")

        lowered = f"{instruction} {input_text} {output}".casefold()
        for term in FORBIDDEN_TERMS:
            if term in lowered:
                errors.append(f"Alpaca row {index} contains forbidden term {term!r}")

    for index, row in enumerate(chat_rows, 1):
        if set(row) != {"messages"} or not isinstance(row.get("messages"), list):
            errors.append(f"Chat row {index} has invalid schema")
            continue
        messages = row["messages"]
        if len(messages) != 2 or any(not isinstance(message, dict) for message in messages):
            errors.append(f"Chat row {index} must contain exactly two message objects")
            continue
        if set(messages[0]) != {"role", "content"} or set(messages[1]) != {"role", "content"}:
            errors.append(f"Chat row {index} message keys are invalid")
            continue
        if messages[0]["role"] != "user" or messages[1]["role"] != "assistant":
            errors.append(f"Chat row {index} roles must be user then assistant")
            continue
        if index <= len(alpaca_rows):
            alpaca = alpaca_rows[index - 1]
            if messages[0]["content"] != alpaca.get("instruction"):
                errors.append(f"Chat row {index} user content differs from Alpaca")
            if messages[1]["content"] != alpaca.get("output"):
                errors.append(f"Chat row {index} assistant content differs from Alpaca")

    test_questions = [normalize_text(row.get("question", "")) for row in test_rows]
    test_answers = [normalize_text(row.get("answer", "")) for row in test_rows]
    max_q_jaccard = max_a_jaccard = 0.0
    max_q_sequence = max_a_sequence = 0.0
    for sft_index, row in enumerate(alpaca_rows, 1):
        if not isinstance(row.get("instruction"), str) or not isinstance(row.get("output"), str):
            continue
        instruction = normalize_text(row["instruction"])
        output = normalize_text(row["output"])
        if instruction in test_questions:
            errors.append(f"Alpaca row {sft_index} exactly matches a normalized benchmark question")
        if output in test_answers:
            errors.append(f"Alpaca row {sft_index} exactly matches a normalized benchmark answer")
        for test_index, (question, answer) in enumerate(zip(test_questions, test_answers), 1):
            q_jaccard = jaccard(token_set(instruction), token_set(question))
            a_jaccard = jaccard(token_set(output), token_set(answer))
            q_sequence = SequenceMatcher(None, instruction, question, autojunk=False).ratio()
            a_sequence = SequenceMatcher(None, output, answer, autojunk=False).ratio()
            max_q_jaccard = max(max_q_jaccard, q_jaccard)
            max_a_jaccard = max(max_a_jaccard, a_jaccard)
            max_q_sequence = max(max_q_sequence, q_sequence)
            max_a_sequence = max(max_a_sequence, a_sequence)
            if q_jaccard >= JACCARD_THRESHOLD:
                errors.append(f"SFT {sft_index}/benchmark {test_index} question Jaccard={q_jaccard:.3f}")
            if a_jaccard >= JACCARD_THRESHOLD:
                errors.append(f"SFT {sft_index}/benchmark {test_index} answer Jaccard={a_jaccard:.3f}")
            if q_sequence >= SEQUENCE_THRESHOLD:
                errors.append(f"SFT {sft_index}/benchmark {test_index} question SequenceMatcher={q_sequence:.3f}")
            if a_sequence >= SEQUENCE_THRESHOLD:
                errors.append(f"SFT {sft_index}/benchmark {test_index} answer SequenceMatcher={a_sequence:.3f}")

    max_output_length = max((len(row.get("output", "")) for row in alpaca_rows), default=0)
    digest = hashlib.sha256(ALPACA_FILE.read_bytes()).hexdigest()
    print(f"Maximum output length: {max_output_length}")
    print(f"Dataset SHA-256: {digest}")
    print(f"Max Jaccard: question={max_q_jaccard:.4f}, answer={max_a_jaccard:.4f}")
    print(f"Max SequenceMatcher: question={max_q_sequence:.4f}, answer={max_a_sequence:.4f}")
    print("=" * 60)
    if errors:
        print(f"[FAIL] Found {len(errors)} violation(s)")
        for error in errors[:30]:
            print(f"  - {error}")
        if len(errors) > 30:
            print(f"  ... and {len(errors) - 30} more")
        return 1
    print("[PASS] All strict checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
