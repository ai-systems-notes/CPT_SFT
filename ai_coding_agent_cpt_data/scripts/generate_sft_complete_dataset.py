#!/usr/bin/env python3
"""Create an append-only, longer SFT variant with a local Ollama model.

The original short answer is preserved verbatim as the first sentence.  The
local model may only add one explanatory sentence without adding code tokens.
No external API is used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import requests


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = (
    REPO_ROOT / "ai_coding_agent_cpt_data/SFT_Dataset/sft_general_train.jsonl"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "ai_coding_agent_cpt_data/SFT_Dataset/sft_general_complete_train.jsonl"
)
DEFAULT_CHAT_OUTPUT = (
    REPO_ROOT
    / "ai_coding_agent_cpt_data/SFT_Dataset/sft_general_complete_train_chat.jsonl"
)
DEFAULT_METADATA = (
    REPO_ROOT
    / "ai_coding_agent_cpt_data/SFT_Dataset/sft_general_complete_metadata.json"
)

RISKY_ADDITION_PATTERNS = (
    r"\b(?:always|guarantee(?:s|d)?|ensure(?:s|d)?|prevent(?:s|ed)?)\b",
    r"\b(?:securely|security|safe(?:ly)?|data loss|overheat(?:ing)?)\b",
    r"\b(?:without requiring|no special|all future|immediately)\b",
    r"\b(?:unauthori[sz]ed|credential(?:s)?|resource leak(?:s)?|crash(?:ing|es)?)\b",
    r"\b(?:independent copy|returns? one row|successful(?:ly)?)\b",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--chat-output", type=Path, default=DEFAULT_CHAT_OUTPUT)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--model", default="qwen3.5:4b-q8_0")
    parser.add_argument("--endpoint", default="http://127.0.0.1:11434/api/chat")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-retries", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if set(row) != {"instruction", "input", "output"}:
                raise ValueError(f"Invalid schema at {path}:{line_number}")
            rows.append(row)
    if len(rows) != 200:
        raise ValueError(f"Expected 200 source rows, got {len(rows)}")
    return rows


def normalize_generated_text(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip().strip('"')
    text = re.sub(r"^(?:Additional sentence|Answer|Explanation)\s*:\s*", "", text, flags=re.I)
    return text


def validate_addition(addition: str, original: str) -> list[str]:
    errors: list[str] = []
    if not 45 <= len(addition) <= 180:
        errors.append(f"addition length {len(addition)} is outside 45..180")
    if "\n" in addition or "Question:" in addition or "Answer:" in addition:
        errors.append("addition contains a newline or QA continuation marker")
    original_spans = set(re.findall(r"`([^`]+)`", original))
    if any(not any(span in original_span for original_span in original_spans) for span in re.findall(r"`([^`]+)`", addition)):
        errors.append("addition introduces a new backticked token")
    if re.search(r"(?<!\w)--[a-zA-Z][\w-]*", addition):
        errors.append("addition introduces a CLI flag")
    for pattern in RISKY_ADDITION_PATTERNS:
        if re.search(pattern, addition, flags=re.I):
            errors.append(f"addition contains an overclaim-risk phrase: {pattern}")
    if not re.search(r"[.!?]$", addition):
        errors.append("addition does not end as a sentence")
    if addition.casefold() in original.casefold() or original.casefold() in addition.casefold():
        errors.append("addition repeats the original answer")
    combined = f"{original} {addition}"
    if not 100 <= len(combined) <= 300:
        errors.append(f"combined length {len(combined)} is outside 100..300")
    return errors


def request_addition(
    endpoint: str,
    model: str,
    seed: int,
    instruction: str,
    original: str,
    previous_errors: list[str],
) -> str:
    feedback = ""
    if previous_errors:
        feedback = "\nPrevious output was rejected because: " + "; ".join(previous_errors)
    prompt = f"""Question:
{instruction}

Trusted original answer:
{original}

Write exactly one additional explanatory sentence of 45 to 180 characters.
The final training answer will be the trusted answer followed by your sentence.
Conservatively restate only the purpose or result already explicit in the trusted answer.
Do not add a benefit, guarantee, side effect, security claim, prerequisite, or behavior
that is not written in the trusted answer. Avoid absolute words such as always, ensures,
prevents, safely, securely, immediately, all, or without requiring.
Do not introduce commands, flags, file names, configuration keys, APIs, numbers, or factual claims absent from it.
Do not use backticks, labels, bullet points, or a new question.
Return only the additional sentence.{feedback}"""
    response = requests.post(
        endpoint,
        json={
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You conservatively expand trusted technical answers without "
                        "changing or adding facts."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "think": False,
            "options": {
                "temperature": 0.2,
                "seed": seed,
                "num_predict": 160,
            },
        },
        timeout=120,
    )
    response.raise_for_status()
    payload: dict[str, Any] = response.json()
    return normalize_generated_text(str(payload["message"]["content"]))


def write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def main() -> int:
    args = parse_args()
    source = args.input.resolve()
    output = args.output.resolve()
    chat_output = args.chat_output.resolve()
    metadata_output = args.metadata.resolve()
    for path in (output, chat_output, metadata_output):
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"Refusing to overwrite {path}; pass --overwrite")

    source_rows = read_jsonl(source)
    expanded_rows: list[dict[str, str]] = []
    retry_counts: list[int] = []
    started = time.perf_counter()
    for index, row in enumerate(source_rows):
        errors: list[str] = []
        addition = ""
        for attempt in range(1, args.max_retries + 1):
            addition = request_addition(
                args.endpoint,
                args.model,
                args.seed + index * args.max_retries + attempt,
                row["instruction"],
                row["output"],
                errors,
            )
            errors = validate_addition(addition, row["output"])
            if not errors:
                retry_counts.append(attempt - 1)
                break
        else:
            raise RuntimeError(
                f"Could not expand row {index + 1}: {errors}; last={addition!r}"
            )
        expanded_rows.append(
            {
                "instruction": row["instruction"],
                "input": row["input"],
                "output": f"{row['output']} {addition}",
            }
        )
        if (index + 1) % 10 == 0 or index + 1 == len(source_rows):
            print(f"expanded {index + 1}/{len(source_rows)}", flush=True)

    chat_rows = [
        {
            "messages": [
                {"role": "user", "content": row["instruction"]},
                {"role": "assistant", "content": row["output"]},
            ]
        }
        for row in expanded_rows
    ]
    write_jsonl_atomic(output, expanded_rows)
    write_jsonl_atomic(chat_output, chat_rows)
    lengths = [len(row["output"]) for row in expanded_rows]
    metadata = {
        "status": "locally_generated_and_rule_audited_not_manually_fact_checked",
        "method": "trusted short answer preserved verbatim plus one conservative restatement",
        "source": str(source.relative_to(REPO_ROOT)),
        "source_sha256": sha256_file(source),
        "model": args.model,
        "endpoint": "local Ollama /api/chat",
        "seed": args.seed,
        "rows": len(expanded_rows),
        "length_chars": {
            "min": min(lengths),
            "mean": sum(lengths) / len(lengths),
            "max": max(lengths),
        },
        "total_retries": sum(retry_counts),
        "generation_seconds": time.perf_counter() - started,
    }
    metadata_output.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"output={output}")
    print(f"chat_output={chat_output}")
    print(f"metadata={metadata_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
