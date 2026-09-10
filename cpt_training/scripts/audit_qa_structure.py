#!/usr/bin/env python3
"""Audit QA row duplication and answer continuation without judging correctness."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(value)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Count exact duplicate questions and Question: continuations. "
            "This script does not evaluate factual correctness."
        )
    )
    parser.add_argument(
        "--qa",
        type=Path,
        default=REPO_ROOT
        / "ai_coding_agent_cpt_data/QA_Dataset/eval_qa_combined.jsonl",
    )
    parser.add_argument(
        "--answers",
        type=Path,
        default=REPO_ROOT
        / "cpt_training/results/current/qa/answers_combined.jsonl",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    qa_path = args.qa.resolve()
    answers_path = args.answers.resolve()
    qa_rows = read_jsonl(qa_path)
    answer_rows = read_jsonl(answers_path)

    question_counts = Counter(str(row.get("question", "")) for row in qa_rows)
    duplicate_groups = [
        {"question": question, "count": count}
        for question, count in question_counts.items()
        if count > 1
    ]
    duplicate_groups.sort(key=lambda row: (-int(row["count"]), str(row["question"])))

    model_names: list[str] = []
    if answer_rows:
        answers = answer_rows[0].get("answers", {})
        if not isinstance(answers, dict):
            raise ValueError("answers_combined first row has no answers object")
        model_names = list(answers)

    continuation_counts: dict[str, int] = {}
    nonempty_counts: dict[str, int] = {}
    for model_name in model_names:
        continuation_counts[model_name] = 0
        nonempty_counts[model_name] = 0
        for row in answer_rows:
            answers = row.get("answers", {})
            answer = str(answers.get(model_name, "")) if isinstance(answers, dict) else ""
            if answer.strip():
                nonempty_counts[model_name] += 1
            if "question:" in answer.lower()[20:]:
                continuation_counts[model_name] += 1

    pairwise_exact: dict[str, int] = {}
    comparison_pairs = [
        ("transformers_cpt", "unsloth_cpt"),
        ("transformers_cpt_sft", "unsloth_cpt_sft"),
        ("transformers_cpt_sft_v1", "unsloth_cpt_sft_v1"),
        ("transformers_cpt_sft_v2", "unsloth_cpt_sft_v2"),
        ("transformers_cpt_sft_v3", "unsloth_cpt_sft_v3"),
    ]
    for left, right in comparison_pairs:
        if left not in model_names or right not in model_names:
            continue
        matches = 0
        for row in answer_rows:
            answers = row.get("answers", {})
            if isinstance(answers, dict) and answers.get(left) == answers.get(right):
                matches += 1
        pairwise_exact[f"{left}__{right}"] = matches

    result = {
        "status": "ok",
        "scope": (
            "Structural and continuation audit only; no factual correctness judgment."
        ),
        "qa": {
            "rows": len(qa_rows),
            "unique_exact_question_strings": len(question_counts),
            "duplicate_question_groups": len(duplicate_groups),
            "rows_in_duplicate_question_groups": sum(
                int(row["count"]) for row in duplicate_groups
            ),
            "largest_duplicate_groups": duplicate_groups[:10],
        },
        "answers": {
            "rows": len(answer_rows),
            "models": model_names,
            "nonempty_rows": nonempty_counts,
            "question_marker_after_character_20": continuation_counts,
            "pairwise_exact_answer_matches": pairwise_exact,
        },
    }

    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
