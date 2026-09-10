#!/usr/bin/env python3
"""Compute lightweight deterministic metrics from saved QA generations."""

from __future__ import annotations

import argparse
import json
import os
import re
import unicodedata
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULT_DIR = REPO_ROOT / "cpt_training/results/current/qa"


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(re.findall(r"[a-z0-9]+", text))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"No answers found in {path}")
    return rows


def write_text_atomic(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    args = parser.parse_args()
    result_dir = args.result_dir.resolve()
    rows = read_jsonl(result_dir / "answers_combined.jsonl")
    model_labels = list(rows[0].get("answers", {}))
    if not model_labels:
        raise ValueError("No model answers found in answers_combined.jsonl")

    metrics: dict[str, dict[str, float | int]] = {}
    for label in model_labels:
        answers = [str(row.get("answers", {}).get(label, "")) for row in rows]
        references = [str(row.get("answer") or row.get("reference_answer", "")) for row in rows]
        exact = sum(normalize(answer) == normalize(reference) for answer, reference in zip(answers, references))
        keyword_hits = keyword_total = 0
        for row, answer in zip(rows, answers):
            normalized_answer = normalize(answer)
            for keyword in row.get("keywords") or []:
                keyword_total += 1
                keyword_hits += normalize(str(keyword)) in normalized_answer
        total = len(rows)
        metrics[label] = {
            "items": total,
            "nonempty_rate": sum(bool(answer.strip()) for answer in answers) / total,
            "under_100_chars_rate": sum(len(answer) <= 100 for answer in answers) / total,
            "average_answer_chars": sum(len(answer) for answer in answers) / total,
            "normalized_exact_match_rate": exact / total,
            "keyword_recall": keyword_hits / keyword_total if keyword_total else 0.0,
            "keyword_hits": keyword_hits,
            "keyword_total": keyword_total,
        }

    payload = {
        "status": "ok",
        "note": "Auxiliary generation metrics; they do not replace factual review or LLM-as-a-Judge.",
        "items": len(rows),
        "models": metrics,
    }
    write_text_atomic(
        result_dir / "qa_automatic_metrics.json",
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
    )
    table = []
    for label, value in metrics.items():
        table.append(
            f"| `{label}` | {value['nonempty_rate'] * 100:.1f}% | "
            f"{value['under_100_chars_rate'] * 100:.1f}% | "
            f"{value['average_answer_chars']:.1f} | "
            f"{value['normalized_exact_match_rate'] * 100:.1f}% | "
            f"{value['keyword_recall'] * 100:.1f}% |"
        )
    report = """# Automatic QA Generation Metrics

These are auxiliary formatting and lexical metrics, not a factual-quality verdict.

| Model | Non-empty | <=100 chars | Avg chars | Exact match | Keyword recall |
| :--- | ---: | ---: | ---: | ---: | ---: |
""" + "\n".join(table) + "\n"
    write_text_atomic(result_dir / "qa_automatic_metrics.md", report)
    print(f"Automatic QA metrics: {result_dir / 'qa_automatic_metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
