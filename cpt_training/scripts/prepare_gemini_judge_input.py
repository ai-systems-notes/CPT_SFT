#!/usr/bin/env python3
"""Create deterministically shuffled, model-blind Gemini judge input."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--judge-output", type=Path, required=True)
    parser.add_argument("--map-output", type=Path, required=True)
    parser.add_argument("--metadata-output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260815)
    args = parser.parse_args()

    source_rows: list[dict[str, Any]] = []
    with args.input.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                source_rows.append(json.loads(line))
    if not source_rows:
        raise ValueError("Combined answer file is empty")

    model_labels = list(source_rows[0]["answers"])
    candidate_names = [chr(ord("A") + index) for index in range(len(model_labels))]
    judge_rows: list[dict[str, Any]] = []
    map_rows: list[dict[str, Any]] = []
    position_counts = {
        candidate: {model: 0 for model in model_labels} for candidate in candidate_names
    }

    for row in source_rows:
        if list(row["answers"]) != model_labels:
            raise ValueError(f"Model labels differ for item {row['id']}")
        seed_material = f"{args.seed}:{row['id']}".encode("utf-8")
        item_seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "big")
        shuffled_models = model_labels.copy()
        random.Random(item_seed).shuffle(shuffled_models)
        mapping = dict(zip(candidate_names, shuffled_models, strict=True))
        candidates = {
            candidate: row["answers"][model] for candidate, model in mapping.items()
        }
        for candidate, model in mapping.items():
            position_counts[candidate][model] += 1
        judge_rows.append(
            {
                key: value for key, value in row.items() if key != "answers"
            } | {"candidates": candidates}
        )
        map_rows.append({"id": row["id"], "candidate_to_model": mapping})

    write_jsonl(args.judge_output, judge_rows)
    write_jsonl(args.map_output, map_rows)
    metadata = {
        "status": "judge_input_prepared_not_judged",
        "source": str(args.input.resolve()),
        "items": len(judge_rows),
        "seed": args.seed,
        "shuffle": "per-item deterministic SHA-256-derived seed",
        "model_labels": model_labels,
        "position_counts": position_counts,
        "instruction": "Send the judge input to Gemini without the mapping file; use the mapping only after judging.",
    }
    args.metadata_output.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Prepared {len(judge_rows)} blind judge items", flush=True)


if __name__ == "__main__":
    main()
