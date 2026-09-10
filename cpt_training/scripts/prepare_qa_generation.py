#!/usr/bin/env python3
"""Apply documented QA fixes without modifying the source dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--fixes", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--max-question-chars", type=int, default=1000)
    args = parser.parse_args()

    source_path = args.input.resolve()
    fixes_path = args.fixes.resolve() if args.fixes else None
    output_path = args.output.resolve()
    metadata_path = args.metadata.resolve()
    fixes: dict[str, dict[str, Any]] = (
        json.loads(fixes_path.read_text(encoding="utf-8")) if fixes_path else {}
    )
    seen_fixes: set[str] = set()
    rows: list[dict[str, Any]] = []

    with source_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            item_id = item.get("id")
            if item_id in fixes:
                fix = fixes[item_id]
                item = {**item, "question": fix["question"]}
                seen_fixes.add(item_id)
            if len(item.get("question", "")) > args.max_question_chars:
                raise ValueError(
                    f"Question {item_id!r} on line {line_number} exceeds "
                    f"{args.max_question_chars} characters after fixes"
                )
            rows.append(item)

    unapplied = set(fixes) - seen_fixes
    if unapplied:
        raise ValueError(f"Fix IDs not found in dataset: {sorted(unapplied)}")
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate QA IDs")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    metadata = {
        "source": str(source_path),
        "source_sha256": sha256_file(source_path),
        "fixes": str(fixes_path) if fixes_path else None,
        "fixes_sha256": sha256_file(fixes_path) if fixes_path else None,
        "applied_fix_ids": sorted(seen_fixes),
        "item_count": len(rows),
        "max_question_chars": args.max_question_chars,
        "output": str(output_path),
        "output_sha256": sha256_file(output_path),
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Prepared {len(rows)} QA items; fixes={sorted(seen_fixes)}", flush=True)


if __name__ == "__main__":
    main()
