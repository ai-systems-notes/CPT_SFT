#!/usr/bin/env python3
"""Build deterministic CPT train/test splits and packed token tensors.

This script deliberately performs only minimal cleaning.  It keeps the broad
Claude Code / Codex / ChatGPT / OpenAI API scope while removing obvious stubs,
aggregate documents, and duplicate representations of the same page.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlsplit, urlunsplit

import torch
from transformers import AutoTokenizer


AGGREGATE_FILENAMES = {"llms.txt", "llms-full.txt", "codex-manual.md"}
REPO_ROOT = Path(__file__).resolve().parents[2]


def public_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("ai_coding_agent_cpt_data/data/scraped_documents.jsonl"),
    )
    parser.add_argument(
        "--qa-file",
        type=Path,
        default=Path("ai_coding_agent_cpt_data/QA_Dataset/eval_qa_combined.jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/datasets/cpt_v1"),
    )
    parser.add_argument(
        "--public-stats-output",
        type=Path,
        default=Path("cpt_training/results/dataset_cpt_v1_stats.json"),
    )
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B-Base")
    parser.add_argument("--sequence-length", type=int, default=1024)
    parser.add_argument("--test-ratio", type=float, default=0.25)
    parser.add_argument("--min-chars", type=int, default=800)
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Allow tokenizer downloads. By default, only the local HF cache is used.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite derived output files if they already exist.",
    )
    args = parser.parse_args()

    if not 0.0 < args.test_ratio < 1.0:
        parser.error("--test-ratio must be between 0 and 1")
    if args.sequence_length <= 0:
        parser.error("--sequence-length must be positive")
    if args.min_chars < 0:
        parser.error("--min-chars must be non-negative")
    return args


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_content(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    return re.sub(r"\s+", " ", text).strip()


def canonicalize_url(url: str) -> str:
    parsed = urlsplit(url.strip())
    path = parsed.path
    if path.lower().endswith(".md"):
        path = path[:-3]
    path = path.rstrip("/") or "/"
    return urlunsplit(
        (parsed.scheme.lower(), parsed.netloc.lower(), path, "", "")
    )


def is_markdown_url(url: str) -> bool:
    return urlsplit(url).path.lower().endswith(".md")


def is_aggregate_url(url: str) -> bool:
    filename = Path(urlsplit(url).path).name.lower()
    return filename in AGGREGATE_FILENAMES


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(item, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            rows.append(item)
    return rows


def document_preference(doc: dict[str, Any]) -> tuple[int, int, str]:
    return (
        1 if is_markdown_url(doc["url"]) else 0,
        len(doc["content"]),
        doc["url"],
    )


def excluded_record(doc: dict[str, Any], reason: str) -> dict[str, Any]:
    content = str(doc.get("content", ""))
    url = str(doc.get("url", ""))
    return {
        "reason": reason,
        "url": url,
        "canonical_url": canonicalize_url(url) if url else "",
        "seed_name": doc.get("seed_name", "unknown"),
        "content_length": len(content),
        "content_sha256": sha256_bytes(normalize_content(content).encode("utf-8")),
    }


def minimal_clean(
    raw_docs: list[dict[str, Any]], min_chars: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    candidates: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []

    for raw_doc in raw_docs:
        doc = dict(raw_doc)
        content = doc.get("content")
        url = doc.get("url")
        if not isinstance(content, str) or not content.strip():
            excluded.append(excluded_record(doc, "missing_content"))
            continue
        if not isinstance(url, str) or not url.strip():
            excluded.append(excluded_record(doc, "missing_url"))
            continue
        if is_aggregate_url(url):
            excluded.append(excluded_record(doc, "aggregate_document"))
            continue
        if len(content) < min_chars:
            excluded.append(excluded_record(doc, "short_stub"))
            continue

        normalized = normalize_content(content)
        doc["content"] = content.strip()
        doc["content_length"] = len(doc["content"])
        doc["canonical_url"] = canonicalize_url(url)
        doc["content_sha256"] = sha256_bytes(normalized.encode("utf-8"))
        candidates.append(doc)

    # First collapse exact normalized-content duplicates, preferring Markdown.
    by_content_hash: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for doc in candidates:
        by_content_hash[doc["content_sha256"]].append(doc)

    exact_deduped: list[dict[str, Any]] = []
    for group in by_content_hash.values():
        chosen = max(group, key=document_preference)
        exact_deduped.append(chosen)
        for doc in group:
            if doc is not chosen:
                excluded.append(excluded_record(doc, "exact_content_duplicate"))

    # Then collapse HTML/.md representations sharing a canonical URL.
    by_canonical_url: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for doc in exact_deduped:
        by_canonical_url[doc["canonical_url"]].append(doc)

    cleaned: list[dict[str, Any]] = []
    for group in by_canonical_url.values():
        chosen = max(group, key=document_preference)
        cleaned.append(chosen)
        for doc in group:
            if doc is not chosen:
                excluded.append(excluded_record(doc, "canonical_url_duplicate"))

    cleaned.sort(key=lambda doc: doc["canonical_url"])
    return cleaned, excluded


def load_qa_sources(path: Path) -> tuple[int, set[str]]:
    if not path.exists():
        return 0, set()
    items = load_jsonl(path)
    sources = {
        canonicalize_url(str(item["source_url"]))
        for item in items
        if item.get("source_url")
    }
    return len(items), sources


def stable_fraction(seed: int, value: str) -> float:
    digest = hashlib.sha256(f"{seed}|{value}".encode("utf-8")).digest()
    integer = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return integer / float(2**64)


def improve_test_selection(
    docs: list[dict[str, Any]],
    forced_train_urls: set[str],
    target_tokens: int,
    seed: int,
    test_ratio: float,
) -> set[str]:
    candidates = [
        doc for doc in docs if doc["canonical_url"] not in forced_train_urls
    ]
    if len(candidates) < 2:
        return set()

    test_urls = {
        doc["canonical_url"]
        for doc in candidates
        if stable_fraction(seed, doc["canonical_url"]) < test_ratio
    }
    if not test_urls:
        test_urls.add(
            min(
                candidates,
                key=lambda doc: stable_fraction(seed, doc["canonical_url"]),
            )["canonical_url"]
        )
    if len(test_urls) == len(candidates):
        test_urls.remove(
            max(
                candidates,
                key=lambda doc: stable_fraction(seed, doc["canonical_url"]),
            )["canonical_url"]
        )

    token_by_url = {doc["canonical_url"]: doc["token_count"] for doc in candidates}
    current_tokens = sum(token_by_url[url] for url in test_urls)

    # Preserve the hash-based random split, but move one document at a time when
    # doing so strictly improves the per-source token target.
    while True:
        current_error = abs(target_tokens - current_tokens)
        best_action: tuple[str, str, int] | None = None
        best_error = current_error

        if current_tokens < target_tokens:
            for url, tokens in token_by_url.items():
                if url in test_urls:
                    continue
                error = abs(target_tokens - (current_tokens + tokens))
                if error < best_error:
                    best_error = error
                    best_action = ("add", url, tokens)
        else:
            if len(test_urls) > 1:
                for url in sorted(test_urls):
                    tokens = token_by_url[url]
                    error = abs(target_tokens - (current_tokens - tokens))
                    if error < best_error:
                        best_error = error
                        best_action = ("remove", url, tokens)

        if best_action is None:
            break
        action, url, tokens = best_action
        if action == "add":
            test_urls.add(url)
            current_tokens += tokens
        else:
            test_urls.remove(url)
            current_tokens -= tokens

    return test_urls


def split_documents(
    docs: list[dict[str, Any]],
    qa_source_urls: set[str],
    test_ratio: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for doc in docs:
        by_source[str(doc.get("seed_name", "unknown"))].append(doc)

    test_urls: set[str] = set()
    for source, source_docs in sorted(by_source.items()):
        source_tokens = sum(doc["token_count"] for doc in source_docs)
        target_tokens = round(source_tokens * test_ratio)
        source_forced_train = {
            doc["canonical_url"]
            for doc in source_docs
            if doc["canonical_url"] in qa_source_urls
        }
        test_urls.update(
            improve_test_selection(
                source_docs,
                source_forced_train,
                target_tokens,
                seed + int.from_bytes(source.encode("utf-8"), "little") % 1_000_003,
                test_ratio,
            )
        )

    train_docs: list[dict[str, Any]] = []
    test_docs: list[dict[str, Any]] = []
    for doc in docs:
        item = dict(doc)
        item["qa_source"] = item["canonical_url"] in qa_source_urls
        if item["canonical_url"] in test_urls:
            item["split"] = "test"
            test_docs.append(item)
        else:
            item["split"] = "train"
            train_docs.append(item)

    train_docs.sort(key=lambda doc: doc["canonical_url"])
    test_docs.sort(key=lambda doc: doc["canonical_url"])
    return train_docs, test_docs


def public_document(doc: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "url",
        "canonical_url",
        "content",
        "content_length",
        "token_count",
        "content_sha256",
        "status",
        "seed_name",
        "qa_source",
        "split",
    )
    return {key: doc[key] for key in keys if key in doc}


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def pack_documents(
    docs: list[dict[str, Any]],
    sequence_length: int,
    eos_token_id: int,
    pad_token_id: int,
    seed: int,
    split: str,
) -> tuple[dict[str, torch.Tensor], dict[str, int]]:
    ordered = sorted(
        docs,
        key=lambda doc: stable_fraction(seed, f"{split}|{doc['canonical_url']}"),
    )
    stream: list[int] = []
    content_tokens = 0
    for doc in ordered:
        ids = doc["_input_ids"]
        stream.extend(ids)
        stream.append(eos_token_id)
        content_tokens += len(ids)

    non_padding_tokens = len(stream)
    block_count = math.ceil(non_padding_tokens / sequence_length)
    padded_tokens = block_count * sequence_length
    input_ids = torch.full(
        (block_count, sequence_length), pad_token_id, dtype=torch.int32
    )
    attention_mask = torch.zeros(
        (block_count, sequence_length), dtype=torch.bool
    )
    flat_ids = torch.tensor(stream, dtype=torch.int32)
    input_ids.view(-1)[:non_padding_tokens] = flat_ids
    attention_mask.view(-1)[:non_padding_tokens] = True

    tensors = {"input_ids": input_ids, "attention_mask": attention_mask}
    metadata = {
        "documents": len(docs),
        "content_tokens": content_tokens,
        "eos_tokens": len(docs),
        "non_padding_tokens": non_padding_tokens,
        "padding_tokens": padded_tokens - non_padding_tokens,
        "blocks": block_count,
        "sequence_length": sequence_length,
    }
    return tensors, metadata


def aggregate_split_stats(docs: list[dict[str, Any]]) -> dict[str, Any]:
    by_source: dict[str, dict[str, int]] = defaultdict(
        lambda: {"documents": 0, "characters": 0, "content_tokens": 0}
    )
    for doc in docs:
        source = str(doc.get("seed_name", "unknown"))
        by_source[source]["documents"] += 1
        by_source[source]["characters"] += doc["content_length"]
        by_source[source]["content_tokens"] += doc["token_count"]
    return {
        "documents": len(docs),
        "characters": sum(doc["content_length"] for doc in docs),
        "content_tokens": sum(doc["token_count"] for doc in docs),
        "qa_source_documents": sum(bool(doc.get("qa_source")) for doc in docs),
        "by_source": dict(sorted(by_source.items())),
    }


def ensure_outputs_available(paths: Iterable[Path], overwrite: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        rendered = "\n".join(f"  - {path}" for path in existing)
        raise FileExistsError(
            "Derived output already exists. Pass --overwrite to replace it:\n"
            f"{rendered}"
        )


def main() -> int:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(args.input)

    packed_dir = args.output_dir / "packed" / f"seq{args.sequence_length}"
    output_paths = {
        "cleaned": args.output_dir / "cleaned.jsonl",
        "train": args.output_dir / "train.jsonl",
        "test": args.output_dir / "test.jsonl",
        "excluded": args.output_dir / "excluded_documents.jsonl",
        "manifest": args.output_dir / "split_manifest.csv",
        "stats": args.output_dir / "dataset_stats.json",
        "train_packed": packed_dir / "train.pt",
        "test_packed": packed_dir / "test.pt",
        "packed_metadata": packed_dir / "metadata.json",
    }
    ensure_outputs_available(
        [*output_paths.values(), args.public_stats_output], args.overwrite
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    packed_dir.mkdir(parents=True, exist_ok=True)
    args.public_stats_output.parent.mkdir(parents=True, exist_ok=True)

    raw_docs = load_jsonl(args.input)
    cleaned_docs, excluded_docs = minimal_clean(raw_docs, args.min_chars)
    qa_item_count, qa_source_urls = load_qa_sources(args.qa_file)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        local_files_only=not args.allow_download,
        use_fast=True,
    )
    if tokenizer.eos_token_id is None:
        raise ValueError(f"Tokenizer {args.model} has no eos_token_id")
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id

    for doc in cleaned_docs:
        ids = tokenizer.encode(
            doc["content"], add_special_tokens=False, verbose=False
        )
        doc["_input_ids"] = ids
        doc["token_count"] = len(ids)
        doc["doc_id"] = sha256_bytes(doc["canonical_url"].encode("utf-8"))[:16]

    cleaned_urls = {doc["canonical_url"] for doc in cleaned_docs}
    matched_qa_sources = qa_source_urls & cleaned_urls
    missing_qa_sources = sorted(qa_source_urls - cleaned_urls)

    train_docs, test_docs = split_documents(
        cleaned_docs,
        matched_qa_sources,
        args.test_ratio,
        args.seed,
    )
    train_urls = {doc["canonical_url"] for doc in train_docs}
    test_urls = {doc["canonical_url"] for doc in test_docs}
    if train_urls & test_urls:
        raise AssertionError("canonical URL leakage between train and test")
    if not matched_qa_sources.issubset(train_urls):
        raise AssertionError("one or more matched QA sources were not forced into train")

    train_tensors, train_packed_stats = pack_documents(
        train_docs,
        args.sequence_length,
        tokenizer.eos_token_id,
        pad_token_id,
        args.seed,
        "train",
    )
    test_tensors, test_packed_stats = pack_documents(
        test_docs,
        args.sequence_length,
        tokenizer.eos_token_id,
        pad_token_id,
        args.seed,
        "test",
    )

    write_jsonl(output_paths["cleaned"], (public_document(d) for d in cleaned_docs))
    write_jsonl(output_paths["train"], (public_document(d) for d in train_docs))
    write_jsonl(output_paths["test"], (public_document(d) for d in test_docs))
    write_jsonl(output_paths["excluded"], excluded_docs)

    with output_paths["manifest"].open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "doc_id",
            "split",
            "qa_source",
            "seed_name",
            "url",
            "canonical_url",
            "content_length",
            "token_count",
            "content_sha256",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        split_docs = [*train_docs, *test_docs]
        for doc in sorted(split_docs, key=lambda d: (d["split"], d["canonical_url"])):
            writer.writerow({key: doc.get(key, "") for key in fieldnames})

    torch.save(train_tensors, output_paths["train_packed"])
    torch.save(test_tensors, output_paths["test_packed"])
    packed_metadata = {
        "format": "torch tensors with int32 input_ids and bool attention_mask",
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": pad_token_id,
        "train": train_packed_stats,
        "test": test_packed_stats,
    }
    output_paths["packed_metadata"].write_text(
        json.dumps(packed_metadata, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )

    exclusion_counts = Counter(item["reason"] for item in excluded_docs)
    train_stats = aggregate_split_stats(train_docs)
    test_stats = aggregate_split_stats(test_docs)
    total_split_tokens = train_stats["content_tokens"] + test_stats["content_tokens"]
    stats: dict[str, Any] = {
        "dataset_version": "cpt_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input": {
            "path": public_path(args.input),
            "sha256": sha256_file(args.input),
            "documents": len(raw_docs),
        },
        "qa": {
            "path": public_path(args.qa_file),
            "sha256": sha256_file(args.qa_file) if args.qa_file.exists() else None,
            "items": qa_item_count,
            "unique_source_urls": len(qa_source_urls),
            "matched_source_urls": len(matched_qa_sources),
            "missing_source_urls": missing_qa_sources,
        },
        "config": {
            "model": args.model,
            "tokenizer_class": tokenizer.__class__.__name__,
            "sequence_length": args.sequence_length,
            "requested_test_ratio": args.test_ratio,
            "min_chars": args.min_chars,
            "seed": args.seed,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": pad_token_id,
            "local_files_only": not args.allow_download,
        },
        "cleaning": {
            "raw_documents": len(raw_docs),
            "cleaned_documents": len(cleaned_docs),
            "excluded_documents": len(excluded_docs),
            "excluded_by_reason": dict(sorted(exclusion_counts.items())),
        },
        "splits": {
            "train": train_stats,
            "test": test_stats,
            "actual_test_document_ratio": len(test_docs) / len(cleaned_docs),
            "actual_test_token_ratio": (
                test_stats["content_tokens"] / total_split_tokens
                if total_split_tokens
                else 0.0
            ),
            "canonical_url_overlap": len(train_urls & test_urls),
        },
        "packed": packed_metadata,
    }

    # Hash the derived files that are consumed by later stages.
    stats["outputs"] = {
        name: {"path": public_path(path), "sha256": sha256_file(path)}
        for name, path in output_paths.items()
        if name not in {"stats"}
    }
    rendered_stats = (
        json.dumps(stats, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    output_paths["stats"].write_text(rendered_stats, encoding="utf-8")
    args.public_stats_output.write_text(rendered_stats, encoding="utf-8")

    print(f"Raw documents:       {len(raw_docs):,}")
    print(f"Cleaned documents:   {len(cleaned_docs):,}")
    print(f"Excluded documents:  {len(excluded_docs):,}")
    print(
        f"Train: {len(train_docs):,} docs / "
        f"{train_stats['content_tokens']:,} content tokens / "
        f"{train_packed_stats['blocks']:,} packed blocks"
    )
    print(
        f"Test:  {len(test_docs):,} docs / "
        f"{test_stats['content_tokens']:,} content tokens / "
        f"{test_packed_stats['blocks']:,} packed blocks"
    )
    print(f"Actual test token ratio: {stats['splits']['actual_test_token_ratio']:.4f}")
    print(
        f"QA source URLs: {len(matched_qa_sources):,} matched / "
        f"{len(missing_qa_sources):,} missing"
    )
    print(f"Stats: {output_paths['stats']}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise

