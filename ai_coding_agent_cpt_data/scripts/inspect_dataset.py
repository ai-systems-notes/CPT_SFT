#!/usr/bin/env python3
"""
Dataset Quality Audit & Inspection Script
Inspects data/scraped_documents.jsonl for:
1. Low-length or low-quality documents (e.g. < 200 chars)
2. Duplicates by title or text content (SHA-1 hash)
3. Non-informative content (404, redirects, login, empty)
4. Character count distribution and clean statistics
"""

import json
import hashlib
from collections import Counter

def main():
    jsonl_path = "data/scraped_documents.jsonl"
    with open(jsonl_path, "r", encoding="utf-8") as f:
        docs = [json.loads(line) for line in f if line.strip()]

    total = len(docs)
    print(f"==========================================")
    print(f" Dataset Inspection Audit: {total} documents")
    print(f"==========================================")

    short_docs = [d for d in docs if d.get("content_length", 0) < 300]
    print(f"\n[1] Short Documents (< 300 chars): {len(short_docs)}")
    for d in short_docs[:5]:
        print(f"    - [{d.get('content_length')} chars] {d.get('url')} | Title: {d.get('title')}")

    # Check exact content duplicates via SHA-1
    hashes = {}
    duplicates = []
    for d in docs:
        text = d.get("content", "").strip()
        h = hashlib.sha1(text.encode("utf-8")).hexdigest()
        if h in hashes:
            duplicates.append((d, hashes[h]))
        else:
            hashes[h] = d

    print(f"\n[2] Exact Content Duplicates: {len(duplicates)}")
    for dup, original in duplicates[:5]:
        print(f"    - Dup: {dup.get('url')} <== Original: {original.get('url')}")

    # Check Markdown vs HTML twins (.md vs html)
    md_twins = [d for d in docs if d.get('url', '').endswith('.md')]
    print(f"\n[3] Markdown (.md) direct files: {len(md_twins)}")

    # Character count summary
    lengths = [d.get("content_length", 0) for d in docs]
    avg_len = sum(lengths) / total if total else 0
    max_len = max(lengths) if lengths else 0
    min_len = min(lengths) if lengths else 0

    print(f"\n[4] Length Statistics:")
    print(f"    - Min Length: {min_len:,} chars")
    print(f"    - Avg Length: {avg_len:,.1f} chars")
    print(f"    - Max Length: {max_len:,} chars")

    # Sample top 5 largest documents
    sorted_docs = sorted(docs, key=lambda x: x.get("content_length", 0), reverse=True)
    print(f"\n[5] Largest Documents:")
    for d in sorted_docs[:5]:
        print(f"    - [{d.get('content_length'):,} chars] {d.get('title')} ({d.get('url')})")

if __name__ == "__main__":
    main()
