#!/usr/bin/env python3
"""Strict, reproducible verification for the canonical SFT v3 artifacts."""

import hashlib
import json
import re
import sys
import unicodedata
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "ai_coding_agent_cpt_data"
SFT_DIR = DATA_DIR / "SFT_Dataset"
QA_DIR = DATA_DIR / "QA_Dataset"

ALPACA_PATH = SFT_DIR / "sft_v3_train.jsonl"
CHAT_PATH = SFT_DIR / "sft_v3_train_chat.jsonl"
ITEM_METADATA_PATH = SFT_DIR / "sft_v3_item_metadata.jsonl"
METADATA_PATH = SFT_DIR / "sft_v3_metadata.json"
DOMAIN_PATH = SFT_DIR / "sft_v3_domain_sources.jsonl"
LEAKAGE_PATH = SFT_DIR / "sft_v3_leakage_report.json"
SCRAPED_PATH = DATA_DIR / "data" / "scraped_documents.jsonl"
EVAL_PATH = QA_DIR / "eval_qa_combined.jsonl"
HELDOUT_PATH = QA_DIR / "eval_qa_v3_heldout.jsonl"
HELDOUT_METADATA_PATH = QA_DIR / "eval_qa_v3_heldout_metadata.json"

QUESTION_JACCARD_LIMIT = 0.35
QUESTION_SEQUENCE_LIMIT = 0.80
ANSWER_JACCARD_LIMIT = 0.35
ANSWER_SEQUENCE_LIMIT = 0.80
KNOWN_BAD_EVAL_TOPICS = {"Agent SDK API"}
EXPECTED_BREAKDOWN = {
    "general_tech": 150,
    "ambiguous_qa": 20,
    "claude_code": 15,
    "codex": 15,
}
FORBIDDEN_OUTPUT_PATTERNS = {
    "Q&A continuation": r"(?:^|\s)(?:Question|Answer):",
    "destructive rm": r"(?:^|\s)rm\s+-rf(?:\s|$)",
    "destructive reset": r"git\s+reset\s+--hard",
    "destructive clean": r"git\s+clean\s+-[^\s]*f",
    "destructive Docker prune": r"docker\s+system\s+prune",
    "forced kill": r"(?:^|\s)kill\s+-9(?:\s|$)",
    "world-writable chmod": r"chmod\s+777",
}


def load_jsonl(path):
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
    return rows


def load_json(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def normalize_text(text):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(text))).strip()


def normalized_key(text):
    return normalize_text(text).casefold()


def canonical_url(url):
    value = str(url).strip().rstrip("/")
    return value[:-3] if value.endswith(".md") else value


def sources_for_url(scraped_docs, source_url):
    exact = [doc for doc in scraped_docs if doc.get("url") == source_url]
    if exact:
        return exact
    canonical = canonical_url(source_url)
    return [
        doc
        for doc in scraped_docs
        if canonical_url(doc.get("url", "")) == canonical
    ]


def jaccard(left, right):
    left_tokens = set(re.findall(r"\w+", str(left).lower()))
    right_tokens = set(re.findall(r"\w+", str(right).lower()))
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def sequence(left, right):
    return SequenceMatcher(None, str(left).lower(), str(right).lower()).ratio()


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def text_sha256(text):
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def sentence_count(text):
    return len([part for part in re.split(r"(?<=[.!?])\s+", text.strip()) if part])


def percentile(values, percentile_value):
    ordered = sorted(values)
    index = int(len(ordered) * percentile_value / 100)
    return ordered[min(index, len(ordered) - 1)]


def check(condition, message, errors):
    if not condition:
        errors.append(message)


def main():
    errors = []
    paths = (
        ALPACA_PATH,
        CHAT_PATH,
        ITEM_METADATA_PATH,
        METADATA_PATH,
        DOMAIN_PATH,
        LEAKAGE_PATH,
        SCRAPED_PATH,
        EVAL_PATH,
        HELDOUT_PATH,
        HELDOUT_METADATA_PATH,
    )
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        for path in missing:
            print(f"FAIL: missing file: {path}")
        return 1

    try:
        alpaca = load_jsonl(ALPACA_PATH)
        chat = load_jsonl(CHAT_PATH)
        item_metadata = load_jsonl(ITEM_METADATA_PATH)
        metadata = load_json(METADATA_PATH)
        domain = load_jsonl(DOMAIN_PATH)
        leakage = load_json(LEAKAGE_PATH)
        scraped = load_jsonl(SCRAPED_PATH)
        eval_rows = load_jsonl(EVAL_PATH)
        heldout = load_jsonl(HELDOUT_PATH)
        heldout_metadata = load_json(HELDOUT_METADATA_PATH)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"FAIL: {exc}")
        return 1

    check(len(alpaca) == 200, f"Alpaca count {len(alpaca)} != 200", errors)
    check(len(chat) == 200, f"Chat count {len(chat)} != 200", errors)
    check(len(item_metadata) == 200, f"item metadata count {len(item_metadata)} != 200", errors)
    check(len(domain) == 30, f"domain source count {len(domain)} != 30", errors)
    check(metadata.get("schema_version") == 2, "metadata schema_version must be 2", errors)
    check(
        metadata.get("status") == "rule_audited_manual_domain_review_required",
        "metadata status must retain the manual-review requirement",
        errors,
    )
    check(metadata.get("item_count") == 200, "metadata item_count must be 200", errors)
    check(metadata.get("breakdown") == EXPECTED_BREAKDOWN, "metadata breakdown mismatch", errors)

    instruction_rows = {}
    output_rows = {}
    observed_breakdown = Counter()
    output_lengths = []
    observed_sentences = {"1": 0, "2": 0, "3": 0, "3+": 0}

    for index, row in enumerate(alpaca):
        check(set(row) == {"instruction", "input", "output"}, f"row {index}: invalid Alpaca schema", errors)
        instruction = row.get("instruction", "")
        output = row.get("output", "")
        check(row.get("input") == "", f"row {index}: input must be empty", errors)
        check(bool(instruction) and bool(output), f"row {index}: instruction/output must be non-empty", errors)
        check(instruction == normalize_text(instruction), f"row {index}: instruction is not normalized", errors)
        check(output == normalize_text(output), f"row {index}: output is not normalized", errors)

        instruction_key = normalized_key(instruction)
        output_key = normalized_key(output)
        if instruction_key in instruction_rows:
            errors.append(f"row {index}: duplicate instruction with row {instruction_rows[instruction_key]}")
        if output_key in output_rows:
            errors.append(f"row {index}: duplicate output with row {output_rows[output_key]}")
        instruction_rows[instruction_key] = index
        output_rows[output_key] = index

        check(len(output) <= 300, f"row {index}: output exceeds 300 characters", errors)
        sentences = sentence_count(output)
        check(1 <= sentences <= 3, f"row {index}: output has {sentences} sentences", errors)
        output_lengths.append(len(output))
        sentence_bucket = str(sentences) if sentences in (1, 2, 3) else "3+"
        observed_sentences[sentence_bucket] += 1

        for label, pattern in FORBIDDEN_OUTPUT_PATTERNS.items():
            if re.search(pattern, output, re.IGNORECASE):
                errors.append(f"row {index}: forbidden {label}")

        if index >= len(chat):
            continue
        messages = chat[index].get("messages")
        expected_messages = [
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": output},
        ]
        check(set(chat[index]) == {"messages"}, f"row {index}: invalid Chat schema", errors)
        check(messages == expected_messages, f"row {index}: Alpaca/Chat parity mismatch", errors)

        if index >= len(item_metadata):
            continue
        item_meta = item_metadata[index]
        check(item_meta.get("row_index") == index, f"row {index}: metadata row_index mismatch", errors)
        check(
            item_meta.get("instruction_sha256") == text_sha256(instruction),
            f"row {index}: instruction hash mismatch",
            errors,
        )
        check(
            item_meta.get("output_sha256") == text_sha256(output),
            f"row {index}: output hash mismatch",
            errors,
        )
        observed_breakdown[item_meta.get("category")] += 1

    check(dict(observed_breakdown) == EXPECTED_BREAKDOWN, "item metadata breakdown mismatch", errors)

    expected_stats = {
        "min": min(output_lengths),
        "mean": round(sum(output_lengths) / len(output_lengths), 2),
        "median": percentile(output_lengths, 50),
        "p95": percentile(output_lengths, 95),
        "max": max(output_lengths),
    }
    check(metadata.get("output_char_length_stats") == expected_stats, "output length statistics mismatch", errors)
    check(metadata.get("sentence_counts") == observed_sentences, "sentence count statistics mismatch", errors)

    eval_by_id = {row.get("id"): row for row in eval_rows}
    leakage_by_topic = {row.get("topic"): row for row in leakage.get("items", [])}
    domain_by_topic = {row.get("topic"): row for row in domain}
    check(len(leakage_by_topic) == 30, "leakage report must contain 30 unique topics", errors)
    check(len(domain_by_topic) == 30, "domain source topics must be unique", errors)
    check(leakage.get("schema_version") == 2, "leakage schema_version must be 2", errors)
    check(leakage.get("all_items_passed_automated_thresholds") is True, "leakage report is not passing", errors)
    check(leakage.get("manual_fact_review_still_required") is True, "leakage report lost manual-review flag", errors)

    max_metrics = {"q_j": 0.0, "q_s": 0.0, "a_j": 0.0, "a_s": 0.0}
    all_related_ids = set()
    for offset, source_row in enumerate(domain):
        row_index = 170 + offset
        topic = source_row.get("topic")
        check(source_row.get("row_index") == row_index, f"domain {topic}: row_index mismatch", errors)
        if row_index < len(item_metadata):
            check(item_metadata[row_index].get("topic") == topic, f"domain {topic}: item topic mismatch", errors)
            check(item_metadata[row_index].get("category") == source_row.get("category"), f"domain {topic}: category mismatch", errors)
        if row_index < len(alpaca):
            check(alpaca[row_index].get("instruction") == source_row.get("instruction"), f"domain {topic}: instruction mismatch", errors)

        source_docs = sources_for_url(scraped, source_row.get("source_url", ""))
        check(len(source_docs) == 1, f"domain {topic}: expected one claimed source, got {len(source_docs)}", errors)
        if len(source_docs) == 1:
            source_text = str(source_docs[0].get("content", ""))
            source_folded = source_text.casefold()
            check(source_row.get("source_sha256") == text_sha256(source_text), f"domain {topic}: source hash mismatch", errors)
            required = source_row.get("required_exact_tokens", [])
            check(bool(required), f"domain {topic}: required_exact_tokens is empty", errors)
            for token in required:
                check(str(token).casefold() in source_folded, f"domain {topic}: missing required token {token!r}", errors)
            answer_tokens = re.findall(r"`([^`]+)`", alpaca[row_index]["output"])
            check(answer_tokens == source_row.get("answer_exact_tokens"), f"domain {topic}: answer token metadata mismatch", errors)
            for token in answer_tokens:
                check(token.casefold() in source_folded, f"domain {topic}: unsupported answer token {token!r}", errors)

        related_ids = source_row.get("related_eval_ids", [])
        check(len(related_ids) == len(set(related_ids)), f"domain {topic}: duplicate related_eval_ids", errors)
        for eval_id in related_ids:
            check(eval_id in eval_by_id, f"domain {topic}: unknown related eval id {eval_id}", errors)
        all_related_ids.update(related_ids)

        report_row = leakage_by_topic.get(topic, {})
        check(report_row.get("related_eval_ids") == related_ids, f"domain {topic}: leakage related IDs mismatch", errors)
        check(report_row.get("passed_thresholds") is True, f"domain {topic}: report marks threshold failure", errors)
        check(not report_row.get("threshold_breaches"), f"domain {topic}: report contains threshold breaches", errors)

        independent = [row for row in eval_rows if row.get("id") not in related_ids]
        scores = []
        for eval_row in independent:
            scores.append({
                "id": eval_row["id"],
                "q_j": jaccard(alpaca[row_index]["instruction"], eval_row["question"]),
                "q_s": sequence(alpaca[row_index]["instruction"], eval_row["question"]),
                "a_j": jaccard(alpaca[row_index]["output"], eval_row["answer"]),
                "a_s": sequence(alpaca[row_index]["output"], eval_row["answer"]),
            })
        maxima = {key: max(scores, key=lambda score: score[key]) for key in max_metrics}
        limits = {
            "q_j": QUESTION_JACCARD_LIMIT,
            "q_s": QUESTION_SEQUENCE_LIMIT,
            "a_j": ANSWER_JACCARD_LIMIT,
            "a_s": ANSWER_SEQUENCE_LIMIT,
        }
        report_fields = {
            "q_j": ("question_jaccard", "question_jaccard_eval_id"),
            "q_s": ("question_seq_matcher", "question_seq_eval_id"),
            "a_j": ("answer_jaccard", "answer_jaccard_eval_id"),
            "a_s": ("answer_seq_matcher", "answer_seq_eval_id"),
        }
        for key, maximum in maxima.items():
            max_metrics[key] = max(max_metrics[key], maximum[key])
            check(maximum[key] < limits[key], f"domain {topic}: {key}={maximum[key]:.4f} breaches limit", errors)
            value_field, id_field = report_fields[key]
            check(report_row.get(value_field) == round(maximum[key], 4), f"domain {topic}: {value_field} mismatch", errors)
            check(report_row.get(id_field) == maximum["id"], f"domain {topic}: {id_field} mismatch", errors)

    file_hashes = metadata.get("files", {})
    for path in (ALPACA_PATH, CHAT_PATH, ITEM_METADATA_PATH, DOMAIN_PATH, LEAKAGE_PATH):
        expected_hash = file_hashes.get(path.name, {}).get("sha256")
        check(expected_hash == sha256(path), f"metadata hash mismatch: {path.name}", errors)

    check(heldout_metadata.get("schema_version") == 2, "held-out metadata schema_version must be 2", errors)
    check(
        heldout_metadata.get("status") == "candidate_requires_manual_fact_review",
        "held-out metadata status must retain manual-review requirement",
        errors,
    )
    check(heldout_metadata.get("heldout_candidate_count") == len(heldout), "held-out candidate count mismatch", errors)
    check(heldout_metadata.get("file_sha256") == sha256(HELDOUT_PATH), "held-out hash mismatch", errors)
    check(heldout_metadata.get("selection_does_not_prove_semantic_correctness") is True, "held-out semantic caveat missing", errors)

    heldout_ids = set()
    heldout_questions = set()
    for row in heldout:
        row_id = row.get("id")
        question_key = normalized_key(row.get("question", ""))
        check(row_id in eval_by_id, f"held-out unknown id {row_id}", errors)
        if row_id in eval_by_id:
            check(row == eval_by_id[row_id], f"held-out row {row_id} differs from canonical eval row", errors)
        check(row_id not in heldout_ids, f"held-out duplicate id {row_id}", errors)
        check(question_key not in heldout_questions, f"held-out duplicate question {row_id}", errors)
        check(row_id not in all_related_ids, f"held-out fact-level overlap {row_id}", errors)
        check(row.get("topic") not in KNOWN_BAD_EVAL_TOPICS, f"held-out known bad topic {row_id}", errors)
        check(len(normalize_text(row.get("question", ""))) <= 1000, f"held-out malformed long question {row_id}", errors)
        heldout_ids.add(row_id)
        heldout_questions.add(question_key)

        source_docs = sources_for_url(scraped, row.get("source_url", ""))
        check(len(source_docs) == 1, f"held-out {row_id}: expected one claimed source, got {len(source_docs)}", errors)
        if len(source_docs) == 1:
            source_folded = str(source_docs[0].get("content", "")).casefold()
            keywords = [normalize_text(value) for value in row.get("keywords", [])]
            check(bool(keywords), f"held-out {row_id}: keywords are empty", errors)
            for keyword in keywords:
                check(keyword.casefold() in source_folded, f"held-out {row_id}: source lacks keyword {keyword!r}", errors)
            for token in re.findall(r"`([^`]+)`", str(row.get("answer", ""))):
                check(token.casefold() in source_folded, f"held-out {row_id}: source lacks answer token {token!r}", errors)

        for source_row in domain:
            domain_row = alpaca[source_row["row_index"]]
            metrics = (
                jaccard(row["question"], domain_row["instruction"]),
                sequence(row["question"], domain_row["instruction"]),
                jaccard(row["answer"], domain_row["output"]),
                sequence(row["answer"], domain_row["output"]),
            )
            limits = (
                QUESTION_JACCARD_LIMIT,
                QUESTION_SEQUENCE_LIMIT,
                ANSWER_JACCARD_LIMIT,
                ANSWER_SEQUENCE_LIMIT,
            )
            check(
                all(value < limit for value, limit in zip(metrics, limits)),
                f"held-out {row_id}: undeclared overlap with domain topic {source_row['topic']}",
                errors,
            )

    audit_log = heldout_metadata.get("audit_log", [])
    accepted_audit_ids = {row.get("id") for row in audit_log if row.get("status") == "accepted"}
    check(len(audit_log) == len(eval_rows), "held-out audit log must cover every canonical eval row", errors)
    check(accepted_audit_ids == heldout_ids, "held-out accepted audit IDs mismatch", errors)
    check(
        heldout_metadata.get("excluded_count") == len(eval_rows) - len(heldout),
        "held-out excluded count mismatch",
        errors,
    )

    print("SFT v3 verification")
    print(f"  train/chat/item metadata: {len(alpaca)}/{len(chat)}/{len(item_metadata)}")
    print(f"  breakdown: {dict(observed_breakdown)}")
    print(f"  output chars: {expected_stats}")
    print(f"  domain sources: {len(domain)}; held-out candidates: {len(heldout)}")
    print(
        "  max independent similarity: "
        f"q_j={max_metrics['q_j']:.4f}, q_seq={max_metrics['q_s']:.4f}, "
        f"a_j={max_metrics['a_j']:.4f}, a_seq={max_metrics['a_s']:.4f}"
    )
    print(f"  train sha256: {sha256(ALPACA_PATH)}")
    print(f"  held-out sha256: {sha256(HELDOUT_PATH)}")

    if errors:
        print(f"FAIL: {len(errors)} verification error(s)")
        for error in errors:
            print(f"  - {error}")
        return 1

    print("PASS: all automated checks passed; manual factual review remains required.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
