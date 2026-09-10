#!/usr/bin/env python3
"""Regenerate every quantitative claim used in the CPT/SFT write-up.

The script is deliberately self-contained: it reads the QA set, the twelve
answer files, both judges' raw scores and (when the locally regenerated corpus
is present) the CPT corpus, then writes one JSON and one Markdown report.

Corpus body text is not published, so the corpus-dependent sections are skipped
when `artifacts/datasets/cpt_v1/` is absent. Their previously computed values
stay available in the published JSON report.

Usage:
    python3 cpt_training/scripts/analyze_blog_claims.py [--out-dir DIR]
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
QA_DIR = REPO / "cpt_training/results/current/qa"
TERRA = REPO / "cpt_training/results/current/qa_12model_terra_judge/terra_judge_results.jsonl"
CORPUS = REPO / "artifacts/datasets/cpt_v1"

MODELS = [
    "base", "transformers_cpt", "unsloth_cpt",
    "base_sft_v1", "transformers_cpt_sft_v1", "unsloth_cpt_sft_v1",
    "base_sft_v2", "transformers_cpt_sft_v2", "unsloth_cpt_sft_v2",
    "base_sft_v3", "transformers_cpt_sft_v3", "unsloth_cpt_sft_v3",
]

BOOTSTRAP_RESAMPLES = 20_000
BOOTSTRAP_SEED = 20260815


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def doc_url(doc: dict) -> str:
    return doc.get("url") or doc.get("source_url") or doc.get("canonical_url") or ""


def doc_text(doc: dict) -> str:
    return doc.get("text") or doc.get("content") or ""


def canonical(url: str) -> str:
    return url.rstrip("/").replace(".md", "")


def corpus_category(url: str) -> str:
    """Split the corpus the way the evaluation questions are split.

    `/api/docs/` on the Codex documentation host is the OpenAI Platform API
    reference, which no evaluation question asks about.
    """
    if "claude.com" in url:
        return "claude_code"
    if "/api/docs/" in url:
        return "openai_platform_api"
    return "codex_cli"


# --------------------------------------------------------------------------
# paired bootstrap
# --------------------------------------------------------------------------

def paired_bootstrap(scores: dict[str, dict[str, float]], ids: list[str],
                     model_a: str, model_b: str) -> dict:
    """Bootstrap the per-item score difference (model_b - model_a).

    Pairing within an item cancels both item difficulty and judge leniency,
    which is why the write-up reports these deltas instead of a ranking table.
    """
    rng = random.Random(BOOTSTRAP_SEED)
    deltas = [scores[i][model_b] - scores[i][model_a] for i in ids]
    n = len(deltas)
    mean = sum(deltas) / n
    means = sorted(
        sum(deltas[rng.randrange(n)] for _ in range(n)) / n
        for _ in range(BOOTSTRAP_RESAMPLES)
    )
    low = means[int(0.025 * BOOTSTRAP_RESAMPLES)]
    high = means[int(0.975 * BOOTSTRAP_RESAMPLES)]
    return {
        "n": n,
        "mean_delta": round(mean, 4),
        "ci95_low": round(low, 4),
        "ci95_high": round(high, 4),
        "significant": bool(low > 0 or high < 0),
        "wins": sum(1 for d in deltas if d > 0),
        "losses": sum(1 for d in deltas if d < 0),
        "ties": sum(1 for d in deltas if d == 0),
    }


def pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    den = math.sqrt(sum((a - mx) ** 2 for a in xs) * sum((b - my) ** 2 for b in ys))
    return num / den if den else float("nan")


# --------------------------------------------------------------------------
# sections
# --------------------------------------------------------------------------

def judge_effects(qa: dict, judges: dict[str, dict]) -> dict:
    splits = {
        "overall_70": list(qa),
        "sft_seen_30": [i for i, q in qa.items() if q["split"] == "sft_seen"],
        "heldout_40": [i for i, q in qa.items() if q["split"] == "heldout"],
        "heldout_claude_20": [i for i, q in qa.items()
                              if q["split"] == "heldout" and q["category"] == "claude_code"],
        "heldout_codex_20": [i for i, q in qa.items()
                             if q["split"] == "heldout" and q["category"] == "openai_codex"],
    }
    comparisons = [
        ("base", "transformers_cpt"),
        ("base", "unsloth_cpt"),
        ("transformers_cpt", "transformers_cpt_sft_v3"),
        ("unsloth_cpt", "unsloth_cpt_sft_v3"),
        ("base", "base_sft_v3"),
    ]
    out: dict = {}
    for split_name, ids in splits.items():
        out[split_name] = {}
        for a, b in comparisons:
            key = f"{a}__to__{b}"
            out[split_name][key] = {
                judge: paired_bootstrap(scores, ids, a, b)
                for judge, scores in judges.items()
            }
    return out


def judge_agreement(qa: dict, judges: dict[str, dict], effects: dict) -> dict:
    names = list(judges)
    xs, ys = [], []
    for item in qa:
        for model in MODELS:
            xs.append(judges[names[0]][item][model])
            ys.append(judges[names[1]][item][model])
    concordant = discordant = 0
    for split_name, comparisons in effects.items():
        for key, per_judge in comparisons.items():
            verdicts = {per_judge[j]["significant"] for j in names}
            if len(verdicts) == 1:
                concordant += 1
            else:
                discordant += 1
    return {
        "judges": names,
        "paired_scores": len(xs),
        "pearson_r": round(pearson(xs, ys), 4),
        "mean_score": {names[0]: round(sum(xs) / len(xs), 4),
                       names[1]: round(sum(ys) / len(ys), 4)},
        "significance_calls_agreeing": concordant,
        "significance_calls_disagreeing": discordant,
    }


def qa_provenance(qa: dict) -> dict | None:
    """Check whether each question's source document was actually trained on.

    A question whose source landed in the CPT test split cannot measure recall,
    so the write-up reports those separately.
    """
    train_path, test_path = CORPUS / "train.jsonl", CORPUS / "test.jsonl"
    if not train_path.exists() or not test_path.exists():
        return None
    train = {canonical(doc_url(d)) for d in read_jsonl(train_path)}
    test = {canonical(doc_url(d)) for d in read_jsonl(test_path)}

    buckets: dict[str, list[str]] = {"train": [], "test": [], "absent": []}
    for item_id, q in qa.items():
        key = canonical(q["source_url"])
        where = "train" if key in train else "test" if key in test else "absent"
        buckets[where].append(item_id)
    return {
        "counts": {k: len(v) for k, v in buckets.items()},
        "heldout_not_in_train": sorted(
            i for k in ("test", "absent") for i in buckets[k] if qa[i]["split"] == "heldout"
        ),
        "item_ids": buckets,
    }


def corpus_composition() -> dict | None:
    """Token and document counts per category, plus duplication metrics.

    This is the section behind the write-up's central finding, so it reports
    both the document counts (which hide the imbalance) and the token counts
    (which expose it).
    """
    if not (CORPUS / "train.jsonl").exists():
        return None
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base", local_files_only=True)
    except Exception as exc:  # noqa: BLE001 - tokenizer is optional
        tokenizer = None
        tokenizer_error = str(exc)
    else:
        tokenizer_error = None

    result: dict = {"tokenizer_available": tokenizer is not None}
    if tokenizer_error:
        result["tokenizer_error"] = tokenizer_error

    for split in ("train", "test"):
        path = CORPUS / f"{split}.jsonl"
        if not path.exists():
            continue
        docs = read_jsonl(path)
        per_category: dict[str, dict] = {}
        for doc in docs:
            cat = corpus_category(doc_url(doc))
            entry = per_category.setdefault(cat, {"documents": 0, "characters": 0, "tokens": 0})
            text = doc_text(doc)
            entry["documents"] += 1
            entry["characters"] += len(text)
            if tokenizer is not None:
                entry["tokens"] += len(tokenizer(text, add_special_tokens=False)["input_ids"])
        total_tokens = sum(e["tokens"] for e in per_category.values())
        total_chars = sum(e["characters"] for e in per_category.values())
        for entry in per_category.values():
            basis = total_tokens or total_chars
            value = entry["tokens"] or entry["characters"]
            entry["share_percent"] = round(100 * value / basis, 2) if basis else 0.0
        result[split] = per_category
    return result


def duplication_metrics() -> dict | None:
    """Near-duplicate and unique-shingle rates per documentation source.

    Counted twice on purpose: raw lines include code fences and JSON braces,
    which inverts the ranking, so substantive lines are counted separately.
    """
    if not (CORPUS / "train.jsonl").exists():
        return None
    docs = read_jsonl(CORPUS / "train.jsonl")
    out: dict = {}
    for source in ("claude_code", "openai_codex"):
        host = "claude.com" if source == "claude_code" else "chatgpt.com"
        texts = [doc_text(d) for d in docs if host in doc_url(d)]

        raw_lines = [ln.strip() for t in texts for ln in t.split("\n") if ln.strip()]
        raw_counts = Counter(raw_lines)
        repeated_five_plus = sum(v for v in raw_counts.values() if v >= 5)

        solid = [ln for ln in raw_lines if len(ln) >= 20]
        solid_counts = Counter(solid)
        solid_chars = sum(len(ln) for ln in solid)
        duplicate_chars = sum(len(k) * (v - 1) for k, v in solid_counts.items() if v > 1)

        def shingles(text: str, k: int = 8) -> set[int]:
            words = text.split()
            return {hash(" ".join(words[i:i + k])) for i in range(max(1, len(words) - k + 1))}

        per_doc = [shingles(t) for t in texts]
        near_duplicate_pairs = 0
        for i in range(len(per_doc)):
            for j in range(i + 1, len(per_doc)):
                a, b = per_doc[i], per_doc[j]
                if not a or not b:
                    continue
                inter = len(a & b)
                if inter and (inter / len(a | b) >= 0.3 or inter / min(len(a), len(b)) >= 0.6):
                    near_duplicate_pairs += 1
        total_shingles = sum(len(s) for s in per_doc)
        unique_shingles = len(set().union(*per_doc)) if per_doc else 0

        out[source] = {
            "documents": len(texts),
            "raw_lines_repeated_5plus_percent": round(100 * repeated_five_plus / len(raw_lines), 2),
            "substantive_line_unique_percent": round(100 * len(solid_counts) / len(solid), 2),
            "substantive_duplicate_char_percent": round(100 * duplicate_chars / solid_chars, 2),
            "near_duplicate_document_pairs": near_duplicate_pairs,
            "unique_8gram_percent": round(100 * unique_shingles / total_shingles, 2),
        }
    return out


def exposure_vs_recall(qa: dict, judges: dict[str, dict]) -> dict | None:
    """Does seeing a fact more often in the corpus predict recalling it?

    Counts the corpus occurrences of each held-out question's answer keywords
    and correlates them with the CPT model's score.
    """
    if not (CORPUS / "train.jsonl").exists():
        return None
    texts = [doc_text(d) for d in read_jsonl(CORPUS / "train.jsonl")]
    blob = "\n".join(texts)

    rows = []
    for item_id, q in qa.items():
        if q["split"] != "heldout" or not q.get("keywords"):
            continue
        counts = [blob.count(k) for k in q["keywords"]]
        rows.append({
            "id": item_id,
            "min_occurrences": min(counts),
            "total_occurrences": sum(counts),
            "base": judges["terra"][item_id]["base"],
            "cpt": judges["terra"][item_id]["transformers_cpt"],
        })

    def bucket(row: dict) -> str:
        m = row["min_occurrences"]
        return "0" if m == 0 else "1-4" if m < 5 else "5-19" if m < 20 else "20+"

    buckets: dict[str, list[dict]] = {}
    for row in rows:
        buckets.setdefault(bucket(row), []).append(row)

    return {
        "n": len(rows),
        "pearson_r_log_min_occurrences_vs_cpt_score": round(
            pearson([math.log1p(r["min_occurrences"]) for r in rows], [r["cpt"] for r in rows]), 4),
        "pearson_r_log_total_occurrences_vs_cpt_score": round(
            pearson([math.log1p(r["total_occurrences"]) for r in rows], [r["cpt"] for r in rows]), 4),
        "by_bucket": {
            name: {
                "items": len(group),
                "base_mean": round(statistics.mean(r["base"] for r in group), 3),
                "cpt_mean": round(statistics.mean(r["cpt"] for r in group), 3),
                "delta": round(statistics.mean(r["cpt"] - r["base"] for r in group), 3),
            }
            for name, group in sorted(buckets.items())
        },
    }


def answer_lengths(qa: dict) -> dict:
    out = {}
    for model in MODELS:
        path = QA_DIR / f"answers_{model}.jsonl"
        if not path.exists():
            continue
        answers = {r["id"]: (r.get("answer") or "") for r in read_jsonl(path)}
        lengths = sorted(len(answers.get(i, "")) for i in qa)
        out[model] = {
            "mean_chars": round(statistics.mean(lengths), 1),
            "median_chars": statistics.median(lengths),
            "p95_chars": lengths[int(0.95 * len(lengths))],
        }
    return out


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------

def render_markdown(report: dict) -> str:
    lines = ["# Quantitative claims behind the CPT/SFT write-up", ""]
    lines += [
        f"- QA items: {report['inputs']['qa_items']}",
        f"- Conditions: {report['inputs']['models']}",
        f"- Graded answers: {report['inputs']['graded_answers']}",
        f"- Bootstrap resamples: {BOOTSTRAP_RESAMPLES:,} (seed {BOOTSTRAP_SEED})",
        "",
        "All deltas are paired within an item, so item difficulty and judge",
        "leniency cancel. `*` marks a 95% CI that excludes zero.",
        "",
    ]

    agreement = report["judge_agreement"]
    lines += [
        "## Judge agreement",
        "",
        f"- Judges: `{agreement['judges'][0]}` and `{agreement['judges'][1]}`",
        f"- Paired scores: {agreement['paired_scores']}",
        f"- Pearson r: **{agreement['pearson_r']}**",
        f"- Significance calls agreeing: "
        f"**{agreement['significance_calls_agreeing']}/"
        f"{agreement['significance_calls_agreeing'] + agreement['significance_calls_disagreeing']}**",
        "",
    ]

    lines += ["## Paired effects", ""]
    for split_name, comparisons in report["effects"].items():
        lines += [f"### {split_name}", "",
                  "| Comparison | Judge | Δ mean | 95% CI | W/L/T |",
                  "| :--- | :--- | ---: | :--- | :--- |"]
        for key, per_judge in comparisons.items():
            a, b = key.split("__to__")
            for judge, stat in per_judge.items():
                star = "*" if stat["significant"] else ""
                lines.append(
                    f"| `{a}` → `{b}` | {judge} | {stat['mean_delta']:+.3f}{star} | "
                    f"[{stat['ci95_low']:+.3f}, {stat['ci95_high']:+.3f}] | "
                    f"{stat['wins']}/{stat['losses']}/{stat['ties']} |"
                )
        lines.append("")

    composition = report.get("corpus_composition")
    if composition and "train" in composition:
        lines += ["## CPT corpus composition (train split)", "",
                  "| Category | Documents | Tokens | Share | Asked about by the 70 questions |",
                  "| :--- | ---: | ---: | ---: | :--- |"]
        asked = {"claude_code": "yes", "codex_cli": "yes", "openai_platform_api": "**no**"}
        for cat, entry in sorted(composition["train"].items(), key=lambda kv: -kv[1]["tokens"]):
            lines.append(
                f"| `{cat}` | {entry['documents']} | {entry['tokens']:,} | "
                f"{entry['share_percent']}% | {asked.get(cat, '?')} |"
            )
        lines.append("")

    exposure = report.get("exposure_vs_recall")
    if exposure:
        lines += ["## Corpus exposure vs recall (held-out)", "",
                  f"- r(log min occurrences, CPT score) = "
                  f"**{exposure['pearson_r_log_min_occurrences_vs_cpt_score']}**",
                  f"- r(log total occurrences, CPT score) = "
                  f"**{exposure['pearson_r_log_total_occurrences_vs_cpt_score']}**", "",
                  "| Min occurrences | Items | base | CPT | Δ |",
                  "| :--- | ---: | ---: | ---: | ---: |"]
        for name, entry in exposure["by_bucket"].items():
            lines.append(f"| {name} | {entry['items']} | {entry['base_mean']} | "
                         f"{entry['cpt_mean']} | {entry['delta']:+.3f} |")
        lines.append("")

    duplication = report.get("duplication")
    if duplication:
        lines += ["## Duplication", "",
                  "| Metric | claude_code | openai_codex |",
                  "| :--- | ---: | ---: |"]
        keys = [
            ("near_duplicate_document_pairs", "Near-duplicate document pairs"),
            ("unique_8gram_percent", "Unique 8-gram %"),
            ("substantive_line_unique_percent", "Unique substantive line %"),
            ("raw_lines_repeated_5plus_percent", "Raw lines repeated 5+ times %"),
        ]
        for key, label in keys:
            lines.append(f"| {label} | {duplication['claude_code'][key]} | "
                         f"{duplication['openai_codex'][key]} |")
        lines.append("")

    provenance = report.get("qa_provenance")
    if provenance:
        lines += ["## Question source provenance", "",
                  f"- In CPT train split: {provenance['counts']['train']} questions",
                  f"- In CPT test split (never trained on): "
                  f"**{provenance['counts']['test']}**",
                  f"- Not in the corpus: {provenance['counts']['absent']}", "",
                  "Held-out questions whose source was never trained on: "
                  + ", ".join(f"`{i}`" for i in provenance["heldout_not_in_train"]), ""]

    skipped = report["inputs"].get("skipped_sections")
    if skipped:
        lines += ["## Skipped sections", "",
                  "The CPT corpus body text is not published, so these sections "
                  "require regenerating it locally first:", ""]
        lines += [f"- `{name}`" for name in skipped]
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path,
                        default=REPO / "cpt_training/results/published/qa70_dual_judge")
    args = parser.parse_args()

    qa = {q["id"]: q for q in read_jsonl(QA_DIR / "qa_input.jsonl")}

    terra = {r["id"]: r["scores"] for r in read_jsonl(TERRA)}
    gemini = {r["id"]: r["judge_scores_by_model"]
              for r in read_jsonl(QA_DIR / "gemini_judge_results.jsonl")}
    judges = {"terra": terra, "gemini": gemini}

    for name, scores in judges.items():
        missing = set(qa) - set(scores)
        if missing:
            raise SystemExit(f"judge {name} is missing {len(missing)} items: {sorted(missing)[:5]}")

    effects = judge_effects(qa, judges)
    report = {
        "inputs": {
            "qa_items": len(qa),
            "models": len(MODELS),
            "graded_answers": len(qa) * len(MODELS),
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "bootstrap_seed": BOOTSTRAP_SEED,
        },
        "effects": effects,
        "judge_agreement": judge_agreement(qa, judges, effects),
        "answer_lengths": answer_lengths(qa),
    }

    optional = {
        "qa_provenance": qa_provenance(qa),
        "corpus_composition": corpus_composition(),
        "duplication": duplication_metrics(),
        "exposure_vs_recall": exposure_vs_recall(qa, judges),
    }
    skipped = [name for name, value in optional.items() if value is None]
    report.update({name: value for name, value in optional.items() if value is not None})
    if skipped:
        report["inputs"]["skipped_sections"] = skipped

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "blog_claims.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.out_dir / "blog_claims.md").write_text(
        render_markdown(report), encoding="utf-8")

    print(f"wrote {args.out_dir/'blog_claims.json'}")
    print(f"wrote {args.out_dir/'blog_claims.md'}")
    if skipped:
        print("skipped (corpus not present locally): " + ", ".join(skipped))


if __name__ == "__main__":
    main()
