# QA 70 questions, 12 conditions, two independent judges

The analysis behind the Qwen3-0.6B CPT + LoRA-SFT write-up. Every number quoted
in the article is regenerated here by
`cpt_training/scripts/analyze_blog_claims.py`.

| File | Contents |
| :--- | :--- |
| `blog_claims.md` | Human-readable report: paired effects, judge agreement, corpus composition, duplication, exposure-vs-recall, source provenance. |
| `blog_claims.json` | The same values as structured data, including the corpus statistics that cannot be recomputed without the local corpus. |

## Where the raw evidence lives

This directory holds the analysis only. The inputs are in
`cpt_training/results/current/`:

| Path | Contents |
| :--- | :--- |
| `qa/qa_input.jsonl` | 70 evaluation questions. Each carries `source_url`, the source document's SHA-256, an evidence excerpt, and answer keywords. |
| `qa/qa_input_metadata.json` | How the question set was built and audited. |
| `qa/answers_<condition>.jsonl` | 70 answers per condition, 12 conditions, 840 answers total. Same prompt and `max_new_tokens=256` everywhere. |
| `qa/gemini_judge_results.jsonl` | Per-item scores from `gemini-3.5-flash-lite`. |
| `qa/gemini_judge_summary.md`, `qa/gemini_judge_metadata.json` | That judge's summary and run configuration. |
| `qa_12model_terra_judge/terra_judge_results.jsonl` | Per-item scores from three `gpt-5.6-terra` graders, stratified by `split × category`. |
| `qa_12model_terra_judge/terra_judge_summary.md`, `..._subsets.md` | That judge's own ranking tables and subset breakdowns. |

`qa_sft_v3/` and `qa_complete_sft/` are earlier runs on the same question set and
are not used by the write-up.

## The 12 conditions

Three starting points (`base`, `transformers_cpt`, `unsloth_cpt`) × four
post-training states (none, `sft_v1`, `sft_v2`, `sft_v3`). The SFT variants are
PEFT LoRA adapters applied to the named starting point.

## The two splits

| split | Items | Role |
| :--- | ---: | :--- |
| `sft_seen` | 30 | Knowledge written directly into the SFT set. A memorisation control. |
| `heldout` | 40 | Not in the SFT set. The primary measurement of CPT knowledge. |

Mixing them produces a misleading single accuracy number, so they are always
reported separately.

## How the effects are measured

Scores are compared **paired within an item**, not as a ranking table. Each item
is graded by one grader across all 12 conditions, so pairing cancels both item
difficulty and grader leniency. Confidence intervals come from 20,000 bootstrap
resamples with a fixed seed, so re-running reproduces them exactly.

Ranking tables over these 840 scores are not informative: more than half the
scores are 2 or below, and the leading conditions sit within 0.1 point of each
other. The paired deltas in `blog_claims.md` are the reportable result.

## Judge agreement

The two judges are independent — different model families, different runs,
neither shown the other's scores — and they agree closely: Pearson r = 0.8688
over 840 paired scores, with 24 of 25 significance calls matching.

The one disagreement is `base → transformers_cpt` on the 40 held-out questions:
`+0.500` `[-0.100, +1.125]` (terra, not significant) against `+0.625`
`[+0.025, +1.250]` (gemini, just significant). Both judges agree on the split
that carries the write-up's claim — Claude Code held-out is significant for
both, Codex held-out is non-significant for both.

## Reproducing

```bash
python3 cpt_training/scripts/analyze_blog_claims.py
```

Four sections additionally need the CPT corpus, which is **not published**
because it is third-party documentation text (see below). Without it the script
skips them and says so; their computed values remain in the committed
`blog_claims.json`:

- corpus composition (the write-up's central finding)
- duplication metrics
- exposure-vs-recall correlation
- question source provenance

To recompute those from scratch, rebuild the corpus first:

```bash
python3 ai_coding_agent_cpt_data/scripts/scrape_docs_spider.py
python3 ai_coding_agent_cpt_data/scripts/preprocess_cpt_data.py
```

## What is not published, and why

- **Corpus body text** (`artifacts/`). The scraped pages belong to their
  publishers, so they are not redistributed here. Every question in
  `qa_input.jsonl` records its `source_url` and the source document's SHA-256,
  so a third party can re-fetch the page and verify the reference independently.
- **Model weights, LoRA adapters, GGUF files, packed tensors.** Too large, and
  regenerable from the training scripts.

## Limitations

- Single seed, one epoch. Run-to-run variance is not measured, so small
  differences between conditions may be seed noise.
- 40 held-out items cannot resolve effects smaller than roughly ±0.6 points. A
  non-significant result here means "not measurable", not "no effect".
- Absolute scores are low across the board and carry each judge's subjectivity.
  Only the paired deltas should be read as results.
- The Claude Code / Codex comparison is n=2 domains. Topic difficulty, document
  style and the base model's prior knowledge are not controlled for.
