# Qwen3-0.6B CPT + LoRA-SFT Experiment

English | [日本語](README.md)

This repository documents a reproducible consumer-GPU experiment that performs Continued Pretraining (CPT) of Qwen3-0.6B-Base on official AI coding-agent web documentation and compares a standard Transformers/PyTorch implementation with Unsloth on an RTX 4070 12GB.

After CPT, the same general-purpose PEFT LoRA-SFT stage was applied to the base model and both CPT checkpoints. Six conditions each generated 200 QA rows, for 1,200 saved answers in total.

The goal is not to publish a production model. The goal is to preserve reproducible code, configuration, procedures, and measurements for documentation collection, full-parameter CPT, LoRA-SFT, and QA generation on a consumer GPU.

> [!IMPORTANT]
> CPT, LoRA-SFT, generation of 840 answers across 12 conditions, primary-source validation of the references, and two independent LLM-as-a-Judge runs are complete.
> Per-domain knowledge gain is reported with confidence intervals, but at 70 questions and a single seed this repository still does not name a “best model.”

## Status

| Stage | Status |
| --- | --- |
| Official web documentation collection and preprocessing | Complete |
| Full-parameter CPT with Transformers and Unsloth | Complete |
| LoRA-SFT from Base, Transformers-CPT, and Unsloth-CPT | Complete |
| Six conditions × 200 rows, 1,200 generated answers | Complete (superseded question set) |
| Formatting and lexical metrics | Complete |
| Question set rebuilt to 70 primary-source-backed items | Complete |
| Twelve conditions × 70 questions, 840 generated answers | Complete |
| Primary-source audit of all QA references | Complete (70/70; two unresolved, see below) |
| External LLM-as-a-Judge | Complete (`gpt-5.6-terra` ×3 and `gemini-3.5-flash-lite`) |
| Paired effect sizes with bootstrap confidence intervals | Complete |

### How far the validation goes

- Every one of the 70 questions records a source URL, the source document's SHA-256, an evidence excerpt and an automated validation result. All 70 pass "all required keywords present in a single official document", and the judges rated all 70 reference answers `good`.
- **Two items cite a document that never entered training.** The source for `seen_codex_013` and `seen_codex_014`, `codex-manual.md`, is a real official page (HTTP 200) that was collected and then dropped in preprocessing as an `aggregate_document`: at 1,062,687 characters it is the whole manual on one page. The references are still grounded in that page. Both sit in `sft_seen`, the memorisation control, so they never enter the CPT-recall measurement. Excluding them raises `base → base_sft_v3` from +1.767 to +1.893, so reporting them is the conservative choice.
- Scoring used two independent judges: Pearson r = 0.8688 across 840 scores, with 24 of 25 significance calls matching.
- Regenerate every number with `python3 cpt_training/scripts/analyze_blog_claims.py`; see [`cpt_training/results/published/qa70_dual_judge/`](cpt_training/results/published/qa70_dual_judge/).

Reports and public artifacts:

- [English experiment report](docs/experiment_report_en.md)
- [日本語レポート](docs/experiment_report_ja.md)
- [Published metrics and generated answers](cpt_training/results/published/)

## Main measurements

### Full CPT comparison

Both implementations used Qwen3-0.6B-Base, sequence length 1,024, one epoch, and the same packed dataset.

| Run | Framework | tokens/s | peak allocated | `nvidia-smi` peak | train time | held-out loss |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | Transformers/PyTorch | 4,754.61 | 8,474 MiB | 9,450 MiB | 321.18 s | 2.5403 → 2.1455 |
| 1 | Unsloth | 4,761.86 | 7,673 MiB | 8,060 MiB | 320.55 s | 2.5402 → 2.1455 |
| 2 | Transformers/PyTorch | 4,981.27 | 8,474 MiB | 9,450 MiB | 305.72 s | 2.5403 → 2.1456 |
| 2 | Unsloth | 4,778.68 | 7,673 MiB | 8,060 MiB | 319.12 s | 2.5402 → 2.1455 |

Unsloth reduced peak allocated VRAM by about 801 MiB in both runs. Speed was effectively tied in run 1, while Transformers/PyTorch was faster in run 2. These measurements do not support a blanket “Unsloth is always faster” conclusion.

### LoRA-SFT

The SFT stage uses PEFT LoRA (`r=16`, `alpha=32`, `dropout=0.05`), not full fine-tuning.

| Source model | loss | train time | tokens/s | peak allocated |
| --- | ---: | ---: | ---: | ---: |
| Base | 0.6172 | 10.36 s | 2,871.27 | 2,519 MiB |
| Transformers-CPT | 0.6135 | 10.24 s | 2,905.33 | 2,519 MiB |
| Unsloth-CPT | 0.6126 | 10.12 s | 2,939.11 | 2,519 MiB |

Each condition used 39 optimizer steps and 29,754 observed non-padding tokens. There were 10,092,544 trainable parameters and 606,142,464 total parameters including the adapter.

### Auxiliary QA-generation metrics

Each condition used the same prompt, greedy decoding, BF16, and `max_new_tokens=256`.

| Condition | ≤100 characters | Mean characters | Keyword recall |
| --- | ---: | ---: | ---: |
| Base | 10.0% | 172.8 | 36.5% |
| Transformers-CPT | 0.5% | 991.2 | 33.5% |
| Unsloth-CPT | 2.0% | 931.0 | 33.3% |
| Base + LoRA-SFT | 65.0% | 92.3 | 21.2% |
| Transformers-CPT + LoRA-SFT | 92.0% | 79.2 | 24.8% |
| Unsloth-CPT + LoRA-SFT | 94.5% | 78.5 | 25.3% |

These are formatting and lexical-overlap metrics, not factual-accuracy scores.

## Six compared conditions

```text
Qwen3-0.6B-Base
├── base
├── transformers_cpt       # full-parameter CPT
├── unsloth_cpt            # full-parameter CPT
├── base_sft               # Base + LoRA-SFT
├── transformers_cpt_sft   # Transformers-CPT + LoRA-SFT
└── unsloth_cpt_sft        # Unsloth-CPT + LoRA-SFT
```

## Training configuration

### CPT

- Base model: [`Qwen/Qwen3-0.6B-Base`](https://huggingface.co/Qwen/Qwen3-0.6B-Base)
- Revision: `da87bfb608c14b7cf20ba1ce41287e8de496c0cd`
- Sequence length: 1,024
- Update: full parameter
- Optimizer: 8-bit AdamW
- Parameter storage: FP32
- Compute: BF16
- Micro batch: 1
- Gradient accumulation: 16
- Learning rate: `1e-5`
- Epochs: 1
- Seed: `20260815`
- GPU: NVIDIA GeForce RTX 4070 12GB

Qwen3's exact knowledge cutoff has not been confirmed in official documentation. This experiment uses late 2024 as a working assumption, not an official specification.

### LoRA-SFT

- Framework: Transformers + PEFT
- Dataset: 200 general instruction examples
- LoRA: `r=16`, `alpha=32`, `dropout=0.05`
- Target modules: `q/k/v/o_proj`, `gate/up/down_proj`
- Learning rate: `2e-4`
- Epochs: 3
- Per-device batch: 4
- Gradient accumulation: 4
- Effective batch: 16
- Sequence length: 512
- Seed: 42
- Loss: prompt tokens masked; answer and EOS tokens only

## Repository layout

```text
.
├── ai_coding_agent_cpt_data/
│   ├── QA_Dataset/          # provisional 200-row QA set
│   ├── SFT_Dataset/         # 200 general SFT examples
│   └── scripts/             # collection, preprocessing, validation, judge
├── cpt_training/
│   ├── configs/             # CPT configuration
│   ├── scripts/             # CPT, SFT, and QA implementations
│   ├── src/                 # shared utilities
│   └── results/published/   # compact Git-tracked artifacts
├── confirm_ui/              # answer and judge-result inspection UI
├── docs/                    # public experiment reports
├── notes/                   # internal notes, excluded from Git
└── scripts/                 # LoRA-SFT shell wrappers
```

## Reproduction

### 1. Requirements

- Ubuntu 24.04
- Python 3.12
- CUDA-capable NVIDIA GPU
- NVIDIA driver and `nvidia-smi`
- [`uv`](https://docs.astral.sh/uv/)
- Disk space for environments, model caches, and checkpoints

### 2. Create environments

```bash
./ai_coding_agent_cpt_data/setup_env.sh
./cpt_training/setup_transformers_env.sh
./cpt_training/setup_unsloth_env.sh
```

Data collection, Transformers training, and Unsloth training use separate virtual environments.

### 3. Collect official web documentation

```bash
cd ai_coding_agent_cpt_data
../.venv-data/bin/python scripts/scrape_docs_spider.py
cd ..
```

Raw page bodies are written to `ai_coding_agent_cpt_data/data/scraped_documents.jsonl` and excluded from Git for size and redistribution-scope reasons.

### 4. Preprocess and pack

```bash
./cpt_training/run_preprocess.sh
```

This performs cleaning, forces matched QA-source documents into CPT train, keeps them out of the held-out test split, and packs tokens under `artifacts/datasets/cpt_v1/`. The QA rows themselves are not training examples.

### 5. CPT smoke test

```bash
./cpt_training/run_compare_all.sh smoke
```

### 6. Full CPT comparison

```bash
./cpt_training/run_compare_all.sh full --confirm-full
```

Results are written to `cpt_training/results/current/compare_full/`; checkpoints are written to `checkpoints/{transformers,unsloth}/current/`.

### 7. Validate SFT data and train LoRA adapters

```bash
python3 ai_coding_agent_cpt_data/scripts/verify_sft_leakage.py
./scripts/run_sft_base.sh --smoke-test
./scripts/run_sft_all.sh --confirm-full
```

The adapters are written to `checkpoints/sft/*/current/`.

### 8. Generate QA answers for all six conditions

```bash
./cpt_training/run_qa_evaluation.sh full --model-set 6
```

Outputs overwrite `cpt_training/results/current/qa/`.

### 9. Optional external judge and local UI

Only when using the external judge, place the key in an untracked `.env` file:

```text
GEMINI_API_KEY=your_api_key
```

```bash
./cpt_training/run_gemini_judge.sh
./confirm_ui/run_ui.sh
```

The UI uses `results/current/qa/` when local answers exist; otherwise it displays the 1,200 Git-tracked answers under `results/published/sft_qa_ablation/`. The external judge is optional, and no judge score is included in the published artifacts.

## Reproducibility boundaries

The repository records source code, configuration, model revision, dependency versions, seeds, and input/output hashes. Bit-for-bit reproduction is not guaranteed because:

- source websites change;
- raw scraped page bodies are excluded from Git;
- CUDA kernels and GPU hardware differ;
- downloader and cache state differ;
- external API models change.

The reference raw input had SHA-256 `3299966311cadfb01cc09f69fa953a5396205c45b5b98d87c6f066bd8cc1983b`. Exact data reproduction requires that same raw JSONL. A normal rerun creates a new snapshot from the current official pages.

## Excluded from Git

- `.env` files and API keys;
- virtual environments and model caches;
- raw scraped page bodies;
- packed datasets;
- CPT checkpoints, LoRA adapters, and GGUF files;
- mutable artifacts under `results/current/`;
- internal notes under `notes/`;
- unvalidated local analysis helpers.

Compact metrics, metadata, all 1,200 per-model answers, and the combined QA output are published under `cpt_training/results/published/`.

