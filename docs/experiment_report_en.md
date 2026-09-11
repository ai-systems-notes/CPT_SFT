# Experiment report: Qwen3-0.6B CPT + LoRA-SFT

## Evaluation scope

Completed:

- full-parameter CPT of Qwen3-0.6B-Base with Transformers/PyTorch and Unsloth;
- two full CPT comparison runs;
- identical PEFT LoRA-SFT from Base and both CPT checkpoints;
- 200 QA rows from each of six conditions, totaling 1,200 saved answers;
- loss, throughput, GPU memory, answer length, and keyword-recall collection.

Completed since:

- the question set rebuilt to 70 primary-source-backed items, each recording a source URL, the source document's SHA-256 and an evidence excerpt;
- primary-source validation of the references (70/70 pass automated validation; the source for `seen_codex_013` and `seen_codex_014`, `codex-manual.md`, is a real official page that preprocessing dropped as an aggregate, so it never entered the training corpus);
- external LLM-as-a-Judge, two independent runs (`gpt-5.6-terra` ×3 and `gemini-3.5-flash-lite`) over 12 conditions × 70 questions = 840 answers;
- paired effect sizes with 95% bootstrap confidence intervals.

Not completed:

- naming a single "best model" on factual accuracy: at 70 questions and one seed, differences below roughly 0.6 points are not resolvable;
- multi-seed estimation of training variance;
- replication at the 10B-30B scale.

## CPT measurements

| Run | Framework | tokens/s | peak allocated | `nvidia-smi` peak | train time | held-out loss |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | Transformers/PyTorch | 4,754.61 | 8,474 MiB | 9,450 MiB | 321.18 s | 2.5403 → 2.1455 |
| 1 | Unsloth | 4,761.86 | 7,673 MiB | 8,060 MiB | 320.55 s | 2.5402 → 2.1455 |
| 2 | Transformers/PyTorch | 4,981.27 | 8,474 MiB | 9,450 MiB | 305.72 s | 2.5403 → 2.1456 |
| 2 | Unsloth | 4,778.68 | 7,673 MiB | 8,060 MiB | 319.12 s | 2.5402 → 2.1455 |

Held-out loss decreased in both implementations. Unsloth used about 801 MiB less peak allocated memory and about 1,390 MiB less `nvidia-smi` peak memory. The speed ordering changed between runs, so no consistent speed advantage was observed on this RTX 4070 setup.

## LoRA-SFT measurements

| Source model | loss | train time | tokens/s | peak allocated |
| --- | ---: | ---: | ---: | ---: |
| Base | 0.6172 | 10.36 s | 2,871.27 | 2,519 MiB |
| Transformers-CPT | 0.6135 | 10.24 s | 2,905.33 | 2,519 MiB |
| Unsloth-CPT | 0.6126 | 10.12 s | 2,939.11 | 2,519 MiB |

All conditions used the same 200 examples, three epochs, seed 42, and 39 optimizer steps. This stage used PEFT LoRA, not full fine-tuning.

## QA formatting and lexical metrics

| Condition | ≤100 characters | Mean characters | Keyword recall |
| --- | ---: | ---: | ---: |
| Base | 10.0% | 172.8 | 36.5% |
| Transformers-CPT | 0.5% | 991.2 | 33.5% |
| Unsloth-CPT | 2.0% | 931.0 | 33.3% |
| Base + LoRA-SFT | 65.0% | 92.3 | 21.2% |
| Transformers-CPT + LoRA-SFT | 92.0% | 79.2 | 24.8% |
| Unsloth-CPT + LoRA-SFT | 94.5% | 78.5 | 25.3% |

Rows where `Question:` reappeared after the first 20 answer characters:

| Condition | rows |
| --- | ---: |
| Base | 0 / 200 |
| Transformers-CPT | 145 / 200 |
| Unsloth-CPT | 130 / 200 |
| Base + LoRA-SFT | 0 / 200 |
| Transformers-CPT + LoRA-SFT | 0 / 200 |
| Unsloth-CPT + LoRA-SFT | 0 / 200 |

## Qualitative observations from the latest 1,200 answers

These are agent observations of the saved outputs, not correctness judgments.

1. CPT-only answers averaged more than 900 characters and frequently continued by generating additional `Question:` blocks.
2. After LoRA-SFT, none of the three SFT conditions triggered this `Question:` continuation detector, and short-answer instruction adherence increased substantially.
3. Transformers-CPT-SFT and Unsloth-CPT-SFT produced exactly the same answer string on 144 of 200 rows, indicating very similar post-SFT behavior.
4. Base-SFT also became shorter, so concise response behavior must be attributed to SFT rather than CPT alone.
5. Some concise answers contain unverified commands or configuration names. Concision does not establish correctness.

## QA-set limitations

The run used 200 rows but only 83 exact unique question strings. There are 30 duplicate-question groups, containing 147 rows in total. The data therefore cannot be treated as a 200-independent-question accuracy benchmark.

The current QA rows and 1,200 answers are published unchanged so the completed run can be reproduced and audited. A future corrected QA set should be treated as a separate experiment with regenerated outputs, not as an overwrite of this run.

## Current conclusions

- Full CPT of a 0.6B model was feasible on an RTX 4070 12GB.
- Held-out loss decreased after CPT.
- Unsloth reduced VRAM in this setup, but did not show a consistent speed advantage.
- LoRA-SFT substantially increased short-answer format adherence.
- Factual accuracy on coding-agent knowledge remains unevaluated.

