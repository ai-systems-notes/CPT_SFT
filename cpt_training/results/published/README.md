# Published experiment artifacts

Small, Git-friendly artifacts from the Qwen3-0.6B RTX 4070 experiment.

- `full_run_1/`: first full Transformers/PyTorch vs Unsloth CPT run.
- `full_run_2/`: repeated full CPT run.
- `sft_qa_ablation/`: three LoRA-SFT runs and six-condition QA generation.
- `qa70_dual_judge/`: the rebuilt 70-question evaluation, 12 conditions, 840 answers, two independent judges, and the regenerated analysis behind the write-up.

Model weights, LoRA adapters, GGUF files, packed datasets, caches, and raw scraped page bodies are not included.

`sft_qa_ablation/` predates the question-set rebuild: its QA references have not completed a full primary-source audit, and it carries no external judge score. It reproduces generation and formatting/lexical metrics only.

`qa70_dual_judge/` supersedes it for factual-accuracy claims. Every question there records its source URL, the source document's SHA-256 and an evidence excerpt, and the answers are scored by two independent judges. Effects are reported as paired per-item deltas with bootstrap confidence intervals rather than as a ranking table; see that directory's README for what the numbers do and do not support.

