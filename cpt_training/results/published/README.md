# Published experiment artifacts

Small, Git-friendly artifacts from the Qwen3-0.6B RTX 4070 experiment.

- `full_run_1/`: first full Transformers/PyTorch vs Unsloth CPT run.
- `full_run_2/`: repeated full CPT run.
- `sft_qa_ablation/`: three LoRA-SFT runs and six-condition QA generation.

Model weights, LoRA adapters, GGUF files, packed datasets, caches, and raw scraped page bodies are not included.

The QA references have not completed a full primary-source audit. QA artifacts reproduce generation and formatting/lexical metrics; they do not establish factual accuracy or a best model. No external Judge score is included.

