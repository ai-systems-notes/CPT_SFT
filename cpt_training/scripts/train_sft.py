#!/usr/bin/env python3
"""Train a PEFT LoRA SFT adapter for the six-condition ablation study."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

import datasets
import peft
import torch
import transformers
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = REPO_ROOT / "ai_coding_agent_cpt_data/SFT_Dataset/sft_general_train.jsonl"
FOUNDATION_REVISION = "da87bfb608c14b7cf20ba1ce41287e8de496c0cd"
SHORT_PROMPT_TEMPLATE = (
    "Answer in one short concise sentence (under 100 characters):\n"
    "Question: {instruction}\n"
    "Answer:"
)
COMPLETE_PROMPT_TEMPLATE = (
    "Answer directly in 1 to 3 concise sentences, preferably within 300 characters. "
    "Preserve exact commands, option names, parameter values, and required conditions. "
    "Do not generate another question.\n"
    "Question: {instruction}\n"
    "Answer:"
)
V3_PROMPT_TEMPLATE = (
    "Answer directly and completely using only supported information.\n"
    "Use one concise sentence when sufficient and up to three sentences when necessary.\n"
    "Preserve exact commands, option names, paths, parameter values, and required conditions.\n"
    "Do not invent missing details, repeat the question, or continue with another Q&A.\n"
    "Question: {instruction}\n"
    "Answer:"
)
PROMPT_TEMPLATES = {
    "short": SHORT_PROMPT_TEMPLATE,
    "complete": COMPLETE_PROMPT_TEMPLATE,
    "v3": V3_PROMPT_TEMPLATE,
}


class TokenCountingTrainer(Trainer):
    """Count non-padding tokens in microbatches actually consumed by Trainer."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._observed_token_counts: list[torch.Tensor] = []

    def training_step(
        self,
        model: torch.nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        num_items_in_batch: torch.Tensor | int | None = None,
    ) -> torch.Tensor:
        attention_mask = inputs.get("attention_mask")
        if isinstance(attention_mask, torch.Tensor):
            self._observed_token_counts.append(attention_mask.detach().sum())
        return super().training_step(model, inputs, num_items_in_batch)

    def observed_tokens(self) -> int:
        return sum(int(value.cpu()) for value in self._observed_token_counts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--revision", default=FOUNDATION_REVISION)
    parser.add_argument("--sft-dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--prompt-style", choices=tuple(PROMPT_TEMPLATES), default="short")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--confirm-full", action="store_true")
    args = parser.parse_args()
    if not args.smoke_test and not args.confirm_full:
        parser.error("Full SFT requires --confirm-full; use --smoke-test for a one-step check")
    for name in ("epochs", "batch_size", "grad_accum", "max_seq_length"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.learning_rate <= 0:
        parser.error("--learning-rate must be positive")
    return args


def portable_path(value: str | Path) -> str:
    path = Path(value).expanduser()
    if not path.is_absolute():
        candidate = (REPO_ROOT / path).resolve()
        if candidate.exists():
            path = candidate
    elif path.exists():
        path = path.resolve()
    else:
        return str(value)
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def load_rows(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if set(row) != {"instruction", "input", "output"}:
                raise ValueError(f"Invalid SFT schema at line {line_number}")
            if row["input"] != "":
                raise ValueError(f"SFT input must be empty at line {line_number}")
            rows.append(row)
    if len(rows) != 200:
        raise ValueError(f"Expected 200 SFT rows, got {len(rows)}")
    return rows


def preprocess_batch(
    examples: dict[str, list[str]],
    tokenizer: Any,
    max_seq_length: int,
    prompt_template: str,
) -> dict[str, list[list[int]]]:
    input_ids_list: list[list[int]] = []
    labels_list: list[list[int]] = []
    for instruction, output in zip(examples["instruction"], examples["output"]):
        prompt = prompt_template.format(instruction=instruction) + " "
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        full_ids = tokenizer.encode(prompt + output + tokenizer.eos_token, add_special_tokens=False)
        full_ids = full_ids[:max_seq_length]
        prompt_length = min(len(prompt_ids), len(full_ids))
        labels = [-100] * prompt_length + full_ids[prompt_length:]
        if not any(label != -100 for label in labels):
            raise ValueError("max sequence length removed every answer token")
        input_ids_list.append(full_ids)
        labels_list.append(labels)
    return {"input_ids": input_ids_list, "labels": labels_list}


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU is required for this experiment")

    dataset_path = args.sft_dataset.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not dataset_path.is_file():
        raise FileNotFoundError(dataset_path)
    model_candidate = Path(args.model_path).expanduser()
    model_is_local = model_candidate.exists()
    model_source = str(model_candidate.resolve()) if model_is_local else args.model_path
    if model_candidate.is_absolute() and not model_is_local:
        raise FileNotFoundError(model_candidate)

    print("=== SFT LoRA training ===")
    print(f"Model: {portable_path(model_source)}")
    print(f"Dataset: {portable_path(dataset_path)}")
    print(f"Output: {portable_path(output_dir)}")
    print(f"Mode: {'smoke (1 optimizer step)' if args.smoke_test else 'full'}")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.reset_peak_memory_stats()
    dataset_sha256 = hashlib.sha256(dataset_path.read_bytes()).hexdigest()

    load_kwargs: dict[str, Any] = {"trust_remote_code": True}
    if not model_is_local:
        load_kwargs["revision"] = args.revision
    tokenizer = AutoTokenizer.from_pretrained(model_source, **load_kwargs)
    if tokenizer.eos_token is None:
        raise ValueError("Tokenizer has no EOS token")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(model_source, dtype=dtype, **load_kwargs)
    loaded_revision = getattr(model.config, "_commit_hash", None)
    model.config.use_cache = False

    lora_r = 16
    lora_alpha = 32
    model = get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=0.05,
            target_modules=(
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ),
        ),
    )
    trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    model.print_trainable_parameters()

    rows = load_rows(dataset_path)
    prompt_template = PROMPT_TEMPLATES[args.prompt_style]
    raw_dataset = Dataset.from_list(rows)
    tokenized_dataset = raw_dataset.map(
        lambda batch: preprocess_batch(
            batch, tokenizer, args.max_seq_length, prompt_template
        ),
        batched=True,
        remove_columns=raw_dataset.column_names,
        desc="Tokenizing SFT dataset",
    )
    max_steps = 1 if args.smoke_test else -1
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        num_train_epochs=1 if args.smoke_test else args.epochs,
        max_steps=max_steps,
        logging_steps=1 if args.smoke_test else 10,
        save_strategy="no",
        seed=args.seed,
        data_seed=args.seed,
        fp16=dtype == torch.float16,
        bf16=dtype == torch.bfloat16,
        report_to="none",
    )
    trainer = TokenCountingTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            pad_to_multiple_of=8,
            label_pad_token_id=-100,
            return_tensors="pt",
        ),
        processing_class=tokenizer,
    )

    torch.cuda.synchronize()
    started = time.perf_counter()
    train_result = trainer.train()
    torch.cuda.synchronize()
    runtime = time.perf_counter() - started
    observed_tokens = trainer.observed_tokens()
    tokens_per_second = observed_tokens / runtime if runtime else 0.0
    peak_allocated_mib = torch.cuda.max_memory_allocated() / 2**20
    peak_reserved_mib = torch.cuda.max_memory_reserved() / 2**20

    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    adapter_config_path = output_dir / "adapter_config.json"
    if adapter_config_path.is_file():
        adapter_config = json.loads(adapter_config_path.read_text(encoding="utf-8"))
        adapter_config["base_model_name_or_path"] = portable_path(model_source)
        adapter_config_path.write_text(json.dumps(adapter_config, indent=2) + "\n", encoding="utf-8")

    metadata = {
        "artifact_type": "peft_lora_adapter",
        "source_model": portable_path(model_source),
        "source_model_revision": loaded_revision,
        "foundation_model_revision": FOUNDATION_REVISION,
        "sft_dataset": portable_path(dataset_path),
        "sft_dataset_sha256": dataset_sha256,
        "prompt_style": args.prompt_style,
        "prompt_template": prompt_template,
        "hyperparameters": {
            "learning_rate": args.learning_rate,
            "epochs": args.epochs,
            "per_device_batch_size": args.batch_size,
            "gradient_accumulation_steps": args.grad_accum,
            "effective_batch_size": args.batch_size * args.grad_accum,
            "seed": args.seed,
            "max_sequence_length": args.max_seq_length,
            "lora_r": lora_r,
            "lora_alpha": lora_alpha,
            "lora_dropout": 0.05,
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "peft": peft.__version__,
            "datasets": datasets.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "dtype": str(dtype).removeprefix("torch."),
        },
    }
    metrics = {
        "status": "ok",
        "is_smoke_test": args.smoke_test,
        "train_loss": train_result.training_loss,
        "optimizer_steps": trainer.state.global_step,
        "observed_nonpadding_tokens": observed_tokens,
        "tokens_per_second": tokens_per_second,
        "training_seconds": runtime,
        "torch_peak_allocated_mib": peak_allocated_mib,
        "torch_peak_reserved_mib": peak_reserved_mib,
        "trainable_parameters": trainable_parameters,
        "total_parameters_with_adapter": total_parameters,
    }
    (output_dir / "sft_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "sft_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "README.md").write_text(
        "# SFT LoRA Adapter\n\n"
        f"- Source model: `{metadata['source_model']}`\n"
        f"- Dataset SHA-256: `{dataset_sha256}`\n"
        f"- Smoke test: `{args.smoke_test}`\n"
        f"- Loss: `{train_result.training_loss:.6f}`\n"
        f"- Throughput: `{tokens_per_second:.2f}` non-padding tokens/s\n"
        f"- Peak allocated VRAM: `{peak_allocated_mib:.2f}` MiB\n",
        encoding="utf-8",
    )
    print(f"status=ok")
    print(f"metrics={output_dir / 'sft_metrics.json'}")
    print(f"loss={train_result.training_loss:.6f}")
    print(f"tokens_per_second={tokens_per_second:.2f}")
    print(f"torch_peak_allocated_mib={peak_allocated_mib:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
