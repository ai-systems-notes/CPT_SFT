#!/usr/bin/env python3
"""Generate closed-book QA answers for plain checkpoints and PEFT SFT adapters."""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path
from typing import Any

import peft
import torch
import transformers
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = REPO_ROOT / "ai_coding_agent_cpt_data/QA_Dataset/eval_qa_combined.jsonl"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "cpt_training/results/current/qa"
DEFAULT_REVISION = "da87bfb608c14b7cf20ba1ce41287e8de496c0cd"
PROMPT_TEMPLATE = (
    "Answer in one short concise sentence (under 100 characters):\n"
    "Question: {question}\n"
    "Answer:"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--model", action="append", dest="models", metavar="KEY=PATH")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--require-model-count", type=int)
    args = parser.parse_args()
    if args.max_new_tokens <= 0 or args.batch_size <= 0:
        parser.error("--max-new-tokens and --batch-size must be positive")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    return args


def portable_path(value: str | Path) -> str:
    path = Path(value).expanduser()
    if not path.is_absolute():
        candidate = (REPO_ROOT / path).resolve()
        if candidate.exists():
            path = candidate
        elif not path.exists():
            return str(value)
    else:
        path = path.resolve()
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def resolve_reference(value: str) -> str:
    path = Path(value).expanduser()
    if path.is_absolute():
        return str(path.resolve())
    repo_path = (REPO_ROOT / path).resolve()
    if repo_path.exists():
        return str(repo_path)
    return value


def parse_model_specs(values: list[str] | None) -> dict[str, str]:
    if not values:
        return {
            "base": "Qwen/Qwen3-0.6B-Base",
            "transformers_cpt": "checkpoints/transformers/current",
            "unsloth_cpt": "checkpoints/unsloth/current",
            "base_sft": "checkpoints/sft/base/current",
            "transformers_cpt_sft": "checkpoints/sft/transformers_cpt/current",
            "unsloth_cpt_sft": "checkpoints/sft/unsloth_cpt/current",
        }
    parsed: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Invalid model specification {value!r}; expected KEY=PATH")
        key, path = value.split("=", 1)
        key, path = key.strip(), path.strip()
        if not key or not path or key in parsed:
            raise ValueError(f"Invalid or duplicate model specification {value!r}")
        parsed[key] = path
    return parsed


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not row.get("id") or not row.get("question"):
                raise ValueError(f"Missing id/question at {path}:{line_number}")
            rows.append(row)
    if not rows:
        raise ValueError(f"Dataset is empty: {path}")
    return rows


def write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def load_model_and_tokenizer(
    model_reference: str, revision: str
) -> tuple[Any, Any, dict[str, Any]]:
    resolved_reference = resolve_reference(model_reference)
    model_path = Path(resolved_reference)
    is_local = model_path.is_dir()
    is_adapter = is_local and (model_path / "adapter_config.json").is_file()
    base_reference = resolved_reference
    adapter_metadata: dict[str, Any] = {}
    if is_adapter:
        metadata_path = model_path / "sft_metadata.json"
        if metadata_path.is_file():
            adapter_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            base_reference = resolve_reference(
                adapter_metadata.get("source_model")
                or adapter_metadata.get("base_model_path")
                or resolved_reference
            )
        else:
            adapter_config = json.loads((model_path / "adapter_config.json").read_text(encoding="utf-8"))
            base_reference = resolve_reference(adapter_config["base_model_name_or_path"])
        if base_reference == resolved_reference:
            raise ValueError(f"Adapter does not identify a distinct source model: {model_path}")

    base_is_local = Path(base_reference).is_dir()
    load_kwargs: dict[str, Any] = {"trust_remote_code": True}
    if not base_is_local:
        load_kwargs["revision"] = revision
    tokenizer_source = resolved_reference if is_adapter and (model_path / "tokenizer_config.json").is_file() else base_reference
    tokenizer_kwargs = {"trust_remote_code": True}
    if not Path(tokenizer_source).is_dir():
        tokenizer_kwargs["revision"] = revision

    print(f"Tokenizer: {portable_path(tokenizer_source)}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, **tokenizer_kwargs)
    if tokenizer.eos_token is None:
        raise ValueError(f"Tokenizer has no EOS token: {tokenizer_source}")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(f"Base model: {portable_path(base_reference)}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_reference,
        dtype=dtype,
        device_map="auto",
        **load_kwargs,
    )
    if is_adapter:
        print(f"LoRA adapter: {portable_path(resolved_reference)}")
        model = PeftModel.from_pretrained(base_model, resolved_reference, is_trainable=False)
    else:
        model = base_model
    model.eval()
    return model, tokenizer, {
        "reference": portable_path(resolved_reference),
        "is_peft_adapter": is_adapter,
        "source_model": portable_path(base_reference),
        "adapter_metadata": adapter_metadata,
        "dtype": str(dtype).removeprefix("torch."),
    }


def generate_answers(
    model: Any,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    max_new_tokens: int,
    batch_size: int,
) -> tuple[list[str], int]:
    answers: list[str] = []
    generated_tokens = 0
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        prompts = [PROMPT_TEMPLATE.format(question=row["question"]) for row in batch]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
        input_width = inputs["input_ids"].shape[1]
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        for output in outputs:
            generated = output[input_width:]
            generated_tokens += int(generated.numel())
            answers.append(tokenizer.decode(generated, skip_special_tokens=True).strip())
        completed = min(start + len(batch), len(rows))
        if completed % 20 == 0 or completed == len(rows):
            print(f"  generated {completed}/{len(rows)}", flush=True)
    return answers, generated_tokens


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU is required for QA generation")
    dataset_path = args.dataset.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    rows = read_jsonl(dataset_path)
    if args.limit is not None:
        rows = rows[: args.limit]
    models = parse_model_specs(args.models)
    if args.require_model_count is not None and len(models) != args.require_model_count:
        raise ValueError(f"Expected {args.require_model_count} model specs, got {len(models)}")

    for reference in models.values():
        resolved = resolve_reference(reference)
        if "/" not in resolved and not Path(resolved).exists():
            raise FileNotFoundError(resolved)
        if reference.startswith("checkpoints/") and not Path(resolved).is_dir():
            raise FileNotFoundError(resolved)

    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    answers_by_model: dict[str, list[str]] = {}
    runtime: dict[str, dict[str, Any]] = {}
    model_metadata: dict[str, dict[str, Any]] = {}

    for alias, reference in models.items():
        print(f"\n--- {alias}: {portable_path(resolve_reference(reference))} ---")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        load_started = time.perf_counter()
        model, tokenizer, details = load_model_and_tokenizer(reference, args.revision)
        torch.cuda.synchronize()
        load_seconds = time.perf_counter() - load_started
        generation_started = time.perf_counter()
        answers, generated_tokens = generate_answers(
            model, tokenizer, rows, args.max_new_tokens, args.batch_size
        )
        torch.cuda.synchronize()
        generation_seconds = time.perf_counter() - generation_started
        answers_by_model[alias] = answers
        model_metadata[alias] = details
        runtime[alias] = {
            "model_load_seconds": load_seconds,
            "generation_seconds": generation_seconds,
            "generated_tokens": generated_tokens,
            "generated_tokens_per_second": generated_tokens / generation_seconds if generation_seconds else 0.0,
            "torch_peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
            "torch_peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
        }
        individual_rows = []
        for row, answer in zip(rows, answers):
            individual_rows.append(
                {
                    "id": row["id"],
                    "question": row["question"],
                    "reference": row.get("answer") or row.get("reference_answer", ""),
                    "model": alias,
                    "answer": answer,
                }
            )
        write_jsonl_atomic(output_dir / f"answers_{alias}.jsonl", individual_rows)
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()

    combined_rows = []
    for index, row in enumerate(rows):
        combined_rows.append(
            {
                key: row.get(key)
                for key in ("id", "category", "topic", "question", "answer", "keywords", "source_url")
                if key in row
            }
            | {"answers": {alias: answers[index] for alias, answers in answers_by_model.items()}}
        )
    write_jsonl_atomic(output_dir / "answers_combined.jsonl", combined_rows)
    metadata = {
        "status": "answers_generated_not_judged",
        "dataset": portable_path(dataset_path),
        "item_count": len(rows),
        "models": {alias: portable_path(resolve_reference(path)) for alias, path in models.items()},
        "model_metadata": model_metadata,
        "generation": {
            "prompt_format": PROMPT_TEMPLATE,
            "decoding": "greedy",
            "dtype": "bf16" if torch.cuda.is_bf16_supported() else "fp16",
            "max_new_tokens": args.max_new_tokens,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "revision": args.revision,
        },
        "runtime": runtime,
        "environment": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "peft": peft.__version__,
            "gpu": torch.cuda.get_device_name(0),
        },
    }
    temporary = output_dir / ".metadata.json.tmp"
    temporary.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, output_dir / "metadata.json")
    print(f"\nResult: {portable_path(output_dir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
