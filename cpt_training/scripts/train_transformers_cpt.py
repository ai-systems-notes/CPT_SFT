#!/usr/bin/env python3
"""Full-parameter CPT baseline using Transformers and a direct PyTorch loop."""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import bitsandbytes as bnb
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
TRAINING_DIR = SCRIPT_DIR.parent
REPO_ROOT = TRAINING_DIR.parent
sys.path.insert(0, str(TRAINING_DIR / "src"))

from cpt_common import (  # noqa: E402
    ResourceSampler,
    base_environment,
    directory_size_bytes,
    evaluate_causal_lm,
    iter_micro_batches,
    load_json,
    load_packed,
    make_order,
    mib,
    prepare_batch,
    resolve_path,
    select_probe_parameter,
    sha256_file,
    target_token_count,
    tensor_probe,
    timing_summary,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=TRAINING_DIR / "configs/qwen3_0.6b_seq1024.json"
    )
    parser.add_argument("--mode", choices=("smoke", "pilot", "full"), default="smoke")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path)
    return parser.parse_args()


def compute_max_steps(config: dict[str, Any], mode: dict[str, Any], train_blocks: int) -> int:
    explicit = mode.get("max_optimizer_steps")
    if explicit is not None:
        return int(explicit)
    train = config["training"]
    blocks_per_step = train["micro_batch_size"] * train["gradient_accumulation_steps"]
    steps_per_epoch = train_blocks // blocks_per_step
    return steps_per_epoch * int(mode["epochs"])


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 is not supported by this GPU.")

    config_path = args.config.resolve()
    config = load_json(config_path)
    mode_config = config["modes"][args.mode]
    model_config = config["model"]
    train_config = config["training"]
    dataset_config = config["dataset"]
    sequence_length = int(dataset_config["sequence_length"])
    train_path = resolve_path(REPO_ROOT, dataset_config["train"])
    test_path = resolve_path(REPO_ROOT, dataset_config["test"])
    metadata_path = resolve_path(REPO_ROOT, dataset_config["metadata"])
    output_path = args.output.resolve()
    checkpoint_dir = args.checkpoint_dir.resolve() if args.checkpoint_dir else None

    torch.manual_seed(train_config["seed"])
    torch.cuda.manual_seed_all(train_config["seed"])
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    device = torch.device("cuda")
    compute_dtype = torch.bfloat16
    run_started = time.perf_counter()
    generated_at = datetime.now(timezone.utc).isoformat()

    train_packed = load_packed(train_path, sequence_length)
    test_packed = load_packed(test_path, sequence_length)
    packed_metadata = load_json(metadata_path)
    max_optimizer_steps = compute_max_steps(config, mode_config, len(train_packed["input_ids"]))
    required_blocks = (
        max_optimizer_steps
        * train_config["gradient_accumulation_steps"]
        * train_config["micro_batch_size"]
    )
    if required_blocks > len(train_packed["input_ids"]):
        raise ValueError(
            f"Mode requires {required_blocks} blocks, but the train dataset has "
            f"{len(train_packed['input_ids'])}."
        )

    with ResourceSampler() as sampler:
        load_started = time.perf_counter()
        tokenizer = AutoTokenizer.from_pretrained(
            model_config["name"], revision=model_config["revision"],
            local_files_only=model_config["local_files_only"], trust_remote_code=False,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_config["name"], revision=model_config["revision"],
            local_files_only=model_config["local_files_only"], dtype=torch.float32,
            trust_remote_code=False,
        ).to(device)
        model.config.use_cache = False
        if model_config["gradient_checkpointing"]:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        model.train()
        torch.cuda.synchronize()
        load_seconds = time.perf_counter() - load_started

        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        trainable_parameter_count = sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        )
        if parameter_count != trainable_parameter_count:
            raise RuntimeError("This experiment requires every model parameter to be trainable.")

        optimizer = bnb.optim.AdamW8bit(
            model.parameters(), lr=train_config["learning_rate"],
            betas=tuple(train_config["betas"]), weight_decay=train_config["weight_decay"],
            min_8bit_size=4096,
        )
        probe_name, probe_parameter = select_probe_parameter(model)
        probe_before = tensor_probe(probe_parameter)
        baseline_eval = evaluate_causal_lm(
            model, test_packed, device, compute_dtype, train_config["micro_batch_size"],
            mode_config["eval_max_blocks"],
        )

        order = make_order(
            len(train_packed["input_ids"]), train_config["seed"], train_config["shuffle"]
        )[:required_blocks]
        micro_batches = iter_micro_batches(train_packed, order, train_config["micro_batch_size"])
        accumulation = int(train_config["gradient_accumulation_steps"])
        optimizer_step_seconds: list[float] = []
        optimizer_step_losses: list[float] = []
        optimizer_step_grad_norms: list[float] = []
        optimizer_step_target_tokens: list[int] = []
        all_losses_finite = True
        training_started = time.perf_counter()

        for _optimizer_step in range(max_optimizer_steps):
            step_started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            accumulated_loss = 0.0
            accumulated_tokens = 0
            for _ in range(accumulation):
                batch = prepare_batch(next(micro_batches), device)
                tokens = target_token_count(batch["attention_mask"])
                with torch.autocast(device_type="cuda", dtype=compute_dtype):
                    loss = model(**batch, use_cache=False).loss
                if not torch.isfinite(loss):
                    all_losses_finite = False
                    raise RuntimeError("Non-finite training loss detected.")
                (loss / accumulation).backward()
                accumulated_loss += float(loss.detach()) * tokens
                accumulated_tokens += tokens

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), train_config["max_grad_norm"])
            if not torch.isfinite(grad_norm):
                raise RuntimeError("Non-finite gradient norm detected.")
            optimizer.step()
            torch.cuda.synchronize()
            optimizer_step_seconds.append(time.perf_counter() - step_started)
            optimizer_step_losses.append(accumulated_loss / accumulated_tokens)
            optimizer_step_grad_norms.append(float(grad_norm))
            optimizer_step_target_tokens.append(accumulated_tokens)

        training_seconds = time.perf_counter() - training_started
        probe_after = tensor_probe(probe_parameter)
        max_abs_probe_delta = float((probe_after - probe_before).abs().max())
        parameter_update_verified = max_abs_probe_delta > 0.0
        post_eval = evaluate_causal_lm(
            model, test_packed, device, compute_dtype, train_config["micro_batch_size"],
            mode_config["eval_max_blocks"],
        )

        checkpoint_seconds = 0.0
        checkpoint_size_bytes = 0
        if mode_config["save_checkpoint"]:
            if checkpoint_dir is None:
                raise ValueError("--checkpoint-dir is required in full mode.")
            save_started = time.perf_counter()
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(checkpoint_dir, safe_serialization=True)
            tokenizer.save_pretrained(checkpoint_dir)
            checkpoint_seconds = time.perf_counter() - save_started
            checkpoint_size_bytes = directory_size_bytes(checkpoint_dir)

        torch.cuda.synchronize()
        peak_allocated_mib = mib(torch.cuda.max_memory_allocated())
        peak_reserved_mib = mib(torch.cuda.max_memory_reserved())

    warmup_steps = min(int(mode_config["timing_warmup_optimizer_steps"]), len(optimizer_step_seconds))
    measured_seconds = optimizer_step_seconds[warmup_steps:]
    measured_tokens = optimizer_step_target_tokens[warmup_steps:]
    total_target_tokens = sum(optimizer_step_target_tokens)
    measured_target_tokens = sum(measured_tokens)
    measured_wall_seconds = sum(measured_seconds)
    status = "ok" if parameter_update_verified and all_losses_finite else "failed"
    result = {
        "schema_version": 1, "status": status, "framework": "transformers_pytorch",
        "mode": args.mode, "generated_at_utc": generated_at,
        "config_path": str(config_path), "config_sha256": sha256_file(config_path),
        "config": config,
        "dataset": {
            "train_path": str(train_path), "train_sha256": sha256_file(train_path),
            "test_path": str(test_path), "test_sha256": sha256_file(test_path),
            "train_blocks_available": len(train_packed["input_ids"]),
            "test_blocks_available": len(test_packed["input_ids"]),
            "packed_metadata": packed_metadata,
        },
        "model": {
            "name": model_config["name"], "revision": model_config["revision"],
            "parameter_count": parameter_count,
            "trainable_parameter_count": trainable_parameter_count,
            "trainable_ratio": trainable_parameter_count / parameter_count,
            "parameter_dtype": str(next(model.parameters()).dtype),
            "compute_dtype": str(compute_dtype),
            "gradient_checkpointing": bool(model_config["gradient_checkpointing"]),
        },
        "training": {
            "optimizer_steps": max_optimizer_steps,
            "micro_steps": max_optimizer_steps * accumulation,
            "target_tokens": total_target_tokens,
            "optimizer_step_losses": optimizer_step_losses,
            "optimizer_step_grad_norms": optimizer_step_grad_norms,
            "optimizer_step_seconds": optimizer_step_seconds,
            "optimizer_step_target_tokens": optimizer_step_target_tokens,
            "first_loss": optimizer_step_losses[0], "last_loss": optimizer_step_losses[-1],
            "all_losses_finite": all_losses_finite, "probe_parameter": probe_name,
            "max_abs_probe_delta": max_abs_probe_delta,
            "parameter_update_verified": parameter_update_verified,
        },
        "evaluation": {
            "baseline": baseline_eval, "post_training": post_eval,
            "loss_delta": post_eval["loss"] - baseline_eval["loss"],
        },
        "performance": {
            "model_load_seconds": load_seconds, "training_seconds": training_seconds,
            "end_to_end_wall_seconds": time.perf_counter() - run_started,
            "all_training_tokens_per_second": total_target_tokens / training_seconds,
            "timing_warmup_optimizer_steps": warmup_steps,
            "measured_target_tokens": measured_target_tokens,
            "measured_seconds": measured_wall_seconds,
            "measured_tokens_per_second": (
                measured_target_tokens / measured_wall_seconds if measured_wall_seconds else None
            ),
            "optimizer_step_timing": timing_summary(measured_seconds),
        },
        "memory": {
            "torch_peak_allocated_mib": peak_allocated_mib,
            "torch_peak_reserved_mib": peak_reserved_mib, **sampler.summary(),
        },
        "checkpoint": {
            "saved": bool(mode_config["save_checkpoint"]),
            "path": str(checkpoint_dir) if checkpoint_dir else None,
            "save_seconds": checkpoint_seconds, "size_bytes": checkpoint_size_bytes,
        },
        "environment": {
            **base_environment(), "transformers": transformers.__version__,
            "bitsandbytes": bnb.__version__, "pid": os.getpid(),
        },
        "notes": [
            "Speed excludes model load, evaluation, and checkpoint save time.",
            "Measured speed excludes the configured optimizer-step warmup.",
            "FP32 master parameters are used with BF16 autocast compute.",
            "nvidia-smi memory is sampled and may miss very short peaks; PyTorch peaks are authoritative."
        ],
    }
    write_json(output_path, result)
    print(f"status={status}")
    print(f"metrics={output_path}")
    print(f"loss={optimizer_step_losses[0]:.6f} -> {optimizer_step_losses[-1]:.6f}")
    print(f"heldout_loss={baseline_eval['loss']:.6f} -> {post_eval['loss']:.6f}")
    print(f"tokens_per_second={result['performance']['measured_tokens_per_second']}")
    print(f"torch_peak_allocated_mib={peak_allocated_mib}")
    print(f"torch_peak_reserved_mib={peak_reserved_mib}")
    if status != "ok":
        raise RuntimeError("CPT validation failed; inspect the metrics JSON.")


if __name__ == "__main__":
    main()
