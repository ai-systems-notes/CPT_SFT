#!/usr/bin/env python3
"""Run a tiny, full-parameter causal-LM training smoke test.

This script intentionally uses synthetic text. It verifies the training stack,
VRAM usage, optimizer step, and actual parameter updates without consuming the
held-out evaluation corpus or the future CPT training corpus.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import bitsandbytes as bnb
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B-Base")
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--output", type=Path, default=Path("artifacts/smoke_full_cpt.json"))
    return parser.parse_args()


def mib(value: int) -> float:
    return round(value / 2**20, 2)


def make_smoke_tokens(tokenizer: AutoTokenizer, sequence_length: int) -> torch.Tensor:
    text = (
        "This is synthetic text for a continued-pretraining smoke test. "
        "The purpose is to verify causal language modeling, gradient flow, "
        "optimizer state allocation, and parameter updates. "
    )
    token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if not token_ids:
        raise RuntimeError("The tokenizer returned no tokens.")
    repeats = (sequence_length + len(token_ids) - 1) // len(token_ids)
    return torch.tensor((token_ids * repeats)[:sequence_length], dtype=torch.long).unsqueeze(0)


def select_probe_parameter(model: torch.nn.Module) -> tuple[str, torch.nn.Parameter]:
    for name, parameter in model.named_parameters():
        if parameter.requires_grad and parameter.ndim >= 2 and "embed_tokens" not in name:
            return name, parameter
    raise RuntimeError("No trainable probe parameter was found.")


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("This GPU does not support BF16 training.")
    if args.steps < 1 or args.sequence_length < 2:
        raise ValueError("steps must be >= 1 and sequence-length must be >= 2")

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)

    load_started = time.perf_counter()
    # Keep FP32 master parameters so small CPT updates are not rounded away in
    # BF16 weights. Forward/backward compute uses BF16 autocast below.
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.float32,
        trust_remote_code=False,
    ).cuda()
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - load_started

    input_ids = make_smoke_tokens(tokenizer, args.sequence_length).cuda()
    labels = input_ids.clone()
    optimizer = bnb.optim.AdamW8bit(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.1,
        min_8bit_size=4096,
    )

    probe_name, probe = select_probe_parameter(model)
    probe_before = probe.detach().flatten()[:4096].float().cpu().clone()

    losses: list[float] = []
    step_seconds: list[float] = []
    for _ in range(args.steps):
        step_started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model(input_ids=input_ids, labels=labels, use_cache=False)
            loss = output.loss
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        losses.append(float(loss.detach().cpu()))
        step_seconds.append(time.perf_counter() - step_started)

    probe_after = probe.detach().flatten()[:4096].float().cpu()
    max_abs_probe_delta = float((probe_after - probe_before).abs().max())
    parameter_update_verified = max_abs_probe_delta > 0.0

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameter_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    total_step_seconds = sum(step_seconds)
    result = {
        "status": "ok" if parameter_update_verified else "failed_no_parameter_update",
        "model": args.model,
        "training_mode": "full_parameter_causal_lm",
        "parameter_dtype": str(next(model.parameters()).dtype),
        "compute_dtype": "torch.bfloat16",
        "optimizer": "bitsandbytes.optim.AdamW8bit",
        "parameter_count": parameter_count,
        "trainable_parameter_count": trainable_parameter_count,
        "trainable_ratio": trainable_parameter_count / parameter_count,
        "sequence_length": args.sequence_length,
        "batch_size": 1,
        "steps": args.steps,
        "learning_rate": args.learning_rate,
        "losses": losses,
        "step_seconds": step_seconds,
        "tokens_per_second": (args.sequence_length * args.steps) / total_step_seconds,
        "load_seconds": load_seconds,
        "wall_seconds": time.perf_counter() - started,
        "peak_allocated_mib": mib(torch.cuda.max_memory_allocated()),
        "peak_reserved_mib": mib(torch.cuda.max_memory_reserved()),
        "probe_parameter": probe_name,
        "max_abs_probe_delta": max_abs_probe_delta,
        "parameter_update_verified": parameter_update_verified,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "transformers": transformers.__version__,
            "bitsandbytes": bnb.__version__,
            "gpu": torch.cuda.get_device_name(0),
            "gpu_total_mib": mib(torch.cuda.get_device_properties(0).total_memory),
            "bf16_supported": torch.cuda.is_bf16_supported(),
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not parameter_update_verified:
        raise RuntimeError("The optimizer step did not change the probe parameter.")


if __name__ == "__main__":
    main()
