#!/usr/bin/env python3
"""Run the shared full-parameter CPT loop with an Unsloth model backend."""

from __future__ import annotations

# Unsloth must patch the stack before transformers is imported by the shared runner.
import unsloth
from unsloth import FastLanguageModel

import json
import sys
from pathlib import Path
from typing import Any

import torch
import unsloth_zoo

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import train_transformers_cpt as runner  # noqa: E402


class UnslothModelFactory:
    """Adapter for the AutoModelForCausalLM call used by the shared runner."""

    @staticmethod
    def from_pretrained(model_name: str, **kwargs: Any) -> torch.nn.Module:
        model, _tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            revision=kwargs.get("revision"),
            local_files_only=kwargs.get("local_files_only", False),
            max_seq_length=1024,
            dtype=torch.bfloat16,
            load_in_4bit=False,
            load_in_8bit=False,
            load_in_16bit=False,
            full_finetuning=True,
            fix_tokenizer=False,
            use_gradient_checkpointing="unsloth",
            float32_mixed_precision=True,
            use_exact_model_name=True,
            trust_remote_code=False,
            random_state=20260815,
            disable_log_stats=True,
        )
        # The shared runner enables native GC after loading. Unsloth GC is already
        # active, so keep that implementation for the comparison.
        model.gradient_checkpointing_enable = lambda *args, **kwargs: None
        return model


def output_path_from_argv() -> Path:
    try:
        position = sys.argv.index("--output")
        return Path(sys.argv[position + 1]).resolve()
    except (ValueError, IndexError) as error:
        raise ValueError("--output is required") from error


def annotate_metrics(output_path: Path) -> None:
    metrics = json.loads(output_path.read_text(encoding="utf-8"))
    metrics["framework"] = "unsloth"
    metrics["environment"]["unsloth"] = unsloth.__version__
    metrics["environment"]["unsloth_zoo"] = unsloth_zoo.__version__
    metrics["notes"].extend(
        [
            "Unsloth full_finetuning=True; 4-bit, 8-bit, and 16-bit LoRA loading are disabled.",
            "Unsloth float32_mixed_precision=True aligns FP32 parameter storage with the baseline.",
            "Unsloth gradient checkpointing is used instead of native non-reentrant checkpointing.",
        ]
    )
    runner.write_json(output_path, metrics)


if __name__ == "__main__":
    output_path = output_path_from_argv()
    runner.AutoModelForCausalLM = UnslothModelFactory
    runner.main()
    annotate_metrics(output_path)
