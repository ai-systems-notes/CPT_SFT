"""Shared data, measurement, and reporting helpers for CPT experiments."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import psutil
import torch


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_path(repo_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def load_packed(path: Path, expected_sequence_length: int) -> dict[str, torch.Tensor]:
    packed = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(packed, dict) or set(packed) != {"input_ids", "attention_mask"}:
        raise ValueError(f"Unexpected packed dataset schema: {path}")
    input_ids = packed["input_ids"]
    attention_mask = packed["attention_mask"]
    if input_ids.ndim != 2 or tuple(input_ids.shape) != tuple(attention_mask.shape):
        raise ValueError(f"Invalid packed tensor shapes: {path}")
    if input_ids.shape[1] != expected_sequence_length:
        raise ValueError(
            f"Expected sequence length {expected_sequence_length}, got {input_ids.shape[1]}"
        )
    return {"input_ids": input_ids, "attention_mask": attention_mask}


def make_order(num_blocks: int, seed: int, shuffle: bool) -> torch.Tensor:
    if not shuffle:
        return torch.arange(num_blocks)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return torch.randperm(num_blocks, generator=generator)


def iter_micro_batches(
    packed: dict[str, torch.Tensor],
    order: torch.Tensor,
    micro_batch_size: int,
    limit_blocks: int | None = None,
) -> Iterator[dict[str, torch.Tensor]]:
    selected = order if limit_blocks is None else order[:limit_blocks]
    for start in range(0, len(selected), micro_batch_size):
        indices = selected[start : start + micro_batch_size]
        yield {
            "input_ids": packed["input_ids"].index_select(0, indices).long(),
            "attention_mask": packed["attention_mask"].index_select(0, indices),
        }


def prepare_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    input_ids = batch["input_ids"].to(device, non_blocking=True)
    attention_mask = batch["attention_mask"].to(device, non_blocking=True)
    labels = input_ids.clone()
    labels.masked_fill_(~attention_mask, -100)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def target_token_count(attention_mask: torch.Tensor) -> int:
    return int(attention_mask[:, 1:].sum().item())


@torch.no_grad()
def evaluate_causal_lm(
    model: torch.nn.Module,
    packed: dict[str, torch.Tensor],
    device: torch.device,
    compute_dtype: torch.dtype,
    micro_batch_size: int,
    max_blocks: int | None,
) -> dict[str, Any]:
    model.eval()
    blocks = len(packed["input_ids"]) if max_blocks is None else min(
        max_blocks, len(packed["input_ids"])
    )
    total_weighted_loss = 0.0
    total_tokens = 0
    started = time.perf_counter()
    order = torch.arange(len(packed["input_ids"]))
    for cpu_batch in iter_micro_batches(packed, order, micro_batch_size, blocks):
        batch = prepare_batch(cpu_batch, device)
        tokens = target_token_count(batch["attention_mask"])
        with torch.autocast(device_type="cuda", dtype=compute_dtype):
            loss = model(**batch, use_cache=False).loss
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite evaluation loss detected.")
        total_weighted_loss += float(loss) * tokens
        total_tokens += tokens
    torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    mean_loss = total_weighted_loss / total_tokens
    model.train()
    return {
        "blocks": blocks,
        "target_tokens": total_tokens,
        "loss": mean_loss,
        "perplexity": math.exp(min(mean_loss, 20.0)),
        "seconds": seconds,
        "tokens_per_second": total_tokens / seconds,
    }


def select_probe_parameter(model: torch.nn.Module) -> tuple[str, torch.nn.Parameter]:
    for name, parameter in model.named_parameters():
        if parameter.requires_grad and parameter.ndim >= 2 and "embed_tokens" not in name:
            return name, parameter
    raise RuntimeError("No trainable matrix parameter found.")


def tensor_probe(parameter: torch.nn.Parameter, size: int = 4096) -> torch.Tensor:
    return parameter.detach().reshape(-1)[:size].float().cpu().clone()


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = math.ceil(quantile * len(ordered)) - 1
    return ordered[max(0, min(index, len(ordered) - 1))]


def timing_summary(values: list[float]) -> dict[str, float | None]:
    return {
        "count": len(values),
        "mean_seconds": statistics.fmean(values) if values else None,
        "median_seconds": statistics.median(values) if values else None,
        "p95_seconds": percentile(values, 0.95),
    }


def directory_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def mib(value: int | float) -> float:
    return round(float(value) / 2**20, 2)


@dataclass
class HostMemorySnapshot:
    process_rss: int
    system_available: int
    swap_used: int
    nvidia_process_mib: float | None


def query_nvidia_process_mib(pid: int) -> float | None:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
        values = []
        for line in completed.stdout.splitlines():
            process_id, used_mib = (part.strip() for part in line.split(",", maxsplit=1))
            if int(process_id) == pid:
                values.append(float(used_mib))
        return sum(values) if values else None
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


class ResourceSampler:
    def __init__(self, interval_seconds: float = 0.25) -> None:
        self.interval_seconds = interval_seconds
        self.process = psutil.Process(os.getpid())
        self.samples: list[HostMemorySnapshot] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _sample(self) -> None:
        virtual = psutil.virtual_memory()
        swap = psutil.swap_memory()
        self.samples.append(
            HostMemorySnapshot(
                process_rss=self.process.memory_info().rss,
                system_available=virtual.available,
                swap_used=swap.used,
                nvidia_process_mib=query_nvidia_process_mib(self.process.pid),
            )
        )

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self._sample()

    def __enter__(self) -> "ResourceSampler":
        self._sample()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3)
        self._sample()

    def summary(self) -> dict[str, Any]:
        gpu_values = [s.nvidia_process_mib for s in self.samples if s.nvidia_process_mib is not None]
        return {
            "sampling_interval_seconds": self.interval_seconds,
            "sample_count": len(self.samples),
            "process_peak_rss_mib": mib(max(s.process_rss for s in self.samples)),
            "system_min_available_mib": mib(min(s.system_available for s in self.samples)),
            "swap_peak_used_mib": mib(max(s.swap_used for s in self.samples)),
            "nvidia_smi_process_peak_mib": max(gpu_values) if gpu_values else None,
        }


def base_environment() -> dict[str, Any]:
    properties = torch.cuda.get_device_properties(0)
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "gpu_total_mib": mib(properties.total_memory),
        "bf16_supported": torch.cuda.is_bf16_supported(),
    }


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
