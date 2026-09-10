#!/usr/bin/env python3
"""Validate and summarize one Transformers/Unsloth CPT result pair."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transformers", type=Path, required=True)
    parser.add_argument("--unsloth", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def row(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "framework": metrics["framework"],
        "mode": metrics["mode"],
        "status": metrics["status"],
        "optimizer_steps": metrics["training"]["optimizer_steps"],
        "target_tokens": metrics["training"]["target_tokens"],
        "measured_tokens_per_second": metrics["performance"]["measured_tokens_per_second"],
        "training_seconds": metrics["performance"]["training_seconds"],
        "end_to_end_wall_seconds": metrics["performance"]["end_to_end_wall_seconds"],
        "torch_peak_allocated_mib": metrics["memory"]["torch_peak_allocated_mib"],
        "torch_peak_reserved_mib": metrics["memory"]["torch_peak_reserved_mib"],
        "nvidia_smi_process_peak_mib": metrics["memory"]["nvidia_smi_process_peak_mib"],
        "process_peak_rss_mib": metrics["memory"]["process_peak_rss_mib"],
        "baseline_test_loss": metrics["evaluation"]["baseline"]["loss"],
        "post_test_loss": metrics["evaluation"]["post_training"]["loss"],
        "test_loss_delta": metrics["evaluation"]["loss_delta"],
        "parameter_update_verified": metrics["training"]["parameter_update_verified"],
        "torch": metrics["environment"]["torch"],
        "transformers": metrics["environment"]["transformers"],
        "unsloth": metrics["environment"].get("unsloth"),
    }


def validate_pair(transformers: dict[str, Any], unsloth: dict[str, Any]) -> list[str]:
    checks = {
        "config_sha256": transformers["config_sha256"] == unsloth["config_sha256"],
        "mode": transformers["mode"] == unsloth["mode"],
        "train_sha256": (
            transformers["dataset"]["train_sha256"] == unsloth["dataset"]["train_sha256"]
        ),
        "test_sha256": (
            transformers["dataset"]["test_sha256"] == unsloth["dataset"]["test_sha256"]
        ),
        "optimizer_steps": (
            transformers["training"]["optimizer_steps"]
            == unsloth["training"]["optimizer_steps"]
        ),
        "target_tokens": (
            transformers["training"]["target_tokens"] == unsloth["training"]["target_tokens"]
        ),
        "parameter_count": (
            transformers["model"]["parameter_count"] == unsloth["model"]["parameter_count"]
        ),
        "parameter_dtype": (
            transformers["model"]["parameter_dtype"] == unsloth["model"]["parameter_dtype"]
        ),
        "compute_dtype": (
            transformers["model"]["compute_dtype"] == unsloth["model"]["compute_dtype"]
        ),
    }
    return [name for name, passed in checks.items() if not passed]


def write_markdown(path: Path, rows: list[dict[str, Any]], comparison: dict[str, Any]) -> None:
    lines = [
        "# Transformers vs Unsloth CPT comparison", "", f"Mode: `{rows[0]['mode']}`", "",
        "| Framework | token/s | peak allocated MiB | nvidia-smi peak MiB | train sec | held-out loss (before → after) |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in rows:
        lines.append(
            f"| {item['framework']} | {item['measured_tokens_per_second']:.2f} | "
            f"{item['torch_peak_allocated_mib']:.2f} | {item['nvidia_smi_process_peak_mib']:.2f} | "
            f"{item['training_seconds']:.2f} | {item['baseline_test_loss']:.6f} → "
            f"{item['post_test_loss']:.6f} |"
        )
    lines.extend([
        "",
        f"- Unsloth throughput ratio: {comparison['unsloth_throughput_ratio']:.4f}x",
        f"- Unsloth peak allocated difference: {comparison['unsloth_peak_allocated_difference_mib']:.2f} MiB",
        f"- Formal benchmark: {'yes' if comparison['formal_benchmark'] else 'no'}",
        "",
    ])
    if not comparison["formal_benchmark"]:
        lines.extend([
            "A single result pair is provisional. Use three repeated runs and an aggregate median for the final benchmark.",
            "",
        ])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    transformers = read_json(args.transformers)
    unsloth = read_json(args.unsloth)
    mismatches = validate_pair(transformers, unsloth)
    if mismatches:
        raise ValueError(f"Results are not comparable: {', '.join(mismatches)}")
    if transformers["status"] != "ok" or unsloth["status"] != "ok":
        raise ValueError("Both runs must have status=ok")

    rows = [row(transformers), row(unsloth)]
    baseline_speed = rows[0]["measured_tokens_per_second"]
    unsloth_speed = rows[1]["measured_tokens_per_second"]
    comparison = {
        "full_training_run": rows[0]["mode"] == "full",
        "formal_benchmark": False,
        "unsloth_throughput_ratio": unsloth_speed / baseline_speed,
        "unsloth_peak_allocated_difference_mib": (
            rows[1]["torch_peak_allocated_mib"] - rows[0]["torch_peak_allocated_mib"]
        ),
        "unsloth_peak_allocated_ratio": (
            rows[1]["torch_peak_allocated_mib"] / rows[0]["torch_peak_allocated_mib"]
        ),
    }
    result = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "pair_validation": "ok",
        "rows": rows,
        "comparison": comparison,
        "source_metrics": {
            "transformers": str(args.transformers.resolve()),
            "unsloth": str(args.unsloth.resolve()),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "comparison.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_markdown(args.output_dir / "comparison.md", rows, comparison)
    print(f"comparison={args.output_dir.resolve() / 'comparison.md'}")


if __name__ == "__main__":
    main()
