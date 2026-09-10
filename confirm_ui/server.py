#!/usr/bin/env python3
"""Serve the QA comparison UI from the fixed current result directory."""

from __future__ import annotations

import argparse
import http.server
import json
import os
import socketserver
from pathlib import Path
from urllib.parse import urlparse

DEFAULT_PORT = 8000
BASE_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = BASE_DIR.parent.resolve()
CURRENT_RESULTS_DIR = PROJECT_ROOT / "cpt_training/results/current/qa"
PUBLISHED_RESULTS_DIR = (
    PROJECT_ROOT / "cpt_training/results/published/sft_qa_ablation"
)
DEFAULT_RESULTS_DIR = (
    CURRENT_RESULTS_DIR
    if (CURRENT_RESULTS_DIR / "answers_combined.jsonl").is_file()
    else PUBLISHED_RESULTS_DIR
)
REPORT_FILE = PROJECT_ROOT / "docs/experiment_report_ja.md"
RESULTS_DIR = DEFAULT_RESULTS_DIR


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def get_combined_data() -> dict:
    metadata = load_json(RESULTS_DIR / "metadata.json")
    if not metadata:
        metadata = load_json(RESULTS_DIR / "qa_generation_metadata.json")
    answers_combined = load_jsonl(RESULTS_DIR / "answers_combined.jsonl")
    blind_inputs = load_jsonl(RESULTS_DIR / "gemini_judge_input_blind.jsonl")
    candidate_maps = load_jsonl(RESULTS_DIR / "gemini_candidate_map.jsonl")
    judge_results = load_jsonl(RESULTS_DIR / "gemini_judge_results.jsonl")

    blind_map = {item["id"]: item for item in blind_inputs}
    candidate_map = {
        item["id"]: item.get("candidate_to_model", {}) for item in candidate_maps
    }
    judge_map = {item["id"]: item for item in judge_results}

    items = []
    for item in answers_combined:
        item_id = item["id"]
        items.append(
            {
                "id": item_id,
                "category": item.get("category", ""),
                "topic": item.get("topic", ""),
                "question": item.get("question", ""),
                "answer": item.get("answer", ""),
                "keywords": item.get("keywords", []),
                "source_url": item.get("source_url", ""),
                "answers": item.get("answers", {}),
                "blind_candidates": blind_map.get(item_id, {}).get("candidates", {}),
                "candidate_to_model": candidate_map.get(item_id, {}),
                "judge": judge_map.get(item_id),
            }
        )
    return {"metadata": metadata, "items": items, "total_items": len(items)}


def get_report_content() -> str:
    if not REPORT_FILE.exists():
        return "Experiment report not found."
    return REPORT_FILE.read_text(encoding="utf-8")


class RequestHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(BASE_DIR), **kwargs)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/api/data":
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(
                json.dumps(get_combined_data(), ensure_ascii=False).encode("utf-8")
            )
            return
        if path == "/api/report":
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(get_report_content().encode("utf-8"))
            return
        super().do_GET()


def main() -> None:
    global RESULTS_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()
    RESULTS_DIR = args.results_dir.resolve()
    if not RESULTS_DIR.exists():
        raise SystemExit(f"QA result directory not found: {RESULTS_DIR}")

    os.chdir(BASE_DIR)
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer((args.host, args.port), RequestHandler) as httpd:
        print(f"QA Comparison Server: http://{args.host}:{args.port}")
        print(f"QA results: {RESULTS_DIR}")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped.")


if __name__ == "__main__":
    main()
