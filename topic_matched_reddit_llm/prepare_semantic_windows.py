"""Build exact-token Human–LLM analysis windows from completed simulations."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Iterable

import tiktoken

from model_providers import MODEL_SPECS, get_model_spec
from protocol import (
    CHUNK_TOKENS,
    CHUNKS_PER_WINDOW,
    CONTINUATION_TOKENS,
    MAX_TOKENS_PER_GENERATION,
    REFEREE_MODEL,
    SIMULATION_SEED,
    TEMPERATURE,
    TOKENIZER_NAME,
    WINDOW_TOKENS,
    WINDOWS_PER_TRAJECTORY,
)

ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-dir", type=Path, default=ROOT / "outputs" / "selection")
    parser.add_argument("--runs-dir", type=Path, default=ROOT / "outputs" / "runs")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "analysis_inputs")
    parser.add_argument("--models", nargs="+", choices=tuple(MODEL_SPECS), default=list(MODEL_SPECS))
    parser.add_argument("--allow-incomplete", action="store_true", help="Skip missing model/thread pairs instead of failing.")
    return parser.parse_args()


def hash_ids(ids: list[int]) -> str:
    return hashlib.sha256(",".join(map(str, ids)).encode("ascii")).hexdigest()


def read_selection(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows.sort(key=lambda row: int(row.get("selected_rank") or 10**9))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def validate_status(status: dict, *, model_key: str, seed: str) -> None:
    spec = get_model_spec(model_key)
    expected = {
        "status": "complete",
        "model_key": model_key,
        "model": spec.model_id,
        "provider": spec.provider,
        "seed": seed,
        "tokenizer": TOKENIZER_NAME,
        "continuation_token_budget": CONTINUATION_TOKENS,
        "temperature": TEMPERATURE,
        "max_tokens_per_generation": MAX_TOKENS_PER_GENERATION,
        "referee_model": REFEREE_MODEL,
        "simulation_random_seed": SIMULATION_SEED,
    }
    for key, value in expected.items():
        if status.get(key) != value:
            raise RuntimeError(f"Protocol mismatch for {model_key}/{status.get('thread_id')}: {key}={status.get(key)!r}, expected {value!r}")


def main() -> None:
    args = parse_args()
    selection_manifest = args.selection_dir / "selection_manifest.csv"
    if not selection_manifest.exists():
        raise SystemExit(f"Missing selection manifest: {selection_manifest}")
    selections = read_selection(selection_manifest)
    encoder = tiktoken.get_encoding(TOKENIZER_NAME)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    chunk_rows, window_rows, manifest_rows, skipped = [], [], [], []

    for model_key in list(dict.fromkeys(args.models)):
        spec = get_model_spec(model_key)
        for selection in selections:
            thread_id, seed = selection["thread_id"], selection["title"]
            run_dir = args.runs_dir / model_key / thread_id
            status_path = run_dir / "status.json"
            ids_path = run_dir / f"llm_analysis_stream_{CONTINUATION_TOKENS}.token_ids.json"
            if not status_path.exists() or not ids_path.exists():
                skipped.append((model_key, thread_id, "missing completed run"))
                continue
            status = json.loads(status_path.read_text(encoding="utf-8"))
            try:
                validate_status(status, model_key=model_key, seed=seed)
            except Exception as exc:
                if args.allow_incomplete:
                    skipped.append((model_key, thread_id, str(exc)))
                    continue
                raise
            human_path = args.selection_dir / "selected_human" / thread_id / f"human_stream_{CONTINUATION_TOKENS}.token_ids.json"
            human_ids = [int(value) for value in json.loads(human_path.read_text(encoding="utf-8"))]
            llm_ids = [int(value) for value in json.loads(ids_path.read_text(encoding="utf-8"))]
            if len(human_ids) != CONTINUATION_TOKENS or len(llm_ids) != CONTINUATION_TOKENS:
                raise RuntimeError(f"Exact-token mismatch for {model_key}/{thread_id}")

            for source, ids in (("human", human_ids), ("llm", llm_ids)):
                base = {
                    "model_key": model_key,
                    "model_label": spec.label,
                    "provider": spec.provider,
                    "model_id": spec.model_id,
                    "thread_id": thread_id,
                    "seed": seed,
                    "source": source,
                }
                for chunk_index in range(CONTINUATION_TOKENS // CHUNK_TOKENS):
                    start = chunk_index * CHUNK_TOKENS
                    chunk_ids = ids[start : start + CHUNK_TOKENS]
                    chunk_rows.append({
                        **base,
                        "chunk_index": chunk_index + 1,
                        "start_token_1based": start + 1,
                        "end_token_1based": start + CHUNK_TOKENS,
                        "token_ids": chunk_ids,
                        "text": encoder.decode(chunk_ids),
                        "token_ids_sha256": hash_ids(chunk_ids),
                    })
                for window_index in range(WINDOWS_PER_TRAJECTORY):
                    start = window_index * WINDOW_TOKENS
                    window_ids = ids[start : start + WINDOW_TOKENS]
                    window_rows.append({
                        **base,
                        "window_index": window_index + 1,
                        "start_token_1based": start + 1,
                        "end_token_1based": start + WINDOW_TOKENS,
                        "token_ids": window_ids,
                        "text": encoder.decode(window_ids),
                        "token_ids_sha256": hash_ids(window_ids),
                    })
                manifest_rows.append({
                    **base,
                    "continuation_tokens": len(ids),
                    "chunks": CONTINUATION_TOKENS // CHUNK_TOKENS,
                    "chunk_tokens": CHUNK_TOKENS,
                    "windows": WINDOWS_PER_TRAJECTORY,
                    "window_tokens": WINDOW_TOKENS,
                    "full_token_ids_sha256": hash_ids(ids),
                })

    expected_pairs = len(list(dict.fromkeys(args.models))) * len(selections)
    included_pairs = len(manifest_rows) // 2
    if skipped and not args.allow_incomplete:
        detail = "; ".join(f"{model}/{thread}: {reason}" for model, thread, reason in skipped)
        raise RuntimeError(f"Missing/incomplete runs: {detail}")
    if not included_pairs:
        raise SystemExit("No complete model/thread pairs were found.")

    write_jsonl(output_dir / "matched_chunks.jsonl", chunk_rows)
    write_jsonl(output_dir / "matched_windows.jsonl", window_rows)
    fields = list(manifest_rows[0])
    with (output_dir / "analysis_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(manifest_rows)
    (output_dir / "analysis_protocol.json").write_text(json.dumps({
        "analysis_name": "topic_matched_reddit_llm_exact_token_v1",
        "tokenizer": TOKENIZER_NAME,
        "continuation_tokens": CONTINUATION_TOKENS,
        "chunk_tokens": CHUNK_TOKENS,
        "chunks_per_window": CHUNKS_PER_WINDOW,
        "window_tokens": WINDOW_TOKENS,
        "windows_per_trajectory": WINDOWS_PER_TRAJECTORY,
        "models": list(dict.fromkeys(args.models)),
        "selected_threads": len(selections),
        "expected_model_thread_pairs": expected_pairs,
        "included_model_thread_pairs": included_pairs,
        "skipped": skipped,
    }, indent=2), encoding="utf-8")
    print(f"Built {included_pairs} model/thread pair(s): {len(chunk_rows)} chunks and {len(window_rows)} windows.")
    print(f"Each source: {CONTINUATION_TOKENS} tokens = {CONTINUATION_TOKENS // CHUNK_TOKENS} x {CHUNK_TOKENS}-token chunks = {WINDOWS_PER_TRAJECTORY} x {WINDOW_TOKENS}-token windows.")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
