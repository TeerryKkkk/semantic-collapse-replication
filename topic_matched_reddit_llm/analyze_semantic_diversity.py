"""Compute window-level Vendi/cosine metrics, early–late Vendi, and paired bootstrap CIs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sqlite3
import time
from typing import Dict, Iterable, Sequence

import numpy as np
from openai import OpenAI
import tiktoken

from protocol import (
    BOOTSTRAP_DRAWS,
    BOOTSTRAP_SEED,
    CHUNKS_PER_WINDOW,
    EARLY_WINDOWS,
    EMBEDDING_MODEL,
    LATE_WINDOWS,
    TOKENIZER_NAME,
    WINDOW_TOKENS,
    WINDOWS_PER_TRAJECTORY,
)

ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "outputs" / "analysis_inputs")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "semantic_analysis")
    parser.add_argument("--cache-path", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--retries", type=int, default=6)
    parser.add_argument("--bootstrap-draws", type=int, default=BOOTSTRAP_DRAWS)
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    return parser.parse_args()


def token_hash(ids: Sequence[int]) -> str:
    return hashlib.sha256(",".join(map(str, ids)).encode("ascii")).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise RuntimeError(f"Invalid JSON at {path}:{line_no}") from exc
    return rows


def write_csv(path: Path, rows: list[dict], fieldnames: Iterable[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


class EmbeddingCache:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(str(path))
        self.connection.execute(
            """CREATE TABLE IF NOT EXISTS embeddings (
                model TEXT NOT NULL,
                token_sha256 TEXT NOT NULL,
                token_count INTEGER NOT NULL,
                dim INTEGER NOT NULL,
                vector BLOB NOT NULL,
                created_unix REAL NOT NULL,
                PRIMARY KEY (model, token_sha256)
            )"""
        )
        self.connection.commit()

    def get(self, model: str, hash_value: str) -> np.ndarray | None:
        row = self.connection.execute(
            "SELECT dim, vector FROM embeddings WHERE model=? AND token_sha256=?",
            (model, hash_value),
        ).fetchone()
        if row is None:
            return None
        dim, blob = row
        vector = np.frombuffer(blob, dtype=np.float32).copy()
        if vector.size != int(dim):
            raise RuntimeError(f"Corrupt embedding cache entry: {hash_value}")
        return vector

    def put(self, model: str, hash_value: str, token_count: int, vector: Sequence[float]) -> None:
        array = np.asarray(vector, dtype=np.float32)
        self.connection.execute(
            "INSERT OR REPLACE INTO embeddings VALUES (?, ?, ?, ?, ?, ?)",
            (model, hash_value, int(token_count), int(array.size), array.tobytes(), time.time()),
        )

    def commit(self) -> None:
        self.connection.commit()

    def close(self) -> None:
        self.connection.commit(); self.connection.close()


def l2_rows(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise RuntimeError("Zero-norm embedding row.")
    return matrix / norms


def vendi_and_similarity(embeddings: np.ndarray) -> dict:
    """Normalized Vendi and mean pairwise cosine for a fixed-size window."""
    x = l2_rows(embeddings)
    m = x.shape[0]
    if m < 2:
        raise ValueError("Vendi requires at least two embeddings.")
    kernel = x @ x.T
    kernel = (kernel + kernel.T) / 2.0
    rho = kernel / float(np.trace(kernel))
    eigenvalues = np.clip(np.linalg.eigvalsh(rho), 0.0, None)
    eigenvalues /= float(eigenvalues.sum())
    positive = eigenvalues[eigenvalues > 1e-15]
    entropy = float(-np.sum(positive * np.log(positive)))
    effective_support = float(np.exp(entropy))
    upper = kernel[np.triu_indices(m, k=1)]
    return {
        "normalized_vendi": effective_support / m,
        "vendi_effective_support": effective_support,
        "vendi_entropy": entropy,
        "mean_pairwise_cosine": float(np.mean(upper)),
        "m": m,
    }


def embed_missing(catalog: Dict[str, list[int]], *, cache: EmbeddingCache, batch_size: int, retries: int) -> tuple[int, int]:
    missing = [key for key in catalog if cache.get(EMBEDDING_MODEL, key) is None]
    hits = len(catalog) - len(missing)
    if not missing:
        print(f"Embedding cache: {hits}/{len(catalog)} exact sequences reused; no API call needed.")
        return hits, 0
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is required for uncached text-embedding-3-large inputs.")
    client = OpenAI(api_key=api_key)
    made = 0
    for batch_no, start in enumerate(range(0, len(missing), batch_size), 1):
        hashes = missing[start : start + batch_size]
        inputs = [catalog[key] for key in hashes]
        response = None
        last_error = None
        for attempt in range(retries):
            try:
                response = client.embeddings.create(model=EMBEDDING_MODEL, input=inputs, encoding_format="float")
                break
            except Exception as exc:
                last_error = exc
                if attempt + 1 < retries:
                    time.sleep(min(60.0, 2**attempt + random.random()))
        if response is None:
            raise RuntimeError(f"Embedding batch {batch_no} failed") from last_error
        items = sorted(response.data, key=lambda item: int(item.index))
        if len(items) != len(hashes):
            raise RuntimeError("Embedding response length mismatch.")
        for hash_value, ids, item in zip(hashes, inputs, items):
            cache.put(EMBEDDING_MODEL, hash_value, len(ids), item.embedding)
            made += 1
        cache.commit()
        print(f"Embedded {made}/{len(missing)} new exact token sequences.")
    return hits, made


def mean_windows(rows: list[dict], field: str, indices: Sequence[int]) -> float:
    wanted = set(indices)
    values = [float(row[field]) for row in rows if int(row["window_index"]) in wanted]
    if len(values) != len(wanted):
        raise RuntimeError(f"Missing window(s) for {field}: expected {sorted(wanted)}")
    return float(np.mean(values))


def build_metrics(chunks: list[dict], *, cache: EmbeddingCache) -> tuple[list[dict], list[dict]]:
    groups: dict[tuple[str, str, str], list[dict]] = {}
    pair_meta: dict[tuple[str, str], dict] = {}
    for row in chunks:
        key = (row["model_key"], row["thread_id"], row["source"])
        groups.setdefault(key, []).append(row)
        pair_meta[(row["model_key"], row["thread_id"])] = {
            "model_key": row["model_key"], "model_label": row["model_label"],
            "provider": row["provider"], "model_id": row["model_id"],
            "thread_id": row["thread_id"], "seed": row["seed"],
        }
    window_rows: list[dict] = []
    source_summary: dict[tuple[str, str, str], dict] = {}
    for (model_key, thread_id), meta in pair_meta.items():
        for source in ("human", "llm"):
            rows = sorted(groups[(model_key, thread_id, source)], key=lambda row: int(row["chunk_index"]))
            if len(rows) != WINDOWS_PER_TRAJECTORY * CHUNKS_PER_WINDOW:
                raise RuntimeError(f"Unexpected chunk count for {model_key}/{thread_id}/{source}")
            vectors = []
            for row in rows:
                hash_value = row["token_ids_sha256"]
                vector = cache.get(EMBEDDING_MODEL, hash_value)
                if vector is None:
                    raise RuntimeError(f"Missing cached embedding: {hash_value}")
                vectors.append(vector)
            matrix = np.vstack(vectors)
            local = []
            for window_index in range(WINDOWS_PER_TRAJECTORY):
                start = window_index * CHUNKS_PER_WINDOW
                metrics = vendi_and_similarity(matrix[start : start + CHUNKS_PER_WINDOW])
                record = {
                    **meta,
                    "source": source,
                    "window_index": window_index + 1,
                    "cumulative_tokens": (window_index + 1) * WINDOW_TOKENS,
                    **metrics,
                }
                window_rows.append(record); local.append(record)
            source_summary[(model_key, thread_id, source)] = {
                **meta,
                "source": source,
                "mean_normalized_vendi": float(np.mean([row["normalized_vendi"] for row in local])),
                "mean_pairwise_cosine": float(np.mean([row["mean_pairwise_cosine"] for row in local])),
                "early_vendi": mean_windows(local, "normalized_vendi", EARLY_WINDOWS),
                "late_vendi": mean_windows(local, "normalized_vendi", LATE_WINDOWS),
            }

    thread_rows: list[dict] = []
    for (model_key, thread_id), meta in pair_meta.items():
        human = source_summary[(model_key, thread_id, "human")]
        llm = source_summary[(model_key, thread_id, "llm")]
        human_change = human["early_vendi"] - human["late_vendi"]
        llm_change = llm["early_vendi"] - llm["late_vendi"]
        thread_rows.append({
            **meta,
            "human_mean_normalized_vendi": human["mean_normalized_vendi"],
            "llm_mean_normalized_vendi": llm["mean_normalized_vendi"],
            "llm_minus_human_vendi": llm["mean_normalized_vendi"] - human["mean_normalized_vendi"],
            "human_mean_pairwise_cosine": human["mean_pairwise_cosine"],
            "llm_mean_pairwise_cosine": llm["mean_pairwise_cosine"],
            "llm_minus_human_pairwise_cosine": llm["mean_pairwise_cosine"] - human["mean_pairwise_cosine"],
            "human_early_vendi": human["early_vendi"], "human_late_vendi": human["late_vendi"],
            "human_early_minus_late": human_change,
            "llm_early_vendi": llm["early_vendi"], "llm_late_vendi": llm["late_vendi"],
            "llm_early_minus_late": llm_change,
            "llm_minus_human_early_minus_late": llm_change - human_change,
        })
    return window_rows, thread_rows


def paired_window_rows(window_rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str, int], dict[str, dict]] = {}
    for row in window_rows:
        grouped.setdefault((row["model_key"], row["thread_id"], int(row["window_index"])), {})[row["source"]] = row
    out = []
    for (model_key, thread_id, window_index), sources in sorted(grouped.items()):
        human, llm = sources["human"], sources["llm"]
        out.append({
            "model_key": model_key, "model_label": human["model_label"], "model_id": human["model_id"],
            "thread_id": thread_id, "seed": human["seed"], "window_index": window_index,
            "cumulative_tokens": human["cumulative_tokens"],
            "human_normalized_vendi": human["normalized_vendi"],
            "llm_normalized_vendi": llm["normalized_vendi"],
            "llm_minus_human_vendi": llm["normalized_vendi"] - human["normalized_vendi"],
            "human_mean_pairwise_cosine": human["mean_pairwise_cosine"],
            "llm_mean_pairwise_cosine": llm["mean_pairwise_cosine"],
            "llm_minus_human_pairwise_cosine": llm["mean_pairwise_cosine"] - human["mean_pairwise_cosine"],
        })
    return out


def build_model_summary(thread_rows: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = {}
    for row in thread_rows:
        grouped.setdefault(row["model_key"], []).append(row)
    summary = []
    for model_key, rows in grouped.items():
        mean = lambda field: float(np.mean([float(row[field]) for row in rows]))
        summary.append({
            "model_key": model_key,
            "model_label": rows[0]["model_label"],
            "model_id": rows[0]["model_id"],
            "threads": len(rows),
            "human_mean_normalized_vendi": mean("human_mean_normalized_vendi"),
            "llm_mean_normalized_vendi": mean("llm_mean_normalized_vendi"),
            "llm_minus_human_vendi": mean("llm_minus_human_vendi"),
            "human_mean_pairwise_cosine": mean("human_mean_pairwise_cosine"),
            "llm_mean_pairwise_cosine": mean("llm_mean_pairwise_cosine"),
            "llm_minus_human_pairwise_cosine": mean("llm_minus_human_pairwise_cosine"),
            "human_early_vendi": mean("human_early_vendi"),
            "human_late_vendi": mean("human_late_vendi"),
            "human_early_minus_late": mean("human_early_minus_late"),
            "llm_early_vendi": mean("llm_early_vendi"),
            "llm_late_vendi": mean("llm_late_vendi"),
            "llm_early_minus_late": mean("llm_early_minus_late"),
        })
    return sorted(summary, key=lambda row: row["model_key"])


def paired_bootstrap(thread_rows: list[dict], *, draws: int, seed: int) -> list[dict]:
    """Question-level paired bootstrap after equal weighting across model families."""
    fields = {
        "normalized_vendi": "llm_minus_human_vendi",
        "within_window_cosine": "llm_minus_human_pairwise_cosine",
        "early_minus_late_vendi": "llm_minus_human_early_minus_late",
    }
    thread_ids = sorted({row["thread_id"] for row in thread_rows})
    per_thread: dict[str, dict[str, float]] = {}
    for thread_id in thread_ids:
        rows = [row for row in thread_rows if row["thread_id"] == thread_id]
        per_thread[thread_id] = {
            metric: float(np.mean([float(row[field]) for row in rows]))
            for metric, field in fields.items()
        }
    output = []
    for metric in fields:
        # Reset the deterministic generator for each reported metric so the percentile
        # interval for one metric does not depend on the ordering of other metrics.
        rng = np.random.default_rng(seed)
        values = np.array([per_thread[thread_id][metric] for thread_id in thread_ids], dtype=float)
        estimates = np.empty(draws, dtype=float)
        for index in range(draws):
            estimates[index] = float(np.mean(rng.choice(values, size=len(values), replace=True)))
        low, high = np.percentile(estimates, [2.5, 97.5])
        output.append({
            "metric": metric,
            "estimate_llm_minus_human": float(values.mean()),
            "ci_2_5": float(low),
            "ci_97_5": float(high),
            "resampling_unit": "Reddit question",
            "model_weighting": "equal weight across four model families within question",
            "bootstrap_draws": draws,
            "bootstrap_seed": seed,
            "n_questions": len(values),
        })
    return output


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    chunks_path = input_dir / "matched_chunks.jsonl"
    protocol_path = input_dir / "analysis_protocol.json"
    if not chunks_path.exists() or not protocol_path.exists():
        raise SystemExit("Missing prepared analysis inputs. Run prepare_semantic_windows.py first.")
    encoder = tiktoken.encoding_for_model(EMBEDDING_MODEL)
    if encoder.name != TOKENIZER_NAME:
        raise RuntimeError(f"Unexpected tokenizer for {EMBEDDING_MODEL}: {encoder.name}")
    chunks = read_jsonl(chunks_path)
    catalog: Dict[str, list[int]] = {}
    for row in chunks:
        ids = [int(value) for value in row["token_ids"]]
        expected = row.get("token_ids_sha256")
        actual = token_hash(ids)
        if expected and expected != actual:
            raise RuntimeError("Token-hash mismatch in prepared inputs.")
        catalog.setdefault(actual, ids)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = args.cache_path.resolve() if args.cache_path else output_dir / "embedding_cache.sqlite3"
    cache = EmbeddingCache(cache_path)
    try:
        hits, made = embed_missing(catalog, cache=cache, batch_size=args.batch_size, retries=args.retries)
        window_rows, thread_rows = build_metrics(chunks, cache=cache)
    finally:
        cache.close()
    paired_windows = paired_window_rows(window_rows)
    model_summary = build_model_summary(thread_rows)
    bootstrap = paired_bootstrap(thread_rows, draws=args.bootstrap_draws, seed=args.bootstrap_seed)
    early_late = [{
        "model_key": row["model_key"], "model_label": row["model_label"], "model_id": row["model_id"],
        "thread_id": row["thread_id"], "seed": row["seed"],
        "human_early_vendi": row["human_early_vendi"], "human_late_vendi": row["human_late_vendi"],
        "human_early_minus_late": row["human_early_minus_late"],
        "llm_early_vendi": row["llm_early_vendi"], "llm_late_vendi": row["llm_late_vendi"],
        "llm_early_minus_late": row["llm_early_minus_late"],
        "llm_minus_human_early_minus_late": row["llm_minus_human_early_minus_late"],
    } for row in thread_rows]

    write_csv(output_dir / "window_semantic_metrics.csv", window_rows)
    write_csv(output_dir / "matched_window_comparisons.csv", paired_windows)
    write_csv(output_dir / "early_late_vendi.csv", early_late)
    write_csv(output_dir / "model_summary.csv", model_summary)
    write_csv(output_dir / "thread_summary.csv", thread_rows)
    write_csv(output_dir / "paired_bootstrap_summary.csv", bootstrap)

    lines = ["TOPIC-MATCHED REDDIT–LLM SEMANTIC SUMMARY", "=" * 42]
    for row in model_summary:
        lines.append(
            f"{row['model_label']}: Vendi Human={float(row['human_mean_normalized_vendi']):.4f}, "
            f"LLM={float(row['llm_mean_normalized_vendi']):.4f}; cosine Human={float(row['human_mean_pairwise_cosine']):.4f}, "
            f"LLM={float(row['llm_mean_pairwise_cosine']):.4f}; Early-Late LLM={float(row['llm_early_minus_late']):+.4f}"
        )
    lines.append("")
    lines.append("Question-level paired bootstrap (LLM - Human):")
    for row in bootstrap:
        lines.append(
            f"  {row['metric']}: {float(row['estimate_llm_minus_human']):+.4f} "
            f"[95% CI {float(row['ci_2_5']):+.4f}, {float(row['ci_97_5']):+.4f}]"
        )
    (output_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output_dir / "analysis_metadata.json").write_text(json.dumps({
        "embedding_model": EMBEDDING_MODEL,
        "tokenizer": TOKENIZER_NAME,
        "embedding_input": "fixed-length token-ID arrays",
        "vendi_definition": "cosine Gram kernel -> trace normalization -> exp(entropy) / m",
        "m": CHUNKS_PER_WINDOW,
        "bootstrap_draws": args.bootstrap_draws,
        "bootstrap_seed": args.bootstrap_seed,
        "cache_hits": hits,
        "new_embeddings": made,
    }, indent=2), encoding="utf-8")
    print((output_dir / "summary.txt").read_text(encoding="utf-8"))
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
