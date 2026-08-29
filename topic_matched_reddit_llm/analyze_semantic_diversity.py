"""Compute trajectory lexical/anchor metrics plus local Vendi/cosine analyses."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import sqlite3
import time
from typing import Dict, Iterable, Sequence
import unicodedata

import numpy as np
from openai import OpenAI
import tiktoken

from protocol import (
    BOOTSTRAP_DRAWS,
    BOOTSTRAP_SEED,
    CHUNK_TOKENS,
    CHUNKS_PER_WINDOW,
    CONTINUATION_TOKENS,
    EARLY_WINDOWS,
    EMBEDDING_MODEL,
    LATE_WINDOWS,
    TOKENIZER_NAME,
    TRAJECTORY_WINDOW_TOKENS,
    TRAJECTORY_WINDOWS_PER_TRAJECTORY,
    WINDOW_TOKENS,
    WINDOWS_PER_TRAJECTORY,
)

ROOT = Path(__file__).resolve().parent

LEXICAL_TOKEN_RE = re.compile(r"[a-z]+")
LEXICAL_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "is", "are", "was", "were", "be",
    "been", "being", "for", "on", "with", "as", "at", "by", "from", "that", "this",
    "it", "its", "into", "about", "than",
}


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


def split_trajectory_token_ids(ids: Sequence[int]) -> list[list[int]]:
    """Split one exact continuation into consecutive 200-token trajectory windows."""
    values = [int(value) for value in ids]
    if len(values) != CONTINUATION_TOKENS:
        raise ValueError(f"Expected {CONTINUATION_TOKENS} trajectory tokens, found {len(values)}")
    windows = [
        values[start : start + TRAJECTORY_WINDOW_TOKENS]
        for start in range(0, len(values), TRAJECTORY_WINDOW_TOKENS)
    ]
    if len(windows) != TRAJECTORY_WINDOWS_PER_TRAJECTORY or any(
        len(window) != TRAJECTORY_WINDOW_TOKENS for window in windows
    ):
        raise RuntimeError("Failed to construct complete 200-token trajectory windows.")
    return windows


def build_trajectory_windows(chunks: list[dict], encoder) -> list[dict]:
    """Recombine authoritative 100-token chunks and split at the trajectory scale."""
    groups: dict[tuple[str, str, str], list[dict]] = {}
    for row in chunks:
        groups.setdefault((row["model_key"], row["thread_id"], row["source"]), []).append(row)
    output: list[dict] = []
    expected_chunks = CONTINUATION_TOKENS // CHUNK_TOKENS
    for key, rows in groups.items():
        rows.sort(key=lambda row: int(row["chunk_index"]))
        if [int(row["chunk_index"]) for row in rows] != list(range(1, expected_chunks + 1)):
            raise RuntimeError(f"Incomplete chunk trajectory: {key}")
        full_ids: list[int] = []
        for index, row in enumerate(rows):
            ids = [int(value) for value in row["token_ids"]]
            if len(ids) != CHUNK_TOKENS:
                raise RuntimeError(f"Unexpected chunk length for {key}/{index + 1}")
            expected_start = index * CHUNK_TOKENS + 1
            expected_end = (index + 1) * CHUNK_TOKENS
            if int(row["start_token_1based"]) != expected_start or int(row["end_token_1based"]) != expected_end:
                raise RuntimeError(f"Gap or overlap in prepared chunks for {key}/{index + 1}")
            full_ids.extend(ids)
        meta = rows[0]
        for index, ids in enumerate(split_trajectory_token_ids(full_ids), 1):
            start = (index - 1) * TRAJECTORY_WINDOW_TOKENS
            output.append({
                "model_key": meta["model_key"], "model_label": meta["model_label"],
                "provider": meta["provider"], "model_id": meta["model_id"],
                "thread_id": meta["thread_id"], "seed": meta["seed"], "source": meta["source"],
                "window_index": index,
                "start_token_1based": start + 1,
                "end_token_1based": start + TRAJECTORY_WINDOW_TOKENS,
                "token_ids": ids,
                "text": encoder.decode(ids),
                "token_ids_sha256": token_hash(ids),
            })
    return output


def lexical_unigrams(text: str) -> list[str]:
    """Apply the repository's paper lexical-token definition."""
    normalized = unicodedata.normalize("NFKD", text)
    normalized = normalized.encode("ascii", "ignore").decode("ascii").lower()
    return [
        token for token in LEXICAL_TOKEN_RE.findall(normalized)
        if 2 <= len(token) <= 20 and token not in LEXICAL_STOPWORDS
    ]


def cumulative_unique_unigram_counts(texts: Sequence[str]) -> list[int]:
    seen: set[str] = set()
    counts = []
    for text in texts:
        seen.update(lexical_unigrams(text))
        counts.append(len(seen))
    return counts


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


def anchored_cosine_distances(embeddings: np.ndarray) -> np.ndarray:
    """Cosine distance from every row to the first row; the anchor is exactly zero."""
    x = l2_rows(embeddings)
    distances = 1.0 - (x @ x[0])
    distances[0] = 0.0
    return distances


def vendi_and_similarity(embeddings: np.ndarray) -> dict:
    """Normalized Vendi and within-interval pairwise cosine for fixed 100-token chunks."""
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
    similarity = float(np.mean(upper))
    return {
        "normalized_vendi": effective_support / m,
        "vendi_effective_support": effective_support,
        "vendi_entropy": entropy,
        # Expose both cosine similarity and the corresponding cosine-distance fields.
        "mean_pairwise_cosine": similarity,
        "mean_pairwise_cosine_similarity": similarity,
        "mean_pairwise_cosine_distance": 1.0 - similarity,
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


def build_trajectory_metrics(
    windows: list[dict], *, cache: EmbeddingCache
) -> tuple[list[dict], list[dict]]:
    """Build 200-token cumulative lexical and first-window-anchored semantic metrics."""
    groups: dict[tuple[str, str, str], list[dict]] = {}
    pair_meta: dict[tuple[str, str], dict] = {}
    for row in windows:
        key = (row["model_key"], row["thread_id"], row["source"])
        groups.setdefault(key, []).append(row)
        pair_meta[(row["model_key"], row["thread_id"])] = {
            "model_key": row["model_key"], "model_label": row["model_label"],
            "provider": row["provider"], "model_id": row["model_id"],
            "thread_id": row["thread_id"], "seed": row["seed"],
        }

    metric_rows: list[dict] = []
    source_summary: dict[tuple[str, str, str], dict] = {}
    for key, rows in groups.items():
        model_key, thread_id, source = key
        rows.sort(key=lambda row: int(row["window_index"]))
        if [int(row["window_index"]) for row in rows] != list(
            range(1, TRAJECTORY_WINDOWS_PER_TRAJECTORY + 1)
        ):
            raise RuntimeError(f"Incomplete trajectory windows: {key}")
        vectors = []
        for row in rows:
            vector = cache.get(EMBEDDING_MODEL, row["token_ids_sha256"])
            if vector is None:
                raise RuntimeError(f"Missing cached trajectory embedding: {row['token_ids_sha256']}")
            vectors.append(vector)
        distances = anchored_cosine_distances(np.vstack(vectors))
        lexical_tokens = [lexical_unigrams(str(row["text"])) for row in rows]
        cumulative_counts = cumulative_unique_unigram_counts([str(row["text"]) for row in rows])
        seen: set[str] = set()
        local = []
        for row, distance, tokens, cumulative_count in zip(rows, distances, lexical_tokens, cumulative_counts):
            window_types = set(tokens)
            new_types = window_types.difference(seen)
            seen.update(window_types)
            record = {
                **pair_meta[(model_key, thread_id)],
                "source": source,
                "window_index": int(row["window_index"]),
                "start_token_1based": int(row["start_token_1based"]),
                "end_token_1based": int(row["end_token_1based"]),
                "cumulative_tokens": int(row["end_token_1based"]),
                "lexical_tokens_in_window": len(tokens),
                "unique_unigrams_in_window": len(window_types),
                "new_unique_unigrams": len(new_types),
                "cumulative_unique_unigrams": int(cumulative_count),
                "anchored_within_run_cosine_distance": float(distance),
            }
            metric_rows.append(record)
            local.append(record)
        source_summary[key] = {
            **pair_meta[(model_key, thread_id)],
            "source": source,
            "mean_anchored_within_run_cosine_distance_windows_2_100": float(np.mean(distances[1:])),
            "final_cumulative_unique_unigrams": int(cumulative_counts[-1]),
        }

    thread_rows = []
    for (model_key, thread_id), meta in pair_meta.items():
        human = source_summary[(model_key, thread_id, "human")]
        llm = source_summary[(model_key, thread_id, "llm")]
        human_distance = human["mean_anchored_within_run_cosine_distance_windows_2_100"]
        llm_distance = llm["mean_anchored_within_run_cosine_distance_windows_2_100"]
        thread_rows.append({
            **meta,
            "human_mean_anchored_within_run_cosine_distance_windows_2_100": human_distance,
            "llm_mean_anchored_within_run_cosine_distance_windows_2_100": llm_distance,
            "human_minus_llm_anchored_within_run_cosine_distance": human_distance - llm_distance,
            "human_final_cumulative_unique_unigrams": human["final_cumulative_unique_unigrams"],
            "llm_final_cumulative_unique_unigrams": llm["final_cumulative_unique_unigrams"],
        })
    return metric_rows, thread_rows


def build_trajectory_model_summary(thread_rows: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = {}
    for row in thread_rows:
        grouped.setdefault(row["model_key"], []).append(row)
    return sorted([
        {
            "model_key": model_key,
            "model_label": rows[0]["model_label"],
            "model_id": rows[0]["model_id"],
            "threads": len(rows),
            "llm_mean_anchored_within_run_cosine_distance_windows_2_100": float(np.mean([
                float(row["llm_mean_anchored_within_run_cosine_distance_windows_2_100"])
                for row in rows
            ])),
        }
        for model_key, rows in grouped.items()
    ], key=lambda row: row["model_key"])


def summarize_matched_trajectory(
    thread_rows: list[dict], *, draws: int, seed: int
) -> tuple[list[dict], list[dict]]:
    """Model-equal question summaries and the paired question-level percentile bootstrap."""
    thread_ids = sorted({row["thread_id"] for row in thread_rows})
    model_keys = {row["model_key"] for row in thread_rows}
    question_rows = []
    for thread_id in thread_ids:
        rows = [row for row in thread_rows if row["thread_id"] == thread_id]
        if {row["model_key"] for row in rows} != model_keys or len(rows) != len(model_keys):
            raise RuntimeError(f"Incomplete model-family set for trajectory summary: {thread_id}")
        human_values = np.asarray([
            float(row["human_mean_anchored_within_run_cosine_distance_windows_2_100"])
            for row in rows
        ])
        if not np.allclose(human_values, human_values[0], rtol=0.0, atol=1e-12):
            raise RuntimeError(f"Repeated Human trajectory differs across models: {thread_id}")
        llm_value = float(np.mean([
            float(row["llm_mean_anchored_within_run_cosine_distance_windows_2_100"])
            for row in rows
        ]))
        human_value = float(human_values[0])
        question_rows.append({
            "thread_id": thread_id,
            "human_mean": human_value,
            "model_equal_llm_mean": llm_value,
            "human_minus_llm": human_value - llm_value,
            "model_families": len(rows),
        })
    differences = np.asarray([float(row["human_minus_llm"]) for row in question_rows])
    rng = np.random.default_rng(seed)
    estimates = np.empty(draws, dtype=float)
    for index in range(draws):
        estimates[index] = float(np.mean(rng.choice(differences, size=len(differences), replace=True)))
    low, high = np.percentile(estimates, [2.5, 97.5])
    summary = [{
        "metric": "first_window_anchored_within_run_cosine_distance",
        "trajectory_window_tokens": TRAJECTORY_WINDOW_TOKENS,
        "trajectory_windows": TRAJECTORY_WINDOWS_PER_TRAJECTORY,
        "summary_windows": "2-100",
        "human_mean": float(np.mean([row["human_mean"] for row in question_rows])),
        "model_equal_llm_mean": float(np.mean([row["model_equal_llm_mean"] for row in question_rows])),
        "human_minus_llm": float(np.mean(differences)),
        "ci_2_5": float(low),
        "ci_97_5": float(high),
        "human_gt_llm_questions": int(np.sum(differences > 0)),
        "resampling_unit": "Reddit question",
        "model_weighting": "equal weight across four model families within question",
        "bootstrap_draws": draws,
        "bootstrap_seed": seed,
        "n_questions": len(question_rows),
    }]
    return summary, question_rows


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
                "mean_within_interval_pairwise_cosine_similarity": float(np.mean([
                    row["mean_pairwise_cosine_similarity"] for row in local
                ])),
                "mean_within_interval_pairwise_cosine_distance": float(np.mean([
                    row["mean_pairwise_cosine_distance"] for row in local
                ])),
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
            "human_mean_within_interval_pairwise_cosine_similarity": human["mean_within_interval_pairwise_cosine_similarity"],
            "llm_mean_within_interval_pairwise_cosine_similarity": llm["mean_within_interval_pairwise_cosine_similarity"],
            "llm_minus_human_within_interval_pairwise_cosine_similarity": (
                llm["mean_within_interval_pairwise_cosine_similarity"]
                - human["mean_within_interval_pairwise_cosine_similarity"]
            ),
            "human_mean_within_interval_pairwise_cosine_distance": human["mean_within_interval_pairwise_cosine_distance"],
            "llm_mean_within_interval_pairwise_cosine_distance": llm["mean_within_interval_pairwise_cosine_distance"],
            "llm_minus_human_within_interval_pairwise_cosine_distance": (
                llm["mean_within_interval_pairwise_cosine_distance"]
                - human["mean_within_interval_pairwise_cosine_distance"]
            ),
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
            "human_mean_within_interval_pairwise_cosine_similarity": human["mean_pairwise_cosine_similarity"],
            "llm_mean_within_interval_pairwise_cosine_similarity": llm["mean_pairwise_cosine_similarity"],
            "llm_minus_human_within_interval_pairwise_cosine_similarity": (
                llm["mean_pairwise_cosine_similarity"] - human["mean_pairwise_cosine_similarity"]
            ),
            "human_mean_within_interval_pairwise_cosine_distance": human["mean_pairwise_cosine_distance"],
            "llm_mean_within_interval_pairwise_cosine_distance": llm["mean_pairwise_cosine_distance"],
            "llm_minus_human_within_interval_pairwise_cosine_distance": (
                llm["mean_pairwise_cosine_distance"] - human["mean_pairwise_cosine_distance"]
            ),
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
            "human_mean_within_interval_pairwise_cosine_similarity": mean("human_mean_within_interval_pairwise_cosine_similarity"),
            "llm_mean_within_interval_pairwise_cosine_similarity": mean("llm_mean_within_interval_pairwise_cosine_similarity"),
            "llm_minus_human_within_interval_pairwise_cosine_similarity": mean("llm_minus_human_within_interval_pairwise_cosine_similarity"),
            "human_mean_within_interval_pairwise_cosine_distance": mean("human_mean_within_interval_pairwise_cosine_distance"),
            "llm_mean_within_interval_pairwise_cosine_distance": mean("llm_mean_within_interval_pairwise_cosine_distance"),
            "llm_minus_human_within_interval_pairwise_cosine_distance": mean("llm_minus_human_within_interval_pairwise_cosine_distance"),
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
        "normalized_vendi": ("llm_minus_human_vendi", 1.0),
        "within_interval_pairwise_cosine_similarity": ("llm_minus_human_pairwise_cosine", 1.0),
        "within_interval_pairwise_cosine_distance": ("llm_minus_human_pairwise_cosine", -1.0),
        "early_minus_late_vendi": ("llm_minus_human_early_minus_late", 1.0),
    }
    thread_ids = sorted({row["thread_id"] for row in thread_rows})
    per_thread: dict[str, dict[str, float]] = {}
    for thread_id in thread_ids:
        rows = [row for row in thread_rows if row["thread_id"] == thread_id]
        per_thread[thread_id] = {
            metric: multiplier * float(np.mean([float(row[field]) for row in rows]))
            for metric, (field, multiplier) in fields.items()
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
    trajectory_windows = build_trajectory_windows(chunks, encoder)
    catalog: Dict[str, list[int]] = {}
    for row in [*chunks, *trajectory_windows]:
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
        trajectory_metric_rows, trajectory_thread_rows = build_trajectory_metrics(
            trajectory_windows, cache=cache
        )
    finally:
        cache.close()
    paired_windows = paired_window_rows(window_rows)
    model_summary = build_model_summary(thread_rows)
    bootstrap = paired_bootstrap(thread_rows, draws=args.bootstrap_draws, seed=args.bootstrap_seed)
    trajectory_model_summary = build_trajectory_model_summary(trajectory_thread_rows)
    trajectory_bootstrap, trajectory_question_rows = summarize_matched_trajectory(
        trajectory_thread_rows, draws=args.bootstrap_draws, seed=args.bootstrap_seed
    )
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
    write_csv(output_dir / "trajectory_window_metrics.csv", trajectory_metric_rows)
    write_csv(output_dir / "trajectory_thread_summary.csv", trajectory_thread_rows)
    write_csv(output_dir / "trajectory_model_summary.csv", trajectory_model_summary)
    write_csv(output_dir / "trajectory_question_summary.csv", trajectory_question_rows)
    write_csv(output_dir / "trajectory_paired_bootstrap_summary.csv", trajectory_bootstrap)

    lines = ["TOPIC-MATCHED REDDIT–LLM SEMANTIC SUMMARY", "=" * 42]
    for row in model_summary:
        lines.append(
            f"{row['model_label']}: Vendi Human={float(row['human_mean_normalized_vendi']):.4f}, "
            f"LLM={float(row['llm_mean_normalized_vendi']):.4f}; within-interval cosine similarity "
            f"Human={float(row['human_mean_within_interval_pairwise_cosine_similarity']):.4f}, "
            f"LLM={float(row['llm_mean_within_interval_pairwise_cosine_similarity']):.4f}; "
            f"distance Human={float(row['human_mean_within_interval_pairwise_cosine_distance']):.4f}, "
            f"LLM={float(row['llm_mean_within_interval_pairwise_cosine_distance']):.4f}; "
            f"Early-Late LLM={float(row['llm_early_minus_late']):+.4f}"
        )
    lines.append("")
    lines.append("Question-level paired bootstrap (LLM - Human):")
    for row in bootstrap:
        lines.append(
            f"  {row['metric']}: {float(row['estimate_llm_minus_human']):+.4f} "
            f"[95% CI {float(row['ci_2_5']):+.4f}, {float(row['ci_97_5']):+.4f}]"
        )
    lines.append("")
    trajectory = trajectory_bootstrap[0]
    lines.append("First-window-anchored within-run cosine distance (200-token windows; mean windows 2-100):")
    lines.append(
        f"  Human={float(trajectory['human_mean']):.6f}; model-equal LLM="
        f"{float(trajectory['model_equal_llm_mean']):.6f}; Human-LLM="
        f"{float(trajectory['human_minus_llm']):+.6f} "
        f"[95% CI {float(trajectory['ci_2_5']):+.6f}, {float(trajectory['ci_97_5']):+.6f}]"
    )
    for row in trajectory_model_summary:
        lines.append(
            f"  {row['model_label']}: "
            f"{float(row['llm_mean_anchored_within_run_cosine_distance_windows_2_100']):.6f}"
        )
    (output_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output_dir / "analysis_metadata.json").write_text(json.dumps({
        "embedding_model": EMBEDDING_MODEL,
        "tokenizer": TOKENIZER_NAME,
        "embedding_input": "fixed-length token-ID arrays",
        "vendi_definition": "cosine Gram kernel -> trace normalization -> exp(entropy) / m",
        "m": CHUNKS_PER_WINDOW,
        "local_semantic_definition": "100-token chunk embeddings grouped into non-overlapping 2,000-token intervals; mean within-interval pairwise cosine similarity and 1-similarity distance",
        "trajectory_window_tokens": TRAJECTORY_WINDOW_TOKENS,
        "trajectory_windows_per_trajectory": TRAJECTORY_WINDOWS_PER_TRAJECTORY,
        "trajectory_semantic_definition": "1 - cosine(window_t, window_1), L2-normalized; trajectory mean excludes window 1",
        "trajectory_lexical_definition": "cumulative unique preprocessed unigrams at each 200-token increment",
        "bootstrap_draws": args.bootstrap_draws,
        "bootstrap_seed": args.bootstrap_seed,
        "cache_hits": hits,
        "new_embeddings": made,
    }, indent=2), encoding="utf-8")
    print((output_dir / "summary.txt").read_text(encoding="utf-8"))
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
