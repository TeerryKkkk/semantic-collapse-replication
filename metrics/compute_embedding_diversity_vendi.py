# -*- coding: utf-8 -*-
"""
Message-level within-window embedding diversity pipeline.

This script is intentionally independent from within.py. It parses transcript
files, embeds individual utterances, caches raw message-level embeddings, and
computes within-window diversity metrics such as Vendi Score.

Configuration is kept as constants. There is no argparse interface.
"""

from __future__ import annotations

import csv
import fnmatch
import hashlib
import math
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# =====================
# ====== CONFIG =======
# =====================

INPUT_DIR = "ALL_RESULTS"
OUTPUT_DIR = "EMBEDDING_DIVERSITY_RESULTS"
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

EMBEDDING_MODEL = "text-embedding-3-large"
EXPECTED_DIM = 3072

WINDOW_SIZE = 10
HOP = 10
MIN_MESSAGES_PER_WINDOW = 2

# "canonical_v1_v2_v3" selects only v1, v2, v3 for each family when present.
# "all_available" includes every matching transcript.
RUN_SELECTION_MODE = "all_available"

# Empty list means no filename-pattern restriction. These patterns implement the
# requested subset for the current run: all 1000-round baselines and all
# temperature=0.9 transcripts. Matching is case-insensitive.
SOURCE_FILE_PATTERNS_TO_INCLUDE: List[str] = [
    "3_deepseek_1000_v*.txt",
    "3_gpt_1000_v*.txt",
    "3_phi-4_1000_v*.txt",
    "deepseek0.9_v*.txt",
    "deepseek0.9v*.txt",
    "gpt0.9_v*.txt",
    "gpt0.9v*.txt",
    "phi0.9_v*.txt",
    "phi0.9v*.txt",
]

# Empty list means include all families. Values are matched case-insensitively
# against both inferred family names and filename stems.
FAMILIES_TO_INCLUDE: List[str] = []

# Keep as "smoke_test" for parser/cache validation; use "full_analysis" for the
# selected production run.
EXECUTION_MODE = "full_analysis"  # "smoke_test" | "full_analysis"
SMOKE_TEST_SOURCE_FILE = ""  # empty means use the smallest selected transcript

EMBED_BATCH_SIZE = 64
MAX_RETRIES = 5
RETRY_BASE_SECONDS = 2.0

# OpenAI embedding inputs are bounded by tokens, not characters. This
# conservative character split avoids most oversize-message failures while
# preserving one final averaged embedding per original message.
OPENAI_MAX_CHARS_PER_INPUT = 24_000
LONG_TEXT_CHUNK_CHARS = 20_000
LONG_TEXT_MAX_CHUNKS = 12

# Optional deterministic subsampling for very large windows. None means use all
# messages in each window.
MAX_MESSAGES_PER_WINDOW: Optional[int] = None
RANDOM_SEED = 20260502

# Complete windows only: with WINDOW_SIZE=10 and HOP=10, a 1000-round transcript
# produces 100 windows and a 200-round transcript produces 20 windows.
COMPLETE_WINDOWS_ONLY = True

# Run-level early/late summary uses all valid windows by splitting them into
# first half versus second half, not just a three-window edge sample.
SUMMARY_SPLIT_MODE = "halves"
MAKE_PLOTS = True
COMPUTE_COVARIANCE_EFFECTIVE_RANK = True


# =====================
# ======= REGEX =======
# =====================

ROUND_HEADER_RE = re.compile(r"^=+\s*Round\s+(\d+)\s+order:", re.IGNORECASE)
MESSAGE_RE = re.compile(
    r"^\[Round\s+(\d+)\]\s*\(([^)]+)\)\s+([^\s]+)\s+said:\s*(.*)$",
    re.IGNORECASE,
)
METADATA_RE = re.compile(r"^\s*->")
RUN_1000_RE = re.compile(r"^3_(?P<family>.+)_1000_v(?P<run>\d+)\.txt$", re.IGNORECASE)
RUN_VERSION_RE = re.compile(r"^(?P<family>.+?)[_-]?V(?P<run>\d+)\.txt$", re.IGNORECASE)


@dataclass(frozen=True)
class FileInfo:
    path: Path
    source_file: str
    family: str
    run_num: Optional[int]
    run_id: str
    inferred_round_count_from_filename: Optional[int]


def safe_stem(name: str) -> str:
    stem = Path(name).stem
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", stem)


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def infer_file_info(path: Path) -> FileInfo:
    name = path.name
    m1000 = RUN_1000_RE.match(name)
    if m1000:
        run_num = int(m1000.group("run"))
        family = m1000.group("family").lower()
        return FileInfo(
            path=path,
            source_file=name,
            family=family,
            run_num=run_num,
            run_id=f"v{run_num}",
            inferred_round_count_from_filename=1000,
        )

    mv = RUN_VERSION_RE.match(name)
    if mv:
        run_num = int(mv.group("run"))
        family = mv.group("family").lower()
        return FileInfo(
            path=path,
            source_file=name,
            family=family,
            run_num=run_num,
            run_id=f"v{run_num}",
            inferred_round_count_from_filename=None,
        )

    return FileInfo(
        path=path,
        source_file=name,
        family=path.stem.lower(),
        run_num=None,
        run_id="unknown",
        inferred_round_count_from_filename=None,
    )


def strip_outer_transcript_quotes(text: str) -> str:
    text = text.strip()
    if text.startswith("'"):
        text = text[1:]
    if text.endswith("'"):
        text = text[:-1]
    return text.strip()


def parse_transcript(path: Path, info: FileInfo) -> pd.DataFrame:
    records: List[Dict[str, object]] = []
    current: Optional[Dict[str, object]] = None
    current_parts: List[str] = []
    per_round_counts: Dict[int, int] = {}

    def flush_current() -> None:
        nonlocal current, current_parts
        if current is None:
            return
        text = strip_outer_transcript_quotes("\n".join(current_parts))
        if text:
            current["text"] = text
            current["text_hash"] = text_hash(text)
            records.append(current)
        current = None
        current_parts = []

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")

            if ROUND_HEADER_RE.match(line):
                flush_current()
                continue

            m = MESSAGE_RE.match(line)
            if m:
                flush_current()
                round_id = int(m.group(1))
                message_type = m.group(2).upper()
                agent = m.group(3)
                initial_text = m.group(4).strip()
                per_round_counts[round_id] = per_round_counts.get(round_id, 0) + 1
                current = {
                    "source_file": info.source_file,
                    "family": info.family,
                    "run_id": info.run_id,
                    "round": round_id,
                    "message_index_global": len(records),
                    "message_index_within_round": per_round_counts[round_id] - 1,
                    "message_type": message_type,
                    "agent": agent,
                }
                current_parts = [initial_text] if initial_text else []
                continue

            if METADATA_RE.match(line):
                continue

            if current is not None:
                current_parts.append(line)

    flush_current()

    columns = [
        "source_file",
        "family",
        "run_id",
        "round",
        "message_index_global",
        "message_index_within_round",
        "message_type",
        "agent",
        "text",
        "text_hash",
    ]
    return pd.DataFrame(records, columns=columns)


def discover_transcripts(input_dir: Path) -> List[FileInfo]:
    return [
        infer_file_info(p)
        for p in sorted(input_dir.glob("*.txt"), key=lambda x: x.name.lower())
    ]


def family_filter_matches(info: FileInfo) -> bool:
    if SOURCE_FILE_PATTERNS_TO_INCLUDE:
        name = info.source_file.lower()
        if not any(fnmatch.fnmatch(name, pattern.lower()) for pattern in SOURCE_FILE_PATTERNS_TO_INCLUDE):
            return False

    if not FAMILIES_TO_INCLUDE:
        return True
    haystacks = [info.family.lower(), info.path.stem.lower(), info.source_file.lower()]
    prefixes = [p.lower() for p in FAMILIES_TO_INCLUDE]
    return any(any(h.startswith(prefix) for h in haystacks) for prefix in prefixes)


def select_for_analysis(infos: Sequence[FileInfo]) -> Dict[str, Tuple[bool, str]]:
    selection: Dict[str, Tuple[bool, str]] = {}
    for info in infos:
        if not family_filter_matches(info):
            selection[info.source_file] = (False, "family_not_included")
            continue

        if RUN_SELECTION_MODE == "all_available":
            selection[info.source_file] = (True, "all_available")
            continue

        if RUN_SELECTION_MODE == "canonical_v1_v2_v3":
            if info.run_num in {1, 2, 3}:
                selection[info.source_file] = (True, "canonical_v1_v2_v3")
            else:
                selection[info.source_file] = (
                    False,
                    f"canonical_mode_excludes_{info.run_id}",
                )
            continue

        raise ValueError(f"Unknown RUN_SELECTION_MODE: {RUN_SELECTION_MODE}")

    if EXECUTION_MODE == "smoke_test":
        selected = [info for info in infos if selection.get(info.source_file, (False, ""))[0]]
        if SMOKE_TEST_SOURCE_FILE:
            smoke = [info for info in selected if info.source_file == SMOKE_TEST_SOURCE_FILE]
            if not smoke:
                raise ValueError(f"SMOKE_TEST_SOURCE_FILE not selected: {SMOKE_TEST_SOURCE_FILE}")
            smoke_name = smoke[0].source_file
        else:
            smoke_name = min(selected, key=lambda x: x.path.stat().st_size).source_file

        for info in infos:
            included, reason = selection[info.source_file]
            if info.source_file == smoke_name:
                selection[info.source_file] = (True, f"smoke_test_selected; planned={reason}")
            elif included:
                selection[info.source_file] = (False, f"smoke_test_not_selected; planned={reason}")

    return selection


def make_openai_client(api_key: str):
    if not api_key:
        return None

    from openai import OpenAI

    return OpenAI(api_key=api_key)


def embedding_create_with_retry(client, inputs: Sequence[str]):
    last_error: Optional[BaseException] = None
    for attempt in range(MAX_RETRIES):
        try:
            return client.embeddings.create(model=EMBEDDING_MODEL, input=list(inputs))
        except Exception as exc:  # OpenAI SDK exposes several transient classes.
            last_error = exc
            if attempt == MAX_RETRIES - 1:
                break
            delay = RETRY_BASE_SECONDS * (2**attempt)
            print(f"[WARN] embedding batch failed on attempt {attempt + 1}; retrying in {delay:.1f}s")
            time.sleep(delay)
    raise RuntimeError("Embedding request failed after retries") from last_error


def l2_normalize_matrix(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / (norms + 1e-12)


def average_chunk_embeddings(vecs: np.ndarray) -> np.ndarray:
    vec = np.asarray(vecs, dtype=np.float32).mean(axis=0)
    norm = float(np.linalg.norm(vec))
    if norm > 0:
        vec = vec / norm
    return vec.astype(np.float32)


def split_long_text(text: str) -> List[str]:
    chunks = []
    for i in range(0, len(text), LONG_TEXT_CHUNK_CHARS):
        chunk = text[i : i + LONG_TEXT_CHUNK_CHARS]
        if chunk.strip():
            chunks.append(chunk)
        if len(chunks) >= LONG_TEXT_MAX_CHUNKS:
            break
    return chunks or [" "]


def embed_texts(client, texts: Sequence[str]) -> np.ndarray:
    vectors: List[Optional[np.ndarray]] = [None] * len(texts)
    batch_inputs: List[str] = []
    batch_indices: List[int] = []

    def flush_batch() -> None:
        nonlocal batch_inputs, batch_indices
        if not batch_inputs:
            return
        resp = embedding_create_with_retry(client, batch_inputs)
        for idx, datum in zip(batch_indices, resp.data):
            vectors[idx] = np.asarray(datum.embedding, dtype=np.float32)
        batch_inputs = []
        batch_indices = []

    for idx, text in enumerate(texts):
        text = text if text and text.strip() else " "
        if len(text) > OPENAI_MAX_CHARS_PER_INPUT:
            flush_batch()
            chunks = split_long_text(text)
            resp = embedding_create_with_retry(client, chunks)
            chunk_vecs = np.vstack([np.asarray(d.embedding, dtype=np.float32) for d in resp.data])
            vectors[idx] = average_chunk_embeddings(chunk_vecs)
        else:
            batch_inputs.append(text)
            batch_indices.append(idx)
            if len(batch_inputs) >= EMBED_BATCH_SIZE:
                flush_batch()

    flush_batch()

    if any(v is None for v in vectors):
        raise RuntimeError("Internal error: missing embeddings after API calls")

    mat = np.vstack([v for v in vectors if v is not None]).astype(np.float32)
    if mat.shape != (len(texts), EXPECTED_DIM):
        raise ValueError(f"Unexpected embedding shape: {mat.shape}, expected {(len(texts), EXPECTED_DIM)}")
    return l2_normalize_matrix(mat)


def cache_paths(cache_dir: Path, source_file: str) -> Tuple[Path, Path]:
    stem = safe_stem(source_file)
    return cache_dir / f"{stem}_metadata.csv", cache_dir / f"{stem}_embeddings.npy"


def cache_is_valid(metadata_path: Path, embeddings_path: Path, parsed_df: pd.DataFrame) -> bool:
    if not metadata_path.exists() or not embeddings_path.exists():
        return False
    try:
        cached_meta = pd.read_csv(metadata_path)
        embeddings = np.load(embeddings_path, mmap_mode="r")
    except Exception:
        return False

    if len(cached_meta) != len(parsed_df):
        return False
    if tuple(embeddings.shape) != (len(parsed_df), EXPECTED_DIM):
        return False
    if "text_hash" in cached_meta.columns and len(parsed_df) > 0:
        if not cached_meta["text_hash"].astype(str).equals(parsed_df["text_hash"].astype(str).reset_index(drop=True)):
            return False
    return True


def save_cache(metadata_path: Path, embeddings_path: Path, metadata: pd.DataFrame, embeddings: np.ndarray) -> None:
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    temp_meta = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
    temp_embeddings = embeddings_path.with_suffix(embeddings_path.suffix + ".tmp")
    metadata.to_csv(temp_meta, index=False, quoting=csv.QUOTE_MINIMAL)
    with temp_embeddings.open("wb") as f:
        np.save(f, embeddings.astype(np.float32))
    temp_meta.replace(metadata_path)
    temp_embeddings.replace(embeddings_path)


def load_or_embed_transcript(
    client,
    cache_dir: Path,
    info: FileInfo,
    parsed_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, np.ndarray, str]:
    metadata_path, embeddings_path = cache_paths(cache_dir, info.source_file)
    if cache_is_valid(metadata_path, embeddings_path, parsed_df):
        return pd.read_csv(metadata_path), np.load(embeddings_path).astype(np.float32), "cache"

    if client is None:
        raise RuntimeError(
            "OPENAI_API_KEY must be set to compute embeddings when the metadata/embedding cache is missing or invalid."
        )

    metadata = parsed_df.copy().reset_index(drop=True)
    metadata["embedding_model"] = EMBEDDING_MODEL
    embeddings = embed_texts(client, metadata["text"].astype(str).tolist())
    save_cache(metadata_path, embeddings_path, metadata, embeddings)
    return metadata, embeddings, "embedded"


def deterministic_subsample_indices(source_file: str, window_index: int, n: int, max_n: int) -> np.ndarray:
    digest = hashlib.sha256(f"{source_file}:{window_index}:{RANDOM_SEED}".encode("utf-8")).hexdigest()
    seed = int(digest[:16], 16) % (2**32)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=max_n, replace=False))


def effective_rank(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = np.clip(values, 0.0, None)
    total = float(values.sum())
    if total <= 0:
        return float("nan")
    probs = values / total
    probs = probs[probs > 0]
    return float(np.exp(-np.sum(probs * np.log(probs))))


def compute_window_metrics(x: np.ndarray) -> Dict[str, float]:
    x = l2_normalize_matrix(x)
    n = x.shape[0]
    k = np.asarray(x @ x.T, dtype=np.float64)
    k = (k + k.T) / 2.0

    k_scaled = k / float(n)
    eigvals = np.linalg.eigvalsh(k_scaled)
    eigvals_clipped = np.clip(eigvals, 0.0, None)
    eig_sum_before = float(eigvals_clipped.sum())
    if eig_sum_before > 0:
        eig_probs = eigvals_clipped / eig_sum_before
    else:
        eig_probs = eigvals_clipped
    nonzero = eig_probs[eig_probs > 0]
    vendi_score = float(np.exp(-np.sum(nonzero * np.log(nonzero)))) if len(nonzero) else float("nan")

    if n > 1:
        upper = k[np.triu_indices(n, k=1)]
        mean_pairwise_cosine = float(np.mean(upper))
    else:
        mean_pairwise_cosine = float("nan")
    mean_pairwise_distance = 1.0 - mean_pairwise_cosine if not math.isnan(mean_pairwise_cosine) else float("nan")

    centroid = x.mean(axis=0)
    centroid_norm = float(np.linalg.norm(centroid))
    if centroid_norm > 0:
        centroid_unit = centroid / centroid_norm
        centroid_mean_cosine_distance = float(np.mean(1.0 - (x @ centroid_unit)))
    else:
        centroid_mean_cosine_distance = float("nan")

    cov_effective_rank = float("nan")
    if COMPUTE_COVARIANCE_EFFECTIVE_RANK and n > 1:
        centered = x - x.mean(axis=0, keepdims=True)
        singular_values = np.linalg.svd(centered, compute_uv=False)
        cov_eigs = (singular_values**2) / float(max(1, n - 1))
        cov_effective_rank = effective_rank(cov_eigs)

    return {
        "vendi_score": vendi_score,
        "vendi_score_norm": vendi_score / float(n),
        "mean_pairwise_cosine": mean_pairwise_cosine,
        "mean_pairwise_cosine_distance": mean_pairwise_distance,
        "centroid_mean_cosine_distance": centroid_mean_cosine_distance,
        "covariance_effective_rank": cov_effective_rank,
        "gram_trace": float(np.trace(k)),
        "kernel_eigenvalue_sum_before_renorm": eig_sum_before,
    }


def build_window_rows(info: FileInfo, metadata: pd.DataFrame, embeddings: np.ndarray) -> pd.DataFrame:
    if metadata.empty:
        return pd.DataFrame()

    actual_max_round = int(metadata["round"].max())
    round_count = actual_max_round
    rows: List[Dict[str, object]] = []
    window_index = 0
    last_start = actual_max_round - WINDOW_SIZE + 1 if COMPLETE_WINDOWS_ONLY else actual_max_round
    for round_start in range(1, max(0, last_start) + 1, HOP):
        window_index += 1
        round_end = round_start + WINDOW_SIZE - 1
        mask = (metadata["round"] >= round_start) & (metadata["round"] <= round_end)
        positions = np.flatnonzero(mask.to_numpy())
        n_original = int(len(positions))

        base = {
            "source_file": info.source_file,
            "family": info.family,
            "run_id": info.run_id,
            "round_count": round_count,
            "window_size": WINDOW_SIZE,
            "hop": HOP,
            "window_index": window_index,
            "round_start": round_start,
            "round_end": round_end,
            "n_messages_original": n_original,
        }

        if n_original < MIN_MESSAGES_PER_WINDOW:
            rows.append(
                {
                    **base,
                    "n_messages": n_original,
                    "vendi_score": np.nan,
                    "vendi_score_norm": np.nan,
                    "mean_pairwise_cosine": np.nan,
                    "mean_pairwise_cosine_distance": np.nan,
                    "centroid_mean_cosine_distance": np.nan,
                    "covariance_effective_rank": np.nan,
                    "gram_trace": np.nan,
                    "kernel_eigenvalue_sum_before_renorm": np.nan,
                    "status": "too_few_messages",
                }
            )
            continue

        status = "ok"
        if MAX_MESSAGES_PER_WINDOW is not None and n_original > MAX_MESSAGES_PER_WINDOW:
            sub = deterministic_subsample_indices(info.source_file, window_index, n_original, MAX_MESSAGES_PER_WINDOW)
            positions = positions[sub]
            status = f"ok_subsampled_from_{n_original}"

        x = embeddings[positions]
        metrics = compute_window_metrics(x)
        rows.append(
            {
                **base,
                "n_messages": int(len(positions)),
                **metrics,
                "status": status,
            }
        )

    return pd.DataFrame(rows)


def summarize_run(info: FileInfo, metadata: pd.DataFrame, window_df: pd.DataFrame) -> Dict[str, object]:
    valid = window_df[window_df["status"].astype(str).str.startswith("ok")].copy()
    if valid.empty:
        early = valid
        late = valid
        early_range = ""
        late_range = ""
    else:
        if SUMMARY_SPLIT_MODE == "halves":
            midpoint = len(valid) // 2
            early = valid.iloc[:midpoint].copy()
            late = valid.iloc[midpoint:].copy()
        else:
            raise ValueError(f"Unknown SUMMARY_SPLIT_MODE: {SUMMARY_SPLIT_MODE}")
        early_range = f"{int(early['window_index'].min())}-{int(early['window_index'].max())}"
        late_range = f"{int(late['window_index'].min())}-{int(late['window_index'].max())}"

    def mean_col(df: pd.DataFrame, col: str) -> float:
        return float(df[col].mean()) if not df.empty else float("nan")

    early_vendi = mean_col(early, "vendi_score")
    late_vendi = mean_col(late, "vendi_score")
    early_vendi_norm = mean_col(early, "vendi_score_norm")
    late_vendi_norm = mean_col(late, "vendi_score_norm")
    early_dist = mean_col(early, "mean_pairwise_cosine_distance")
    late_dist = mean_col(late, "mean_pairwise_cosine_distance")

    return {
        "source_file": info.source_file,
        "family": info.family,
        "run_id": info.run_id,
        "round_count": int(metadata["round"].max()) if not metadata.empty else 0,
        "n_messages_total": int(len(metadata)),
        "n_windows": int(len(window_df)),
        "n_valid_windows": int(len(valid)),
        "early_window_range": early_range,
        "late_window_range": late_range,
        "early_mean_vendi": early_vendi,
        "late_mean_vendi": late_vendi,
        "late_minus_early_vendi": late_vendi - early_vendi,
        "early_mean_vendi_norm": early_vendi_norm,
        "late_mean_vendi_norm": late_vendi_norm,
        "late_minus_early_vendi_norm": late_vendi_norm - early_vendi_norm,
        "early_mean_pairwise_distance": early_dist,
        "late_mean_pairwise_distance": late_dist,
        "late_minus_early_pairwise_distance": late_dist - early_dist,
    }


def build_manifest_rows(infos: Sequence[FileInfo], parsed_by_file: Dict[str, pd.DataFrame], selection: Dict[str, Tuple[bool, str]]) -> pd.DataFrame:
    rows = []
    for info in infos:
        parsed = parsed_by_file.get(info.source_file)
        included, reason = selection[info.source_file]
        rows.append(
            {
                "source_file": info.source_file,
                "inferred_family": info.family,
                "inferred_run_id": info.run_id,
                "inferred_round_count_from_filename": info.inferred_round_count_from_filename,
                "actual_max_round_parsed": int(parsed["round"].max()) if parsed is not None and not parsed.empty else 0,
                "n_messages": int(len(parsed)) if parsed is not None else 0,
                "included_in_analysis": bool(included),
                "exclusion_reason": "" if included else reason,
            }
        )
    return pd.DataFrame(rows)


def write_plots(window_df: pd.DataFrame, summary_df: pd.DataFrame, plots_dir: Path) -> None:
    if not MAKE_PLOTS or window_df.empty:
        return
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[WARN] matplotlib unavailable; skipping plots: {exc}")
        return

    plots_dir.mkdir(parents=True, exist_ok=True)
    valid = window_df[window_df["status"].astype(str).str.startswith("ok")].copy()
    for family, group in valid.groupby("family"):
        for metric, suffix, ylabel in [
            ("vendi_score", "vendi_score", "Vendi Score"),
            ("vendi_score_norm", "vendi_score_norm", "Normalized Vendi Score"),
        ]:
            plt.figure(figsize=(9, 5))
            for source_file, run_group in group.groupby("source_file"):
                plt.plot(
                    run_group["window_index"],
                    run_group[metric],
                    marker="o",
                    linewidth=1.2,
                    markersize=3,
                    label=source_file,
                )
            plt.xlabel("Window index")
            plt.ylabel(ylabel)
            plt.title(f"{family}: {ylabel} by window")
            plt.grid(True, linestyle="--", alpha=0.3)
            plt.legend(fontsize=7)
            plt.tight_layout()
            plt.savefig(plots_dir / f"{safe_stem(family)}_{suffix}.png", dpi=160)
            plt.close()

    if not summary_df.empty:
        plt.figure(figsize=(8, 5))
        x = np.arange(len(summary_df))
        plt.scatter(x, summary_df["early_mean_vendi"], label="early", alpha=0.8)
        plt.scatter(x, summary_df["late_mean_vendi"], label="late", alpha=0.8)
        plt.xticks(x, summary_df["source_file"], rotation=90, fontsize=7)
        plt.ylabel("Mean Vendi Score")
        plt.title("Early vs late Vendi Score by run")
        plt.grid(True, linestyle="--", alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(plots_dir / "early_vs_late_vendi_by_run.png", dpi=160)
        plt.close()


def main() -> None:
    root = Path.cwd()
    input_dir = root / INPUT_DIR
    output_dir = root / OUTPUT_DIR
    cache_dir = output_dir / "cache"
    plots_dir = output_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    infos = discover_transcripts(input_dir)
    if not infos:
        raise FileNotFoundError(f"No transcript .txt files found in {input_dir}")

    parsed_by_file: Dict[str, pd.DataFrame] = {}
    for info in infos:
        parsed_by_file[info.source_file] = parse_transcript(info.path, info)

    selection = select_for_analysis(infos)
    manifest_df = build_manifest_rows(infos, parsed_by_file, selection)
    manifest_path = output_dir / "input_manifest.csv"
    manifest_df.to_csv(manifest_path, index=False)

    selected_infos = [info for info in infos if selection[info.source_file][0]]
    if not selected_infos:
        raise ValueError("No transcripts selected for analysis")

    print(f"[INFO] execution_mode={EXECUTION_MODE}")
    print(f"[INFO] run_selection_mode={RUN_SELECTION_MODE}")
    print(f"[INFO] transcripts_discovered={len(infos)} selected={len(selected_infos)}")
    print("[INFO] selected files:")
    for info in selected_infos:
        parsed = parsed_by_file[info.source_file]
        print(f"  - {info.source_file}: family={info.family}, run={info.run_id}, messages={len(parsed)}, max_round={int(parsed['round'].max()) if not parsed.empty else 0}")

    client = make_openai_client(OPENAI_API_KEY)

    all_window_dfs: List[pd.DataFrame] = []
    summary_rows: List[Dict[str, object]] = []
    embedded_count = 0
    loaded_count = 0

    for info in selected_infos:
        parsed = parsed_by_file[info.source_file]
        print(f"[INFO] processing {info.source_file} ({len(parsed)} messages)")
        metadata, embeddings, source = load_or_embed_transcript(client, cache_dir, info, parsed)
        if source == "cache":
            loaded_count += len(metadata)
        else:
            embedded_count += len(metadata)
        print(f"[INFO] embeddings_{source}: rows={len(metadata)}, shape={embeddings.shape}")

        window_df = build_window_rows(info, metadata, embeddings)
        all_window_dfs.append(window_df)
        summary_rows.append(summarize_run(info, metadata, window_df))

        window_path = output_dir / "window_diversity_metrics.csv"
        summary_path = output_dir / "run_summary.csv"
        pd.concat(all_window_dfs, ignore_index=True).to_csv(window_path, index=False)
        pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
        print(f"[INFO] saved partial results after {info.source_file}")

    window_all = pd.concat(all_window_dfs, ignore_index=True) if all_window_dfs else pd.DataFrame()
    summary_df = pd.DataFrame(summary_rows)
    window_all.to_csv(output_dir / "window_diversity_metrics.csv", index=False)
    summary_df.to_csv(output_dir / "run_summary.csv", index=False)
    write_plots(window_all, summary_df, plots_dir)

    print(f"[OK] manifest={manifest_path}")
    print(f"[OK] window_metrics={output_dir / 'window_diversity_metrics.csv'}")
    print(f"[OK] run_summary={output_dir / 'run_summary.csv'}")
    print(f"[OK] embedded_messages={embedded_count} loaded_from_cache={loaded_count}")

    if EXECUTION_MODE == "smoke_test" and not window_all.empty:
        example = window_all[window_all["status"].astype(str).str.startswith("ok")].head(3)
        print("[SMOKE] example valid windows:")
        print(
            example[
                [
                    "source_file",
                    "window_index",
                    "round_start",
                    "round_end",
                    "n_messages",
                    "vendi_score",
                    "vendi_score_norm",
                    "mean_pairwise_cosine_distance",
                    "status",
                ]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()
