#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Compute ED Fig. 5e time-resolved normalized Vendi trajectories.

The analysis uses individual agent-utterance embeddings from the
``text-embedding-3-large`` lineage. Each 1,000-round transcript is divided into
100 non-overlapping 10-round intervals. Within every run-interval, 30
utterances are sampled without replacement 200 times. For each draw, the
cosine Gram matrix is converted to an eigenspectrum, Vendi effective support is
computed, and the result is normalized by 30.

The five-family, 15-run cohort is configured below. The script writes tables only;
it contains no plotting, smoothing, regression, or figure-generation code.
Existing compatible embedding caches are reused. New embeddings are generated
only when ``--allow-api`` is supplied explicitly.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


EMBEDDING_MODEL = "text-embedding-3-large"
EXPECTED_EMBEDDING_DIMENSION = 3072
EXPECTED_ROUNDS = 1000
WINDOW_SIZE_ROUNDS = 10
EXPECTED_WINDOWS = EXPECTED_ROUNDS // WINDOW_SIZE_ROUNDS

RAREFACTION_M = 30
N_RAREFACTION_DRAWS = 200
BASE_RANDOM_SEED = 20260507
RAREFACTION_RANDOM_SEED = BASE_RANDOM_SEED + RAREFACTION_M
DRAW_CHUNK_SIZE = 200
EPS = 1e-12

OPENAI_BATCH_SIZE = 64
OPENAI_MAX_BATCH_CHARS = 180_000
OPENAI_CHUNK_CHARS = 25_000
OPENAI_MAX_PARTS = 15
OPENAI_RETRY_ATTEMPTS = 6
OPENAI_RETRY_BASE_SECONDS = 2.0


@dataclass(frozen=True)
class RunSpec:
    model_family: str
    run_id: str
    filename: str


# Analysis cohort: five model families x three independent runs.
COHORT: Tuple[RunSpec, ...] = (
    RunSpec("DeepSeek-V3", "3_deepseek_1000_v1", "3_deepseek_1000_v1.txt"),
    RunSpec("DeepSeek-V3", "3_deepseek_1000_v2", "3_deepseek_1000_v2.txt"),
    RunSpec("DeepSeek-V3", "3_deepseek_1000_v3", "3_deepseek_1000_v3.txt"),
    RunSpec("GPT-4-mini", "3_gpt_1000_v1", "3_gpt_1000_v1.txt"),
    RunSpec("GPT-4-mini", "3_gpt_1000_v2", "3_gpt_1000_v2.txt"),
    RunSpec("GPT-4-mini", "3_gpt_1000_v3", "3_gpt_1000_v3.txt"),
    RunSpec("Phi-4", "3_phi-4_1000_v1", "3_phi-4_1000_v1.txt"),
    RunSpec("Phi-4", "3_phi-4_1000_v2", "3_phi-4_1000_v2.txt"),
    RunSpec("Phi-4", "3_phi-4_1000_v3", "3_phi-4_1000_v3.txt"),
    RunSpec("GPT-5.6 Terra", "gpt5.6_1000_v1", "gpt5.6_1000_v1.txt"),
    RunSpec("GPT-5.6 Terra", "gpt5.6_1000_v2", "gpt5.6_1000_v2.txt"),
    RunSpec("GPT-5.6 Terra", "gpt5.6_1000_v3", "gpt5.6_1000_v3.txt"),
    RunSpec("Claude Sonnet 5", "sonnet_1000_v1", "sonnet_1000_v1.txt"),
    RunSpec("Claude Sonnet 5", "sonnet_1000_v2", "sonnet_1000_v2.txt"),
    RunSpec("Claude Sonnet 5", "sonnet_1000_v3", "sonnet_1000_v3.txt"),
)


ROUND_ORDER_RE = re.compile(r"^=+\s*Round\s+(\d+)\s+order", re.IGNORECASE)
ROUND_BRACKET_RE = re.compile(r"^\[Round\s+(\d+)\]", re.IGNORECASE)
SAID_LINE_RE = re.compile(
    r"^\[Round\s+(\d+)\]\s*(?:\(([^)]*)\)\s*)?(.*?)\s+said:\s*(.*)$",
    re.IGNORECASE,
)
NON_AGENT_MARKERS = (
    "referee",
    "system",
    "routing",
    "router",
    "classifier",
    "classification",
    "summary",
    "metadata",
    "moderator",
    "judge",
    "evaluator",
)


@dataclass(frozen=True)
class ParsedMessage:
    round_id: int
    speaker: str
    tag: str
    text: str


def normalize_message_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def text_hash(text: str) -> str:
    normalized = normalize_message_text(text)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def strip_within_style_quote_markers(content: str) -> str:
    content = content.strip()
    if content.startswith("'"):
        content = content[1:]
    if content.endswith("'"):
        content = content[:-1]
    return content.strip()


def is_non_agent_record(tag: str, speaker: str) -> bool:
    probe = f"{tag} {speaker}".lower()
    return any(marker in probe for marker in NON_AGENT_MARKERS)


def parse_agent_messages(text: str) -> Tuple[List[ParsedMessage], List[int]]:
    """Follow the original message parser and retain agent ``said:`` records."""

    messages: List[Dict[str, object]] = []
    rounds_seen: set[int] = set()
    current_round: Optional[int] = None
    current_message: Optional[Dict[str, object]] = None

    def finalize_current_message() -> None:
        nonlocal current_message
        if current_message is None:
            return
        chunks = [
            normalize_message_text(str(chunk))
            for chunk in current_message["chunks"]  # type: ignore[index]
            if normalize_message_text(str(chunk))
        ]
        text_value = normalize_message_text(" ".join(chunks))
        if text_value:
            current_message["text"] = text_value
            messages.append(current_message)
        current_message = None

    for raw_line in text.splitlines():
        line = raw_line.rstrip("\n")
        if not line.strip():
            continue

        order_match = ROUND_ORDER_RE.match(line)
        if order_match:
            finalize_current_message()
            current_round = int(order_match.group(1))
            rounds_seen.add(current_round)
            continue

        said_match = SAID_LINE_RE.match(line)
        if said_match:
            finalize_current_message()
            current_round = int(said_match.group(1))
            rounds_seen.add(current_round)
            tag = (said_match.group(2) or "").strip()
            speaker = normalize_message_text(said_match.group(3) or "")
            content = strip_within_style_quote_markers(said_match.group(4) or "")
            if is_non_agent_record(tag, speaker):
                current_message = None
                continue
            current_message = {
                "round": current_round,
                "speaker": speaker or "unknown",
                "tag": tag,
                "chunks": [],
            }
            if content:
                current_message["chunks"].append(content)  # type: ignore[index]
            continue

        bracket_match = ROUND_BRACKET_RE.match(line)
        if bracket_match:
            finalize_current_message()
            current_round = int(bracket_match.group(1))
            rounds_seen.add(current_round)
            continue

        stripped = line.lstrip()
        if stripped.startswith("->"):
            continue
        if stripped.startswith("=====") or stripped.startswith("[Round "):
            finalize_current_message()
            continue
        if current_round is None or current_message is None:
            continue

        continuation = stripped
        if continuation.endswith("'"):
            continuation = continuation[:-1].rstrip()
        if continuation:
            current_message["chunks"].append(continuation)  # type: ignore[index]

    finalize_current_message()
    parsed = [
        ParsedMessage(
            round_id=int(message["round"]),
            speaker=str(message["speaker"]),
            tag=str(message["tag"]),
            text=str(message["text"]),
        )
        for message in messages
    ]
    return parsed, sorted(rounds_seen)


def parse_transcript(path: Path, spec: RunSpec) -> pd.DataFrame:
    messages, rounds_seen = parse_agent_messages(
        path.read_text(encoding="utf-8", errors="ignore")
    )
    expected_rounds = list(range(1, EXPECTED_ROUNDS + 1))
    if rounds_seen != expected_rounds:
        raise ValueError(
            f"{spec.filename}: expected rounds 1-{EXPECTED_ROUNDS}, "
            f"found {len(rounds_seen)} distinct rounds"
        )
    if not messages:
        raise ValueError(f"{spec.filename}: no eligible agent utterances found")

    rows: List[Dict[str, object]] = []
    for message_index, message in enumerate(messages):
        window_id = ((message.round_id - 1) // WINDOW_SIZE_ROUNDS) + 1
        rows.append(
            {
                "embedding_row": message_index,
                "source_file": spec.filename,
                "model_family": spec.model_family,
                "run_id": spec.run_id,
                "round": message.round_id,
                "window_id": window_id,
                "message_index_within_run": message_index,
                "speaker": message.speaker,
                "tag": message.tag,
                "text_hash": text_hash(message.text),
                "text": message.text,
            }
        )
    frame = pd.DataFrame(rows)
    observed_windows = sorted(frame["window_id"].astype(int).unique().tolist())
    if observed_windows != list(range(1, EXPECTED_WINDOWS + 1)):
        raise ValueError(f"{spec.filename}: incomplete 10-round interval coverage")
    return frame


def l2_normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        return vector.astype(np.float32)
    return (vector / norm).astype(np.float32)


def normalize_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.where(norms <= 0, 1.0, norms)
    return x / norms


def embedding_request_with_retry(client, inputs: Sequence[str]):
    last_error: Optional[Exception] = None
    for attempt in range(OPENAI_RETRY_ATTEMPTS):
        try:
            return client.embeddings.create(model=EMBEDDING_MODEL, input=list(inputs))
        except Exception as exc:  # OpenAI exception classes vary by client version.
            last_error = exc
            if attempt == OPENAI_RETRY_ATTEMPTS - 1:
                break
            time.sleep(OPENAI_RETRY_BASE_SECONDS * (2**attempt))
    raise RuntimeError("OpenAI embedding request failed after retries") from last_error


def split_long_text(text: str) -> List[str]:
    safe_text = text if text else " "
    parts = [
        safe_text[index : index + OPENAI_CHUNK_CHARS]
        for index in range(0, len(safe_text), OPENAI_CHUNK_CHARS)
    ]
    return parts[:OPENAI_MAX_PARTS] or [" "]


def embed_unique_texts(
    texts: Sequence[str],
    api_key_environment_variable: str,
) -> np.ndarray:
    """Embed utterances using the original model and batching/chunking lineage."""

    api_key = os.environ.get(api_key_environment_variable, "").strip()
    if not api_key:
        raise RuntimeError(
            f"Environment variable {api_key_environment_variable!r} is empty; "
            "provide compatible caches or set it and pass --allow-api"
        )
    from openai import OpenAI  # Imported lazily; cache-only runs need no client.

    client = OpenAI(api_key=api_key)
    result: List[Optional[np.ndarray]] = [None] * len(texts)
    normal_indices: List[int] = []

    for index, text in enumerate(texts):
        if len(text) > OPENAI_CHUNK_CHARS:
            response = embedding_request_with_retry(client, split_long_text(text))
            vectors = [np.asarray(item.embedding, dtype=np.float32) for item in response.data]
            result[index] = l2_normalize_vector(np.mean(vectors, axis=0))
        else:
            normal_indices.append(index)

    batch_indices: List[int] = []
    batch_texts: List[str] = []
    batch_chars = 0

    def flush_batch() -> None:
        nonlocal batch_indices, batch_texts, batch_chars
        if not batch_indices:
            return
        response = embedding_request_with_retry(client, batch_texts)
        for index, item in zip(batch_indices, response.data):
            result[index] = l2_normalize_vector(
                np.asarray(item.embedding, dtype=np.float32)
            )
        batch_indices = []
        batch_texts = []
        batch_chars = 0

    for index in normal_indices:
        text = texts[index] or " "
        if batch_indices and (
            len(batch_indices) >= OPENAI_BATCH_SIZE
            or batch_chars + len(text) > OPENAI_MAX_BATCH_CHARS
        ):
            flush_batch()
        batch_indices.append(index)
        batch_texts.append(text)
        batch_chars += len(text)
    flush_batch()

    if any(vector is None for vector in result):
        raise RuntimeError("Embedding response did not cover every utterance")
    matrix = np.vstack(result).astype(np.float32)  # type: ignore[arg-type]
    if matrix.shape[1] != EXPECTED_EMBEDDING_DIMENSION:
        raise ValueError(
            f"Expected {EXPECTED_EMBEDDING_DIMENSION} embedding dimensions, "
            f"received {matrix.shape[1]}"
        )
    return matrix


def cache_paths(cache_dir: Path, spec: RunSpec) -> Tuple[Path, Path]:
    return (
        cache_dir / f"{spec.run_id}_message_metadata.csv",
        cache_dir / f"{spec.run_id}_message_embeddings.npy",
    )


def load_compatible_cache(
    metadata_path: Path,
    embeddings_path: Path,
    parsed: pd.DataFrame,
) -> Optional[Tuple[pd.DataFrame, np.ndarray]]:
    if not metadata_path.exists() or not embeddings_path.exists():
        return None
    metadata = pd.read_csv(metadata_path, dtype={"text_hash": str})
    embeddings = np.load(embeddings_path)
    required = {
        "embedding_row",
        "source_file",
        "model_family",
        "run_id",
        "round",
        "window_id",
        "message_index_within_run",
        "speaker",
        "tag",
        "text_hash",
        "embedding_model",
    }
    if not required.issubset(metadata.columns):
        return None
    if len(metadata) != len(parsed) or embeddings.shape != (
        len(parsed),
        EXPECTED_EMBEDDING_DIMENSION,
    ):
        return None
    if not metadata["embedding_model"].eq(EMBEDDING_MODEL).all():
        return None
    keys = [
        "source_file",
        "run_id",
        "round",
        "window_id",
        "message_index_within_run",
        "text_hash",
    ]
    if not metadata[keys].reset_index(drop=True).equals(
        parsed[keys].reset_index(drop=True)
    ):
        return None
    return metadata, embeddings.astype(np.float32, copy=False)


def save_embedding_cache(
    metadata_path: Path,
    embeddings_path: Path,
    parsed: pd.DataFrame,
    embeddings: np.ndarray,
) -> pd.DataFrame:
    metadata = parsed.drop(columns=["text"]).copy()
    metadata["embedding_model"] = EMBEDDING_MODEL
    temporary_metadata = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
    temporary_embeddings = embeddings_path.with_suffix(embeddings_path.suffix + ".tmp")
    metadata.to_csv(temporary_metadata, index=False, quoting=csv.QUOTE_MINIMAL)
    with temporary_embeddings.open("wb") as handle:
        np.save(handle, embeddings.astype(np.float32))
    temporary_metadata.replace(metadata_path)
    temporary_embeddings.replace(embeddings_path)
    return metadata


def load_or_create_embeddings(
    spec: RunSpec,
    parsed: pd.DataFrame,
    cache_dir: Path,
    allow_api: bool,
    api_key_environment_variable: str,
) -> Tuple[pd.DataFrame, np.ndarray]:
    metadata_path, embeddings_path = cache_paths(cache_dir, spec)
    cached = load_compatible_cache(metadata_path, embeddings_path, parsed)
    if cached is not None:
        return cached
    if not allow_api:
        raise RuntimeError(
            f"No compatible embedding cache for {spec.filename}. "
            "Pass --allow-api only when embedding generation is intended."
        )
    embeddings = embed_unique_texts(
        parsed["text"].astype(str).tolist(),
        api_key_environment_variable,
    )
    metadata = save_embedding_cache(
        metadata_path,
        embeddings_path,
        parsed,
        embeddings,
    )
    return metadata, embeddings


def vendi_from_eigenvalues(
    eigvals: np.ndarray,
    m: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    clipped = np.clip(eigvals, 0.0, None)
    sums = clipped.sum(axis=1, keepdims=True)
    clipped = np.divide(clipped, sums, out=np.zeros_like(clipped), where=sums > 0)
    h_v = -np.sum(clipped * np.log(clipped + EPS), axis=1)
    s_eff = np.exp(h_v)
    norm_v = s_eff / float(m)
    log_s_eff = np.log(s_eff + EPS)
    logit_norm_v = np.log((norm_v + EPS) / (1.0 - norm_v + EPS))
    return h_v, s_eff, norm_v, log_s_eff, logit_norm_v


def compute_rarefied_metrics(
    x_raw: np.ndarray,
    m: int,
    n_repeats: int,
    rng: np.random.Generator,
) -> Tuple[Dict[str, float], Dict[str, object]]:
    """Preserve the confirmed m=30 rarefaction and Vendi calculation."""

    n_messages = int(x_raw.shape[0])
    if n_messages < m:
        raise ValueError(f"Cannot rarefy {n_messages} messages to m={m}")

    x = normalize_rows(x_raw)
    h_all: List[np.ndarray] = []
    s_all: List[np.ndarray] = []
    v_all: List[np.ndarray] = []
    log_s_all: List[np.ndarray] = []
    logit_v_all: List[np.ndarray] = []
    min_raw_eig = math.inf
    max_eig_sum_error = 0.0

    if n_messages == m:
        gram = np.matmul(x, x.T).astype(np.float64, copy=False)
        trace = float(np.trace(gram))
        if trace <= 0:
            raise ValueError("Non-positive Gram trace")
        eigvals = np.linalg.eigvalsh(gram / trace)[None, :]
        min_raw_eig = float(np.min(eigvals))
        h, s, v, log_s, logit_v = vendi_from_eigenvalues(eigvals, m)
        h_all.append(np.repeat(h, n_repeats))
        s_all.append(np.repeat(s, n_repeats))
        v_all.append(np.repeat(v, n_repeats))
        log_s_all.append(np.repeat(log_s, n_repeats))
        logit_v_all.append(np.repeat(logit_v, n_repeats))
        max_eig_sum_error = float(
            abs(np.clip(eigvals, 0.0, None).sum() - 1.0)
        )
    else:
        for start in range(0, n_repeats, DRAW_CHUNK_SIZE):
            chunk = min(DRAW_CHUNK_SIZE, n_repeats - start)
            random_keys = rng.random((chunk, n_messages), dtype=np.float32)
            # The m smallest independent random keys select m distinct rows,
            # exactly preserving the original without-replacement behavior.
            draw_indices = np.argpartition(random_keys, kth=m - 1, axis=1)[:, :m]
            samples = x[draw_indices]
            grams = np.einsum(
                "rmd,rnd->rmn", samples, samples, optimize=True
            ).astype(np.float64, copy=False)
            traces = np.trace(grams, axis1=1, axis2=2)
            if np.any(traces <= 0):
                raise ValueError("Non-positive Gram trace in rarefaction draw")
            grams = grams / traces[:, None, None]
            eigvals = np.linalg.eigvalsh(grams)
            min_raw_eig = min(min_raw_eig, float(np.min(eigvals)))
            clipped = np.clip(eigvals, 0.0, None)
            eig_sums = clipped.sum(axis=1)
            max_eig_sum_error = max(
                max_eig_sum_error,
                float(np.max(np.abs(eig_sums - 1.0))),
            )
            h, s, v, log_s, logit_v = vendi_from_eigenvalues(eigvals, m)
            h_all.append(h)
            s_all.append(s)
            v_all.append(v)
            log_s_all.append(log_s)
            logit_v_all.append(logit_v)

    draws = pd.DataFrame(
        {
            "H_V": np.concatenate(h_all),
            "S_eff": np.concatenate(s_all),
            "norm_vendi": np.concatenate(v_all),
            "log_S_eff": np.concatenate(log_s_all),
            "logit_norm_vendi": np.concatenate(logit_v_all),
        }
    )

    def summarize(values: pd.Series, prefix: str) -> Dict[str, float]:
        array = values.to_numpy(dtype=float)
        mean = float(np.mean(array))
        sd = float(np.std(array, ddof=1)) if len(array) > 1 else 0.0
        half_width = 1.96 * sd / math.sqrt(len(array)) if len(array) > 1 else 0.0
        return {
            f"mean_{prefix}": mean,
            f"sd_{prefix}": sd,
            f"ci95_low_{prefix}": mean - half_width,
            f"ci95_high_{prefix}": mean + half_width,
        }

    summary: Dict[str, float] = {}
    summary.update(summarize(draws["H_V"], "H_V"))
    summary.update(summarize(draws["S_eff"], "S_eff"))
    summary.update(summarize(draws["norm_vendi"], "norm_vendi"))
    summary["mean_log_S_eff"] = float(draws["log_S_eff"].mean())
    summary["mean_logit_norm_vendi"] = float(
        draws["logit_norm_vendi"].mean()
    )

    qc = {
        "min_raw_eigenvalue": min_raw_eig,
        "max_eigenvalue_sum_error_after_clip": max_eig_sum_error,
        "h_range_ok": bool(
            draws["H_V"].between(-1e-8, math.log(m) + 1e-6).all()
        ),
        "s_eff_range_ok": bool(
            draws["S_eff"].between(1.0 - 1e-6, m + 1e-5).all()
        ),
        "norm_vendi_range_ok": bool(
            draws["norm_vendi"].between(1.0 / m - 1e-6, 1.0 + 1e-6).all()
        ),
    }
    return summary, qc


def validate_frozen_cohort() -> None:
    if len(COHORT) != 15:
        raise RuntimeError(f"Expected 15 cohort entries, found {len(COHORT)}")
    counts = pd.Series([spec.model_family for spec in COHORT]).value_counts().to_dict()
    if len(counts) != 5 or set(counts.values()) != {3}:
        raise RuntimeError(f"Expected five families with three runs each, found {counts}")
    filenames = [spec.filename for spec in COHORT]
    run_ids = [spec.run_id for spec in COHORT]
    if len(set(filenames)) != 15 or len(set(run_ids)) != 15:
        raise RuntimeError("Cohort filenames and run IDs must be unique")


def build_manifest(input_dir: Path) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    missing: List[str] = []
    for spec in COHORT:
        path = input_dir / spec.filename
        if not path.is_file():
            missing.append(spec.filename)
        rows.append(
            {
                "model_family": spec.model_family,
                "run_id": spec.run_id,
                "filename": spec.filename,
                "input_present": path.is_file(),
            }
        )
    if missing:
        raise FileNotFoundError(
            "Missing required cohort transcripts: " + ", ".join(missing)
        )
    return pd.DataFrame(rows)


def compute_all_intervals(
    prepared_runs: Sequence[Tuple[RunSpec, pd.DataFrame, np.ndarray]],
) -> pd.DataFrame:
    rng = np.random.default_rng(RAREFACTION_RANDOM_SEED)
    output_rows: List[Dict[str, object]] = []

    # Use one continuous RNG stream over run_id and window_id sorted order.
    # The deterministic N == m branch consumes no draws.
    for spec, metadata, embeddings in sorted(
        prepared_runs,
        key=lambda item: item[0].run_id,
    ):
        for window_id in range(1, EXPECTED_WINDOWS + 1):
            group = metadata.loc[metadata["window_id"].astype(int).eq(window_id)]
            group = group.sort_values("message_index_within_run", kind="stable")
            n_utterances = int(len(group))
            if n_utterances < RAREFACTION_M:
                raise ValueError(
                    f"{spec.filename}, interval {window_id}: {n_utterances} "
                    f"utterances is below m={RAREFACTION_M}"
                )
            embedding_rows = group["embedding_row"].astype(int).to_numpy()
            summary, qc = compute_rarefied_metrics(
                embeddings[embedding_rows],
                m=RAREFACTION_M,
                n_repeats=N_RAREFACTION_DRAWS,
                rng=rng,
            )
            output_rows.append(
                {
                    "source_file": spec.filename,
                    "model_family": spec.model_family,
                    "run_id": spec.run_id,
                    "window_id": window_id,
                    "interval_id": window_id,
                    "round_start": (window_id - 1) * WINDOW_SIZE_ROUNDS + 1,
                    "round_end": window_id * WINDOW_SIZE_ROUNDS,
                    "n_utterances": n_utterances,
                    "rarefaction_m": RAREFACTION_M,
                    "n_draws": N_RAREFACTION_DRAWS,
                    **summary,
                    **qc,
                }
            )
    result = pd.DataFrame(output_rows)
    expected_rows = len(COHORT) * EXPECTED_WINDOWS
    if len(result) != expected_rows:
        raise RuntimeError(f"Expected {expected_rows} output rows, found {len(result)}")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute m=30 rarefied normalized Vendi for the configured 15-run cohort."
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--embedding-cache-dir",
        type=Path,
        default=None,
        help="Default: <output-dir>/embedding_cache",
    )
    parser.add_argument(
        "--allow-api",
        action="store_true",
        help="Generate missing text-embedding-3-large caches via the OpenAI API.",
    )
    parser.add_argument(
        "--api-key-env",
        default="OPENAI_API_KEY",
        help="Environment variable holding the API key when --allow-api is used.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    validate_frozen_cohort()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    cache_dir = (
        args.embedding_cache_dir.resolve()
        if args.embedding_cache_dir is not None
        else output_dir / "embedding_cache"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    manifest = build_manifest(input_dir)
    prepared_runs: List[Tuple[RunSpec, pd.DataFrame, np.ndarray]] = []
    for spec in COHORT:
        parsed = parse_transcript(input_dir / spec.filename, spec)
        metadata, embeddings = load_or_create_embeddings(
            spec,
            parsed,
            cache_dir,
            allow_api=args.allow_api,
            api_key_environment_variable=args.api_key_env,
        )
        prepared_runs.append((spec, metadata, embeddings))

    interval_results = compute_all_intervals(prepared_runs)
    manifest.to_csv(output_dir / "input_manifest.csv", index=False)
    interval_results.to_csv(output_dir / "vendi_by_run_interval.csv", index=False)
    print(f"Wrote {output_dir / 'input_manifest.csv'}")
    print(f"Wrote {output_dir / 'vendi_by_run_interval.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
