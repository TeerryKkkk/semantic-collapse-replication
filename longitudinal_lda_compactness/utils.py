"""Embedding I/O and memory-bounded cosine-distance calculations."""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import math
import os
import sqlite3
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    EMBEDDING_API_URL,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_DIMENSION,
    EMBEDDING_MAX_ATTEMPTS,
    EMBEDDING_MODEL,
    HISTOGRAM_BINS,
    PAIRWISE_BLOCK_SIZE,
)


LOGGER = logging.getLogger(__name__)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _initialize_embedding_cache(conn: sqlite3.Connection, inventory: pd.DataFrame) -> None:
    conn.executescript(
        """
        PRAGMA journal_mode=WAL;
        PRAGMA synchronous=FULL;
        CREATE TABLE IF NOT EXISTS metadata(
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS embeddings(
            sha256 TEXT PRIMARY KEY,
            exact_text TEXT NOT NULL,
            vector BLOB,
            vector_dtype TEXT,
            vector_sha256 TEXT,
            vector_l2_norm REAL,
            local_request_id TEXT,
            api_request_id TEXT,
            response_model TEXT,
            response_index INTEGER,
            embedded_utc TEXT
        );
        CREATE TABLE IF NOT EXISTS api_requests(
            local_request_id TEXT PRIMARY KEY,
            api_request_id TEXT,
            started_utc TEXT NOT NULL,
            completed_utc TEXT,
            input_count INTEGER NOT NULL,
            ordered_input_hashes_json TEXT NOT NULL,
            request_body_sha256 TEXT NOT NULL,
            http_status INTEGER,
            prompt_tokens INTEGER,
            total_tokens INTEGER,
            attempt_count INTEGER NOT NULL,
            status TEXT NOT NULL,
            error_type TEXT,
            error_message TEXT
        );
        """
    )
    metadata = {
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dimension": str(EMBEDDING_DIMENSION),
        "character_truncation": "none",
        "api_input_transformation": "none; exact parser Unicode text serialized in JSON",
    }
    for key, value in metadata.items():
        existing = conn.execute("SELECT value FROM metadata WHERE key=?", (key,)).fetchone()
        if existing and existing[0] != value:
            raise RuntimeError(f"Embedding cache metadata mismatch: {key}")
        conn.execute("INSERT OR IGNORE INTO metadata(key,value) VALUES (?,?)", (key, value))
    conn.executemany(
        "INSERT OR IGNORE INTO embeddings(sha256,exact_text) VALUES (?,?)",
        inventory[["text_sha256", "text"]].itertuples(index=False, name=None),
    )
    conn.commit()
    for digest, text in conn.execute(
        "SELECT sha256,exact_text FROM embeddings WHERE sha256 IN (SELECT sha256 FROM embeddings)"
    ):
        if sha256_text(text) != digest:
            raise RuntimeError(f"Exact-text hash mismatch in embedding cache: {digest}")


def _encode_embedding_request(texts: list[str]) -> bytes:
    return json.dumps(
        {"model": EMBEDDING_MODEL, "input": texts, "encoding_format": "base64"},
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")


def _call_embedding_api(
    api_key: str, body: bytes, local_id: str
) -> tuple[dict[str, object], str | None, int, int]:
    last_error: Exception | None = None
    for attempt in range(1, EMBEDDING_MAX_ATTEMPTS + 1):
        request = urllib.request.Request(
            EMBEDDING_API_URL,
            data=body,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "User-Agent": "topic-compactness-analysis",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                payload = json.loads(response.read().decode("utf-8"))
                return payload, response.headers.get("x-request-id"), int(response.status), attempt
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:2000]
            last_error = RuntimeError(f"HTTP {exc.code}: {detail}")
            if exc.code not in {408, 409, 429, 500, 502, 503, 504}:
                raise last_error
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            last_error = exc
        if attempt < EMBEDDING_MAX_ATTEMPTS:
            time.sleep(min(30.0, 2.0 ** (attempt - 1)))
    raise RuntimeError(
        f"Embedding request {local_id} failed after {EMBEDDING_MAX_ATTEMPTS} attempts: "
        f"{type(last_error).__name__}"
    )


def _decode_vector(value: object) -> bytes:
    if isinstance(value, str):
        raw = base64.b64decode(value, validate=True)
    else:
        raw = np.asarray(value, dtype="<f4").tobytes(order="C")
    if len(raw) != EMBEDDING_DIMENSION * 4:
        raise RuntimeError(f"Unexpected embedding byte length: {len(raw)}")
    if not np.isfinite(np.frombuffer(raw, dtype="<f4")).all():
        raise RuntimeError("Embedding response contains a nonfinite vector")
    return raw


def _embed_missing(conn: sqlite3.Connection, api_key: str) -> None:
    remaining = int(conn.execute("SELECT COUNT(*) FROM embeddings WHERE vector IS NULL").fetchone()[0])
    LOGGER.info("Embedding cache contains %s missing unique texts", f"{remaining:,}")
    while remaining:
        rows = conn.execute(
            "SELECT sha256,exact_text FROM embeddings WHERE vector IS NULL ORDER BY sha256 LIMIT ?",
            (EMBEDDING_BATCH_SIZE,),
        ).fetchall()
        hashes = [str(row[0]) for row in rows]
        body = _encode_embedding_request([str(row[1]) for row in rows])
        local_id = str(uuid.uuid4())
        conn.execute(
            """
            INSERT INTO api_requests(
                local_request_id,started_utc,input_count,ordered_input_hashes_json,
                request_body_sha256,attempt_count,status
            ) VALUES (?,?,?,?,?,?,?)
            """,
            (
                local_id,
                utc_now(),
                len(rows),
                json.dumps(hashes, separators=(",", ":")),
                hashlib.sha256(body).hexdigest(),
                0,
                "started",
            ),
        )
        conn.commit()
        try:
            payload, api_request_id, status, attempts = _call_embedding_api(api_key, body, local_id)
            data = payload.get("data")
            response_model = str(payload.get("model", ""))
            if response_model != EMBEDDING_MODEL:
                raise RuntimeError(f"Unexpected embedding response model: {response_model}")
            if not isinstance(data, list) or len(data) != len(rows):
                raise RuntimeError("Embedding response length mismatch")
            ordered = sorted(data, key=lambda item: int(item["index"]))
            if [int(item["index"]) for item in ordered] != list(range(len(rows))):
                raise RuntimeError("Embedding response indices are incomplete or out of order")
            completed = utc_now()
            updates = []
            for item, digest in zip(ordered, hashes):
                raw = _decode_vector(item["embedding"])
                vector = np.frombuffer(raw, dtype="<f4")
                updates.append(
                    (
                        raw,
                        "float32-little-endian",
                        hashlib.sha256(raw).hexdigest(),
                        float(np.linalg.norm(vector.astype(np.float64))),
                        local_id,
                        api_request_id,
                        response_model,
                        int(item["index"]),
                        completed,
                        digest,
                    )
                )
            conn.executemany(
                """
                UPDATE embeddings SET vector=?,vector_dtype=?,vector_sha256=?,vector_l2_norm=?,
                    local_request_id=?,api_request_id=?,response_model=?,response_index=?,embedded_utc=?
                WHERE sha256=? AND vector IS NULL
                """,
                updates,
            )
            usage = payload.get("usage") or {}
            conn.execute(
                """
                UPDATE api_requests SET api_request_id=?,completed_utc=?,http_status=?,prompt_tokens=?,
                    total_tokens=?,attempt_count=?,status='complete' WHERE local_request_id=?
                """,
                (
                    api_request_id,
                    completed,
                    status,
                    usage.get("prompt_tokens"),
                    usage.get("total_tokens"),
                    attempts,
                    local_id,
                ),
            )
            conn.commit()
        except Exception as exc:
            conn.execute(
                """
                UPDATE api_requests SET completed_utc=?,attempt_count=?,status='failed',
                    error_type=?,error_message=? WHERE local_request_id=?
                """,
                (utc_now(), EMBEDDING_MAX_ATTEMPTS, type(exc).__name__, str(exc)[:2000], local_id),
            )
            conn.commit()
            raise
        remaining = int(conn.execute("SELECT COUNT(*) FROM embeddings WHERE vector IS NULL").fetchone()[0])
        LOGGER.info("Embedding cache progress: %s remaining", f"{remaining:,}")


def _write_aligned_embeddings(
    inventory: pd.DataFrame,
    destination: Path,
    vectors: dict[str, bytes],
) -> np.memmap:
    destination.parent.mkdir(parents=True, exist_ok=True)
    array = np.lib.format.open_memmap(
        destination,
        mode="w+",
        dtype=np.float32,
        shape=(len(inventory), EMBEDDING_DIMENSION),
    )
    for row in inventory.itertuples(index=False):
        raw = vectors[str(row.text_sha256)]
        vector = np.frombuffer(raw, dtype="<f4").astype(np.float64)
        norm = float(np.linalg.norm(vector))
        if not math.isfinite(norm) or norm <= 0:
            raise RuntimeError(f"Invalid embedding norm: {row.text_sha256}")
        array[int(row.embedding_index)] = (vector / norm).astype(np.float32)
    array.flush()
    return np.load(destination, mmap_mode="r")


def get_embeddings(
    inventory: pd.DataFrame,
    destination: Path,
    *,
    embeddings_npy: Path | None = None,
    embedding_inventory_csv: Path | None = None,
    embedding_cache: Path | None = None,
) -> np.ndarray:
    """Load supplied embeddings or obtain missing vectors using the established model."""
    required_columns = {"embedding_index", "text_sha256", "text"}
    if not required_columns.issubset(inventory.columns):
        raise ValueError(f"Embedding inventory requires columns {sorted(required_columns)}")
    if any(
        sha256_text(text) != digest
        for digest, text in inventory[["text_sha256", "text"]].itertuples(index=False, name=None)
    ):
        raise RuntimeError("Embedding inventory text hashes do not match exact text")

    if embeddings_npy is not None:
        if embedding_inventory_csv is None:
            raise ValueError("--embedding-inventory is required with --embeddings-npy")
        source = np.load(embeddings_npy, mmap_mode="r")
        source_inventory = pd.read_csv(embedding_inventory_csv)
        if not {"text_sha256", "embedding_index"}.issubset(source_inventory.columns):
            raise ValueError("Supplied embedding inventory lacks text_sha256/embedding_index")
        source_index = dict(
            zip(source_inventory["text_sha256"].astype(str), source_inventory["embedding_index"].astype(int))
        )
        missing = set(inventory["text_sha256"].astype(str)) - set(source_index)
        if missing:
            raise RuntimeError(f"Supplied embeddings lack {len(missing)} required exact texts")
        destination.parent.mkdir(parents=True, exist_ok=True)
        aligned = np.lib.format.open_memmap(
            destination,
            mode="w+",
            dtype=np.float32,
            shape=(len(inventory), EMBEDDING_DIMENSION),
        )
        for row in inventory.itertuples(index=False):
            vector = np.asarray(source[source_index[str(row.text_sha256)]], dtype=np.float64)
            norm = float(np.linalg.norm(vector))
            if vector.shape != (EMBEDDING_DIMENSION,) or not math.isfinite(norm) or norm <= 0:
                raise RuntimeError(f"Invalid supplied embedding: {row.text_sha256}")
            aligned[int(row.embedding_index)] = (vector / norm).astype(np.float32)
        aligned.flush()
        return np.load(destination, mmap_mode="r")

    cache_path = embedding_cache or destination.with_suffix(".sqlite3")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(cache_path, timeout=120) as conn:
        _initialize_embedding_cache(conn, inventory)
        hashes = inventory["text_sha256"].astype(str).tolist()
        missing = 0
        for start in range(0, len(hashes), 700):
            part = hashes[start : start + 700]
            placeholders = ",".join("?" for _ in part)
            missing += int(
                conn.execute(
                    f"SELECT COUNT(*) FROM embeddings WHERE sha256 IN ({placeholders}) AND vector IS NULL",
                    part,
                ).fetchone()[0]
            )
        if missing:
            api_key = os.environ.get("OPENAI_API_KEY", "").strip()
            if not api_key or any(character.isspace() for character in api_key):
                raise RuntimeError("OPENAI_API_KEY is required to generate missing embeddings")
            _embed_missing(conn, api_key)
            api_key = ""
        vectors: dict[str, bytes] = {}
        for start in range(0, len(hashes), 700):
            part = hashes[start : start + 700]
            query_placeholders = ",".join("?" for _ in part)
            for digest, exact_text, raw, model in conn.execute(
                f"SELECT sha256,exact_text,vector,response_model FROM embeddings WHERE sha256 IN ({query_placeholders})",
                part,
            ):
                if sha256_text(exact_text) != digest or model != EMBEDDING_MODEL:
                    raise RuntimeError(f"Embedding cache provenance mismatch: {digest}")
                if raw is None or len(raw) != EMBEDDING_DIMENSION * 4:
                    raise RuntimeError(f"Missing or invalid cached embedding: {digest}")
                vectors[str(digest)] = bytes(raw)
        if len(vectors) != len(inventory):
            raise RuntimeError("Embedding cache did not provide every required exact text")
    return _write_aligned_embeddings(inventory, destination, vectors)


def array_backend(name: str) -> tuple[object, bool]:
    """Return NumPy or CuPy while keeping the statistical code backend-neutral."""
    if name not in {"auto", "cpu", "gpu"}:
        raise ValueError(name)
    if name in {"auto", "gpu"}:
        try:
            os.environ.setdefault("NVIDIA_TF32_OVERRIDE", "0")
            import cupy as cp
            from cupy_backends.cuda.libs import cublas

            if cp.cuda.runtime.getDeviceCount() > 0:
                cublas.setMathMode(cp.cuda.device.get_cublas_handle(), cublas.CUBLAS_DEFAULT_MATH)
                return cp, True
        except (ImportError, RuntimeError):
            if name == "gpu":
                raise
    return np, False


def _to_numpy(value: object, xp: object, is_gpu: bool) -> np.ndarray:
    return xp.asnumpy(value) if is_gpu else np.asarray(value)


def _histogram_quantile(histogram: np.ndarray, quantile: float) -> float:
    count = int(histogram.sum())
    if count == 0:
        return math.nan
    target = quantile * (count - 1)
    cumulative = np.cumsum(histogram, dtype=np.int64)
    index = int(np.searchsorted(cumulative, target + 1, side="left"))
    index = min(max(index, 0), len(histogram) - 1)
    before = int(cumulative[index - 1]) if index else 0
    in_bin = int(histogram[index])
    fraction = 0.5 if in_bin <= 0 else min(max((target - before) / in_bin, 0.0), 1.0)
    return float((index + fraction) * (2.0 / len(histogram)))


def blockwise_cosine_stats(
    matrix: object,
    *,
    xp: object = np,
    is_gpu: bool = False,
    block_size: int = PAIRWISE_BLOCK_SIZE,
) -> dict[str, object]:
    """Scan every unordered pair without allocating a full N-by-N matrix."""
    n = int(matrix.shape[0])
    expected = n * (n - 1) // 2
    if expected == 0:
        return {
            "N": n,
            "pair_count": 0,
            "mean_cosine_distance": math.nan,
            "sd_cosine_distance": math.nan,
            "min_cosine_distance": math.nan,
            "p05_cosine_distance_histogram_approx": math.nan,
            "median_cosine_distance_histogram_approx": math.nan,
            "p95_cosine_distance_histogram_approx": math.nan,
            "max_cosine_distance": math.nan,
            "distance_sum": 0.0,
            "distance_sum_squares": 0.0,
        }
    total = 0.0
    total_squared = 0.0
    minimum = math.inf
    maximum = -math.inf
    count = 0
    histogram = xp.zeros(HISTOGRAM_BINS, dtype=xp.int64)
    edges = xp.linspace(0.0, 2.0, HISTOGRAM_BINS + 1, dtype=xp.float32)
    for left_start in range(0, n, block_size):
        left_stop = min(left_start + block_size, n)
        left = matrix[left_start:left_stop]
        for right_start in range(left_start, n, block_size):
            right_stop = min(right_start + block_size, n)
            raw_distance = 1.0 - left @ matrix[right_start:right_stop].T
            if left_start == right_start:
                values = raw_distance[xp.triu_indices(left_stop - left_start, k=1)]
            else:
                values = raw_distance.ravel()
            if not int(values.size):
                continue
            values = xp.clip(values, 0.0, 2.0)
            values64 = values.astype(xp.float64)
            block = _to_numpy(
                xp.stack(
                    (values64.sum(), (values64 * values64).sum(), values64.min(), values64.max())
                ),
                xp,
                is_gpu,
            )
            total += float(block[0])
            total_squared += float(block[1])
            minimum = min(minimum, float(block[2]))
            maximum = max(maximum, float(block[3]))
            count += int(values.size)
            histogram += xp.histogram(values, bins=edges)[0]
    if count != expected:
        raise RuntimeError(f"Pair count mismatch: observed {count}, expected {expected}")
    mean = total / count
    variance = max(total_squared / count - mean * mean, 0.0)
    histogram_cpu = _to_numpy(histogram, xp, is_gpu)
    return {
        "N": n,
        "pair_count": count,
        "mean_cosine_distance": mean,
        "sd_cosine_distance": math.sqrt(variance),
        "min_cosine_distance": minimum,
        "p05_cosine_distance_histogram_approx": _histogram_quantile(histogram_cpu, 0.05),
        "median_cosine_distance_histogram_approx": _histogram_quantile(histogram_cpu, 0.50),
        "p95_cosine_distance_histogram_approx": _histogram_quantile(histogram_cpu, 0.95),
        "max_cosine_distance": maximum,
        "distance_sum": total,
        "distance_sum_squares": total_squared,
    }
