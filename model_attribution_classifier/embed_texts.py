from __future__ import annotations

"""Fresh OpenAI embedding generation with an exact-text resumable cache."""

import base64
import hashlib
import json
import sqlite3
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_DIMENSION = 3072
API_URL = "https://api.openai.com/v1/embeddings"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class EmbeddingCache:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path, timeout=120)
        self.connection.executescript(
            """
            PRAGMA journal_mode=WAL;
            PRAGMA synchronous=FULL;
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS embeddings (
                sha256 TEXT PRIMARY KEY,
                exact_text TEXT NOT NULL,
                model TEXT NOT NULL,
                dimension INTEGER NOT NULL,
                vector BLOB,
                vector_sha256 TEXT,
                local_request_id TEXT,
                api_request_id TEXT,
                created_utc TEXT
            );
            CREATE TABLE IF NOT EXISTS requests (
                local_request_id TEXT PRIMARY KEY,
                api_request_id TEXT,
                started_utc TEXT NOT NULL,
                completed_utc TEXT,
                endpoint TEXT NOT NULL,
                requested_model TEXT NOT NULL,
                response_model TEXT,
                encoding_format TEXT NOT NULL,
                ordered_hashes_json TEXT NOT NULL,
                request_body_sha256 TEXT NOT NULL,
                input_count INTEGER NOT NULL,
                prompt_tokens INTEGER,
                attempt_count INTEGER NOT NULL,
                status TEXT NOT NULL,
                error TEXT
            );
            """
        )
        metadata = {
            "embedding_model": EMBEDDING_MODEL,
            "embedding_dimension": str(EMBEDDING_DIMENSION),
            "encoding_format": "base64",
            "character_truncation": "none",
            "cache_key": "sha256 of exact UTF-8 canonical string",
            "vector_dtype": "float32-little-endian",
        }
        existing = dict(self.connection.execute("SELECT key, value FROM metadata"))
        if existing:
            for key, value in metadata.items():
                if existing.get(key) != value:
                    raise RuntimeError(f"Cache metadata mismatch for {key}")
        else:
            self.connection.executemany("INSERT INTO metadata(key, value) VALUES (?, ?)", metadata.items())
            self.connection.commit()

    def close(self) -> None:
        self.connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        self.connection.close()

    def __enter__(self) -> "EmbeddingCache":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def register_inputs(self, texts: dict[str, str]) -> None:
        rows = [(digest, text, EMBEDDING_MODEL, EMBEDDING_DIMENSION) for digest, text in sorted(texts.items())]
        self.connection.executemany(
            "INSERT OR IGNORE INTO embeddings(sha256, exact_text, model, dimension) VALUES (?, ?, ?, ?)",
            rows,
        )
        cached_hashes: set[str] = set()
        for digest, text in self.connection.execute("SELECT sha256, exact_text FROM embeddings"):
            cached_hashes.add(digest)
            if digest in texts and texts[digest] != text:
                raise RuntimeError(f"Existing cache text mismatch for {digest}")
        if cached_hashes != set(texts):
            raise RuntimeError("Cache contains inputs outside the current workload; use a dedicated cache path")
        self.connection.commit()

    def missing(self, limit: int) -> list[tuple[str, str]]:
        return self.connection.execute(
            "SELECT sha256, exact_text FROM embeddings WHERE vector IS NULL ORDER BY sha256 LIMIT ?",
            (limit,),
        ).fetchall()

    def store_request(
        self,
        local_id: str,
        api_request_id: str | None,
        rows: list[tuple[str, bytes, int]],
        response_model: str,
        prompt_tokens: int | None,
        attempts: int,
    ) -> None:
        completed = _utc_now()
        for digest, raw, response_index in rows:
            if len(raw) != EMBEDDING_DIMENSION * 4:
                raise RuntimeError(f"Invalid vector size for {digest}")
            vector = np.frombuffer(raw, dtype="<f4")
            if vector.shape != (EMBEDDING_DIMENSION,) or not np.isfinite(vector).all():
                raise RuntimeError(f"Invalid vector values for {digest}")
            self.connection.execute(
                """
                UPDATE embeddings SET vector=?, vector_sha256=?, local_request_id=?,
                    api_request_id=?, created_utc=?
                WHERE sha256=? AND vector IS NULL
                """,
                (raw, hashlib.sha256(raw).hexdigest(), local_id, api_request_id, completed, digest),
            )
        self.connection.execute(
            """
            UPDATE requests SET api_request_id=?, completed_utc=?, prompt_tokens=?,
                attempt_count=?, response_model=?, status='complete' WHERE local_request_id=?
            """,
            (api_request_id, completed, prompt_tokens, attempts, response_model, local_id),
        )
        self.connection.commit()

    def vectors(self, hashes: list[str]) -> dict[str, bytes]:
        wanted = sorted(set(hashes))
        output: dict[str, bytes] = {}
        for start in range(0, len(wanted), 800):
            part = wanted[start : start + 800]
            placeholders = ",".join("?" for _ in part)
            for digest, vector in self.connection.execute(
                f"SELECT sha256, vector FROM embeddings WHERE sha256 IN ({placeholders})", part
            ):
                if vector is not None:
                    output[digest] = vector
        missing = set(wanted) - set(output)
        if missing:
            raise RuntimeError(f"Missing {len(missing)} vectors from cache")
        return output


def _request_embeddings(api_key: str, body: bytes, max_retries: int = 8) -> tuple[dict, str | None, int]:
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        request = urllib.request.Request(
            API_URL,
            data=body,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "User-Agent": "canonical-model-attribution-classifier",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                return json.loads(response.read().decode("utf-8")), response.headers.get("x-request-id"), attempt
        except urllib.error.HTTPError as exc:
            message = exc.read().decode("utf-8", errors="replace")[:2000]
            last_error = RuntimeError(f"HTTP {exc.code}: {message}")
            if exc.code not in {408, 409, 429, 500, 502, 503, 504}:
                raise last_error
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            last_error = exc
        if attempt < max_retries:
            time.sleep(min(30.0, 2.0 ** (attempt - 1)))
    raise RuntimeError(f"Embedding request failed after {max_retries} attempts: {last_error}")


def ensure_embeddings(cache: EmbeddingCache, texts: dict[str, str], api_key: str, batch_size: int = 64) -> None:
    cache.register_inputs(texts)
    while True:
        batch = cache.missing(batch_size)
        if not batch:
            return
        hashes = [row[0] for row in batch]
        body = json.dumps(
            {"model": EMBEDDING_MODEL, "input": [row[1] for row in batch], "encoding_format": "base64"},
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        local_id = str(uuid.uuid4())
        cache.connection.execute(
            """
            INSERT INTO requests(local_request_id, started_utc, endpoint, requested_model,
                encoding_format, ordered_hashes_json, request_body_sha256,
                input_count, attempt_count, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, 'started')
            """,
            (
                local_id,
                _utc_now(),
                API_URL,
                EMBEDDING_MODEL,
                "base64",
                json.dumps(hashes, separators=(",", ":")),
                hashlib.sha256(body).hexdigest(),
                len(batch),
            ),
        )
        cache.connection.commit()
        try:
            payload, api_request_id, attempts = _request_embeddings(api_key, body)
            data = sorted(payload["data"], key=lambda item: int(item["index"]))
            if len(data) != len(batch) or [int(item["index"]) for item in data] != list(range(len(batch))):
                raise RuntimeError("Embedding response indices do not match request order")
            stored_rows = []
            for digest, item in zip(hashes, data):
                raw = base64.b64decode(item["embedding"], validate=True)
                stored_rows.append((digest, raw, int(item["index"])))
            cache.store_request(
                local_id,
                api_request_id,
                stored_rows,
                str(payload.get("model", "")),
                (payload.get("usage") or {}).get("prompt_tokens"),
                attempts,
            )
        except Exception as exc:
            cache.connection.execute(
                "UPDATE requests SET completed_utc=?, attempt_count=?, status='failed', error=? WHERE local_request_id=?",
                (_utc_now(), 8, f"{type(exc).__name__}: {str(exc)[:1800]}", local_id),
            )
            cache.connection.commit()
            raise
