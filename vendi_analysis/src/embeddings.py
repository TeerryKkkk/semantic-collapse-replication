from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

from config import CACHE_DIR, EMBEDDING_DIMENSION, EMBEDDING_MODEL


def text_id(text: str) -> str:
    payload = f"{EMBEDDING_MODEL}\0{EMBEDDING_DIMENSION}\0{text}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_or_create_embeddings(texts: set[str]) -> dict[str, np.ndarray]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    ordered = sorted(texts, key=lambda value: (text_id(value), value))
    index_path = CACHE_DIR / "baseline_block_embedding_index.csv.gz"
    matrix_path = CACHE_DIR / "baseline_block_embeddings.npy"
    expected_ids = [text_id(value) for value in ordered]
    if index_path.exists() and matrix_path.exists():
        index = pd.read_csv(index_path)
        matrix = np.load(matrix_path, mmap_mode="r")
        if index.text_id.astype(str).tolist() != expected_ids:
            raise RuntimeError("Embedding cache does not match the requested exact block texts")
        if matrix.shape != (len(index), EMBEDDING_DIMENSION):
            raise RuntimeError("Embedding cache has the wrong shape")
        return {value: np.asarray(matrix[i], dtype=np.float32) for i, value in enumerate(ordered)}

    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        raise RuntimeError("OPENAI_API_KEY is required when no local embedding cache is available")
    from openai import OpenAI

    client = OpenAI(api_key=key)
    matrix = np.lib.format.open_memmap(
        matrix_path, mode="w+", dtype=np.float32, shape=(len(ordered), EMBEDDING_DIMENSION)
    )
    for start in range(0, len(ordered), 512):
        stop = min(len(ordered), start + 512)
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=ordered[start:stop],
            encoding_format="float",
        )
        values = np.asarray([item.embedding for item in response.data], dtype=np.float32)
        if values.shape != (stop - start, EMBEDDING_DIMENSION) or not np.isfinite(values).all():
            raise RuntimeError("Invalid embedding API response")
        matrix[start:stop] = values
        matrix.flush()
    pd.DataFrame({"text_id": expected_ids}).to_csv(index_path, index=False, compression="gzip")
    return {value: np.asarray(matrix[i], dtype=np.float32) for i, value in enumerate(ordered)}
