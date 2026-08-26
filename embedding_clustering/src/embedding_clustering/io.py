"""Input loading, validation, hashing, and JSON helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import (
    EXPECTED_EMBEDDING_SHAPE,
    EXPECTED_MANIFEST_SHA256,
    EXPECTED_MATRIX_SHA256,
    MODEL_NAMES,
    RUN_ORDER,
)


REQUIRED_MANIFEST_COLUMNS = (
    "final_row_index_0_based",
    "interval_id",
    "model_family",
    "run_version",
    "interval_number",
    "start_round",
    "end_round",
)


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest for *path*."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write stable, human-readable JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def load_inputs(
    embeddings_path: Path,
    manifest_path: Path,
    *,
    check_hashes: bool = True,
) -> tuple[np.ndarray, pd.DataFrame, dict[str, Any]]:
    """Load and validate the fixed matrix and row manifest."""

    embeddings_path = embeddings_path.resolve()
    manifest_path = manifest_path.resolve()
    if not embeddings_path.is_file():
        raise FileNotFoundError(f"Embedding matrix not found: {embeddings_path}")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Embedding manifest not found: {manifest_path}")

    matrix_hash = sha256_file(embeddings_path)
    manifest_hash = sha256_file(manifest_path)
    if check_hashes and matrix_hash != EXPECTED_MATRIX_SHA256:
        raise ValueError(
            "Embedding matrix SHA-256 does not match the expected input: "
            f"{matrix_hash}"
        )
    if check_hashes and manifest_hash != EXPECTED_MANIFEST_SHA256:
        raise ValueError(
            "Manifest SHA-256 does not match the expected input: "
            f"{manifest_hash}"
        )

    matrix = np.load(embeddings_path, allow_pickle=False)
    manifest = pd.read_csv(manifest_path)
    validation = validate_inputs(matrix, manifest)
    validation.update(
        {
            "embeddings_file": embeddings_path.name,
            "manifest_file": manifest_path.name,
            "matrix_sha256": matrix_hash,
            "manifest_sha256": manifest_hash,
            "hash_check_enabled": check_hashes,
        }
    )
    return matrix, manifest, validation


def validate_inputs(matrix: np.ndarray, manifest: pd.DataFrame) -> dict[str, Any]:
    """Enforce the exact 5 x 3 x 100 manuscript analysis design."""

    if matrix.shape != EXPECTED_EMBEDDING_SHAPE:
        raise ValueError(
            f"Expected embedding shape {EXPECTED_EMBEDDING_SHAPE}, found {matrix.shape}"
        )
    if matrix.dtype != np.float32:
        raise ValueError(f"Expected float32 embeddings, found {matrix.dtype}")
    if not np.isfinite(matrix).all():
        raise ValueError("Embedding matrix contains NaN or infinite values")

    missing_columns = sorted(set(REQUIRED_MANIFEST_COLUMNS) - set(manifest.columns))
    if missing_columns:
        raise ValueError(f"Manifest is missing columns: {missing_columns}")
    if len(manifest) != EXPECTED_EMBEDDING_SHAPE[0]:
        raise ValueError(f"Expected 1,500 manifest rows, found {len(manifest)}")
    if manifest[list(REQUIRED_MANIFEST_COLUMNS)].isna().any().any():
        raise ValueError("Required manifest fields contain missing values")
    if manifest["final_row_index_0_based"].astype(int).tolist() != list(range(1500)):
        raise ValueError("Manifest row indices are not exactly 0-1499 in matrix order")
    if manifest["interval_id"].duplicated().any():
        raise ValueError("Manifest contains duplicate interval IDs")
    if set(manifest["model_family"]) != set(MODEL_NAMES):
        raise ValueError("Manifest model families do not match the five prespecified families")
    if set(manifest["run_version"]) != set(RUN_ORDER):
        raise ValueError("Manifest runs are not exactly V1, V2, and V3")

    expected_intervals = list(range(1, 101))
    grouped_counts: dict[str, int] = {}
    for family in MODEL_NAMES:
        family_data = manifest.loc[manifest["model_family"].eq(family)]
        if len(family_data) != 300:
            raise ValueError(f"{family}: expected 300 intervals, found {len(family_data)}")
        for run in RUN_ORDER:
            run_data = family_data.loc[family_data["run_version"].eq(run)]
            intervals = sorted(run_data["interval_number"].astype(int).tolist())
            if intervals != expected_intervals:
                raise ValueError(f"{family} {run}: intervals are not exactly 1-100")
            expected_start = (run_data["interval_number"].astype(int) - 1) * 10 + 1
            expected_end = run_data["interval_number"].astype(int) * 10
            if not np.array_equal(run_data["start_round"].astype(int), expected_start):
                raise ValueError(f"{family} {run}: start-round mapping is invalid")
            if not np.array_equal(run_data["end_round"].astype(int), expected_end):
                raise ValueError(f"{family} {run}: end-round mapping is invalid")
            grouped_counts[f"{family}__{run}"] = len(run_data)

    norms = np.linalg.norm(matrix.astype(np.float64), axis=1)
    if not np.allclose(norms, 1.0, atol=2e-6):
        raise ValueError("Input matrix is not L2-normalized within tolerance")

    return {
        "matrix_shape": list(matrix.shape),
        "matrix_dtype": str(matrix.dtype),
        "minimum_norm": float(norms.min()),
        "maximum_norm": float(norms.max()),
        "finite": True,
        "model_families": list(MODEL_NAMES),
        "runs": list(RUN_ORDER),
        "intervals_per_run": 100,
        "rows_per_family_run": grouped_counts,
    }
