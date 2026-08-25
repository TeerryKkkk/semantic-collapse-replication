from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from config import EMBEDDING_DIMENSION, M, RAREFACTION_DRAWS, SEED_BASE


EPS = 1e-12
NEGATIVE_EIGENVALUE_HARD_TOLERANCE = -1e-6


def deterministic_seed(*parts: object) -> int:
    payload = "|".join(str(part) for part in (SEED_BASE, *parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**32)


def normalized_vendi_batch(vectors: np.ndarray) -> np.ndarray:
    values = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(values, axis=2, keepdims=True)
    if np.any(norms <= 0) or not np.isfinite(norms).all():
        raise RuntimeError("Invalid embedding norm")
    normalized = values / norms
    grams = np.matmul(normalized, np.swapaxes(normalized, 1, 2)).astype(np.float64)
    grams = np.clip(grams, -1.0, 1.0)
    traces = np.trace(grams, axis1=1, axis2=2)
    if np.any(traces <= 0) or not np.isfinite(traces).all():
        raise RuntimeError("Invalid Gram trace")
    grams /= traces[:, None, None]
    eigenvalues = np.linalg.eigvalsh(grams)
    if float(eigenvalues.min()) < NEGATIVE_EIGENVALUE_HARD_TOLERANCE:
        raise RuntimeError("Non-numerical negative Gram eigenvalue")
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    eigenvalues /= eigenvalues.sum(axis=1, keepdims=True)
    entropy = -np.sum(
        np.where(eigenvalues > 0, eigenvalues * np.log(eigenvalues + EPS), 0.0),
        axis=1,
    )
    return np.exp(entropy) / M


def score_block_pools(
    pools: dict[tuple[str, str], list[str]],
    unit_index: pd.DataFrame,
    embeddings: dict[str, np.ndarray],
) -> pd.DataFrame:
    phase_rows: list[dict[str, object]] = []
    for (unit_id, phase), pool in sorted(pools.items()):
        rng = np.random.default_rng(deterministic_seed("rarefaction", unit_id, phase))
        selected = np.vstack(
            [rng.choice(len(pool), size=M, replace=False) for _ in range(RAREFACTION_DRAWS)]
        )
        vectors = np.empty((RAREFACTION_DRAWS, M, EMBEDDING_DIMENSION), dtype=np.float32)
        for replicate in range(RAREFACTION_DRAWS):
            for position in range(M):
                vectors[replicate, position] = embeddings[pool[int(selected[replicate, position])]]
        scores = normalized_vendi_batch(vectors)
        phase_rows.append(
            {
                "unit_id": unit_id,
                "phase": phase,
                "normalized_vendi": float(scores.mean()),
            }
        )

    phase_frame = pd.DataFrame(phase_rows)
    wide = phase_frame.pivot(index="unit_id", columns="phase", values="normalized_vendi").reset_index()
    wide = wide.merge(unit_index, on="unit_id", validate="one_to_one")
    wide["late_minus_early"] = wide["late"] - wide["early"]
    return wide[["unit_id", "family", "early", "middle", "late", "late_minus_early"]]
