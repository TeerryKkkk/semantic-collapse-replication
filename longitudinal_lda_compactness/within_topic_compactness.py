"""Measure semantic compactness after assignment by an existing fixed LDA model."""

from __future__ import annotations

import argparse
import logging
import math
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from config import (
    BETWEEN_SAMPLE_PER_TOPIC_PAIR,
    BETWEEN_SAMPLE_SEED,
    EMBEDDING_DIMENSION,
    HIGH_CONFIDENCE_THRESHOLD,
    PAIRWISE_BLOCK_SIZE,
    PRIMARY_K,
    PRIMARY_SEED,
)
from parse_data import load_baseline_corpus
from utils import array_backend, blockwise_cosine_stats, get_embeddings, sha256_text


LOGGER = logging.getLogger(__name__)


def assign_topics(
    utterances: pd.DataFrame,
    vectorizer: object,
    lda_model: object,
) -> pd.DataFrame:
    """Assign utterances in the fixed bag-of-words topic space without refitting."""
    X = vectorizer.transform(utterances["text"].astype(str).tolist())
    token_counts = np.asarray(X.sum(axis=1)).ravel().astype(np.int64)
    valid = token_counts > 0
    theta = np.full((len(utterances), PRIMARY_K), np.nan, dtype=np.float64)
    theta[valid] = lda_model.transform(X[valid])
    if valid.any() and np.max(np.abs(theta[valid].sum(axis=1) - 1.0)) > 1e-10:
        raise RuntimeError("Utterance topic posteriors do not sum to one")

    dominant = np.full(len(utterances), -1, dtype=np.int16)
    probability = np.full(len(utterances), np.nan, dtype=np.float64)
    dominant[valid] = np.argmax(theta[valid], axis=1).astype(np.int16)
    probability[valid] = np.max(theta[valid], axis=1)

    assignments = utterances.copy()
    assignments["text_sha256"] = assignments["text"].map(sha256_text)
    assignments["dominant_topic"] = pd.array(np.where(valid, dominant, None), dtype="Int64")
    assignments["dominant_topic_probability"] = probability
    assignments["in_vocabulary_token_count"] = token_counts
    assignments["valid_lda_assignment"] = valid
    for topic in range(PRIMARY_K):
        assignments[f"topic_probability_{topic}"] = theta[:, topic]
    return assignments


def make_embedding_inventory(
    assignments: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray]:
    valid = assignments["valid_lda_assignment"].astype(bool).to_numpy()
    source = assignments.loc[valid, ["text_sha256", "text"]].copy()
    if (source.groupby("text_sha256")["text"].nunique() > 1).any():
        raise RuntimeError("SHA-256 collision detected in utterance text")
    inventory = source.drop_duplicates("text_sha256").sort_values("text_sha256").reset_index(drop=True)
    inventory.insert(0, "embedding_index", np.arange(len(inventory), dtype=np.int64))
    lookup = dict(zip(inventory["text_sha256"], inventory["embedding_index"].astype(int)))
    occurrence_index = np.asarray(
        [lookup.get(digest, -1) for digest in assignments["text_sha256"].astype(str)],
        dtype=np.int64,
    )
    if np.any(occurrence_index[valid] < 0) or np.any(occurrence_index[~valid] >= 0):
        raise RuntimeError("Embedding indices are inconsistent with valid LDA assignments")
    return inventory, occurrence_index


def within_topic_distances(
    assignments: pd.DataFrame,
    occurrence_index: np.ndarray,
    unique_embeddings: object,
    subset: np.ndarray,
    *,
    xp: object,
    is_gpu: bool,
    block_size: int,
) -> pd.DataFrame:
    dominant = assignments["dominant_topic"].fillna(-1).to_numpy(int)
    rows: list[dict[str, object]] = []
    for topic in range(PRIMARY_K):
        positions = np.flatnonzero(subset & (dominant == topic))
        indices = occurrence_index[positions]
        if np.any(indices < 0):
            raise RuntimeError(f"Topic {topic} contains an utterance without an embedding")
        LOGGER.info(
            "Scanning topic %d: N=%s, pairs=%s",
            topic,
            f"{len(indices):,}",
            f"{len(indices) * (len(indices) - 1) // 2:,}",
        )
        matrix = (
            unique_embeddings[xp.asarray(indices, dtype=xp.int64)]
            if len(indices)
            else xp.empty((0, EMBEDDING_DIMENSION), dtype=xp.float32)
        )
        rows.append(
            {
                "topic_id": topic,
                **blockwise_cosine_stats(
                    matrix,
                    xp=xp,
                    is_gpu=is_gpu,
                    block_size=block_size,
                ),
            }
        )
        del matrix
        if is_gpu:
            xp.get_default_memory_pool().free_all_blocks()
    return pd.DataFrame(rows)


def _backend_array(values: object, xp: object, is_gpu: bool) -> np.ndarray:
    return xp.asnumpy(values) if is_gpu else np.asarray(values)


def between_topic_sample(
    assignments: pd.DataFrame,
    occurrence_index: np.ndarray,
    unique_embeddings: object,
    valid: np.ndarray,
    *,
    xp: object,
    is_gpu: bool,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    """Sample unique pairs within each unordered topic-pair stratum reproducibly."""
    dominant = assignments["dominant_topic"].fillna(-1).to_numpy(int)
    by_topic = {
        topic: occurrence_index[np.flatnonzero(valid & (dominant == topic))]
        for topic in range(PRIMARY_K)
    }
    rng = np.random.default_rng(BETWEEN_SAMPLE_SEED)
    rows: list[dict[str, object]] = []
    weighted_sum = 0.0
    weighted_second = 0.0
    population_pairs = 0
    total_sample = 0
    sample_maximum = -math.inf

    for left_topic in range(PRIMARY_K):
        left_indices = by_topic[left_topic]
        if not len(left_indices):
            continue
        for right_topic in range(left_topic + 1, PRIMARY_K):
            right_indices = by_topic[right_topic]
            if not len(right_indices):
                continue
            population = int(len(left_indices) * len(right_indices))
            sample_size = min(BETWEEN_SAMPLE_PER_TOPIC_PAIR, population)
            flat = rng.choice(population, size=sample_size, replace=False)
            left_rows = flat // len(right_indices)
            right_rows = flat % len(right_indices)
            total = 0.0
            total_squared = 0.0
            minimum = math.inf
            maximum = -math.inf
            for start in range(0, sample_size, 4096):
                stop = min(start + 4096, sample_size)
                left = unique_embeddings[
                    xp.asarray(left_indices[left_rows[start:stop]], dtype=xp.int64)
                ]
                right = unique_embeddings[
                    xp.asarray(right_indices[right_rows[start:stop]], dtype=xp.int64)
                ]
                values = xp.clip(1.0 - xp.sum(left * right, axis=1), 0.0, 2.0).astype(xp.float64)
                block = _backend_array(
                    xp.stack((values.sum(), (values * values).sum(), values.min(), values.max())),
                    xp,
                    is_gpu,
                )
                total += float(block[0])
                total_squared += float(block[1])
                minimum = min(minimum, float(block[2]))
                maximum = max(maximum, float(block[3]))
            mean = total / sample_size
            variance = max(total_squared / sample_size - mean * mean, 0.0)
            rows.append(
                {
                    "left_topic": left_topic,
                    "right_topic": right_topic,
                    "left_N": len(left_indices),
                    "right_N": len(right_indices),
                    "exact_between_pair_population": population,
                    "sample_size": sample_size,
                    "sample_mean_cosine_distance": mean,
                    "sample_sd_cosine_distance": math.sqrt(variance),
                    "sample_min_cosine_distance": minimum,
                    "sample_max_cosine_distance": maximum,
                }
            )
            weighted_sum += population * mean
            weighted_second += population * (variance + mean * mean)
            population_pairs += population
            total_sample += sample_size
            sample_maximum = max(sample_maximum, maximum)

    total_pairs = int(valid.sum() * (valid.sum() - 1) // 2)
    within_pairs = sum(len(indices) * (len(indices) - 1) // 2 for indices in by_topic.values())
    if population_pairs != total_pairs - within_pairs:
        raise RuntimeError("Between-topic pair-population accounting failed")
    mean = weighted_sum / population_pairs
    variance = max(weighted_second / population_pairs - mean * mean, 0.0)
    return pd.DataFrame(rows), {
        "exact_pair_population": population_pairs,
        "sample_size": total_sample,
        "mean": mean,
        "sd": math.sqrt(variance),
        "sample_maximum": sample_maximum,
        "sample_seed": BETWEEN_SAMPLE_SEED,
        "sample_per_topic_pair": BETWEEN_SAMPLE_PER_TOPIC_PAIR,
    }


def pooled_pair_summary(table: pd.DataFrame) -> dict[str, float | int]:
    estimable = table.loc[table["pair_count"] > 0]
    count = int(estimable["pair_count"].sum())
    total = float(estimable["distance_sum"].sum())
    total_squared = float(estimable["distance_sum_squares"].sum())
    mean = total / count
    return {
        "pair_count": count,
        "mean": mean,
        "sd": math.sqrt(max(total_squared / count - mean * mean, 0.0)),
        "maximum": float(estimable["max_cosine_distance"].max()),
    }


def topic_equal_summary(table: pd.DataFrame) -> dict[str, float | int]:
    means = table.loc[table["pair_count"] > 0, "mean_cosine_distance"].to_numpy(float)
    return {
        "estimable_topics": int(len(means)),
        "mean": float(means.mean()),
        "sd": float(means.std(ddof=1)) if len(means) > 1 else math.nan,
    }


def summary_table(
    within: pd.DataFrame,
    high_confidence: pd.DataFrame,
    between: dict[str, float | int],
    valid_count: int,
    high_confidence_count: int,
) -> pd.DataFrame:
    primary_pairs = pooled_pair_summary(within)
    primary_topics = topic_equal_summary(within)
    sensitivity_pairs = pooled_pair_summary(high_confidence)
    sensitivity_topics = topic_equal_summary(high_confidence)
    retained = 100.0 * high_confidence_count / valid_count
    return pd.DataFrame(
        [
            {
                "analysis_subset": "all valid assignments",
                "summary": "within_topic_topic_equal",
                "utterances": valid_count,
                "retained_percent_of_valid": 100.0,
                "estimable_topics": primary_topics["estimable_topics"],
                "exact_pair_population": primary_pairs["pair_count"],
                "evaluated_pair_count": primary_pairs["pair_count"],
                "mean_distance": primary_topics["mean"],
                "sd_distance": primary_topics["sd"],
                "max_distance": math.nan,
                "method": "equal weight per estimable topic mean",
            },
            {
                "analysis_subset": "all valid assignments",
                "summary": "within_topic_pair_weighted",
                "utterances": valid_count,
                "retained_percent_of_valid": 100.0,
                "estimable_topics": primary_topics["estimable_topics"],
                "exact_pair_population": primary_pairs["pair_count"],
                "evaluated_pair_count": primary_pairs["pair_count"],
                "mean_distance": primary_pairs["mean"],
                "sd_distance": primary_pairs["sd"],
                "max_distance": primary_pairs["maximum"],
                "method": "exact scan; equal weight per within-topic pair",
            },
            {
                "analysis_subset": "all valid assignments",
                "summary": "between_topic_pair_weighted_sample",
                "utterances": valid_count,
                "retained_percent_of_valid": 100.0,
                "estimable_topics": primary_topics["estimable_topics"],
                "exact_pair_population": between["exact_pair_population"],
                "evaluated_pair_count": between["sample_size"],
                "mean_distance": between["mean"],
                "sd_distance": between["sd"],
                "max_distance": between["sample_maximum"],
                "method": (
                    f"seed {BETWEEN_SAMPLE_SEED}; up to {BETWEEN_SAMPLE_PER_TOPIC_PAIR} unique pairs "
                    "per nonempty unordered topic-pair stratum; exact population weights"
                ),
            },
            {
                "analysis_subset": "dominant probability >= 0.50",
                "summary": "high_confidence_topic_equal",
                "utterances": high_confidence_count,
                "retained_percent_of_valid": retained,
                "estimable_topics": sensitivity_topics["estimable_topics"],
                "exact_pair_population": sensitivity_pairs["pair_count"],
                "evaluated_pair_count": sensitivity_pairs["pair_count"],
                "mean_distance": sensitivity_topics["mean"],
                "sd_distance": sensitivity_topics["sd"],
                "max_distance": math.nan,
                "method": "fixed 0.50 threshold; equal weight per estimable topic mean",
            },
            {
                "analysis_subset": "dominant probability >= 0.50",
                "summary": "high_confidence_pair_weighted",
                "utterances": high_confidence_count,
                "retained_percent_of_valid": retained,
                "estimable_topics": sensitivity_topics["estimable_topics"],
                "exact_pair_population": sensitivity_pairs["pair_count"],
                "evaluated_pair_count": sensitivity_pairs["pair_count"],
                "mean_distance": sensitivity_pairs["mean"],
                "sd_distance": sensitivity_pairs["sd"],
                "max_distance": sensitivity_pairs["maximum"],
                "method": "exact scan; fixed 0.50 threshold; equal weight per pair",
            },
        ]
    )


def run_analysis(arguments: argparse.Namespace) -> None:
    compactness_dir = arguments.output_dir / "compactness"
    cache_dir = compactness_dir / "cache"
    compactness_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    vectorizer = joblib.load(arguments.vectorizer)
    lda_model = joblib.load(arguments.lda_model)
    if getattr(lda_model, "n_components", None) != PRIMARY_K or getattr(
        lda_model, "random_state", None
    ) != PRIMARY_SEED:
        raise RuntimeError("The supplied LDA model is not the fixed K=20, seed=42 model")

    _, utterances, _, _ = load_baseline_corpus(arguments.raw_dir)
    assignments = assign_topics(utterances, vectorizer, lda_model)
    assignments.to_csv(compactness_dir / "utterance_topic_assignments.csv", index=False)
    inventory, occurrence_index = make_embedding_inventory(assignments)
    inventory.to_csv(compactness_dir / "embedding_inventory.csv", index=False)

    aligned_path = cache_dir / "unique_utterance_embeddings_float32.npy"
    embedding_array = get_embeddings(
        inventory,
        aligned_path,
        embeddings_npy=arguments.embeddings_npy,
        embedding_inventory_csv=arguments.embedding_inventory,
        embedding_cache=arguments.embedding_cache,
    )
    os.environ.setdefault("CUPY_CACHE_DIR", str(cache_dir / "cupy"))
    xp, is_gpu = array_backend(arguments.backend)
    unique_embeddings = xp.asarray(embedding_array) if is_gpu else embedding_array

    valid = assignments["valid_lda_assignment"].astype(bool).to_numpy()
    high_confidence_mask = valid & (
        assignments["dominant_topic_probability"].to_numpy(float) >= HIGH_CONFIDENCE_THRESHOLD
    )
    within = within_topic_distances(
        assignments,
        occurrence_index,
        unique_embeddings,
        valid,
        xp=xp,
        is_gpu=is_gpu,
        block_size=arguments.block_size,
    )
    high_confidence = within_topic_distances(
        assignments,
        occurrence_index,
        unique_embeddings,
        high_confidence_mask,
        xp=xp,
        is_gpu=is_gpu,
        block_size=arguments.block_size,
    )
    between_strata, between = between_topic_sample(
        assignments,
        occurrence_index,
        unique_embeddings,
        valid,
        xp=xp,
        is_gpu=is_gpu,
    )
    within.to_csv(compactness_dir / "within_topic_distances.csv", index=False)
    high_confidence.to_csv(
        compactness_dir / "within_topic_distances_high_confidence.csv", index=False
    )
    between_strata.to_csv(compactness_dir / "between_topic_strata.csv", index=False)
    summary_table(
        within,
        high_confidence,
        between,
        int(valid.sum()),
        int(high_confidence_mask.sum()),
    ).to_csv(compactness_dir / "within_between_summary.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True, help="Directory containing simulation .txt logs")
    parser.add_argument("--vectorizer", type=Path, required=True, help="Fitted CountVectorizer joblib file")
    parser.add_argument("--lda-model", type=Path, required=True, help="Fixed K=20, seed=42 LDA joblib file")
    parser.add_argument("--output-dir", type=Path, required=True, help="Destination for generated analysis files")
    parser.add_argument("--embeddings-npy", type=Path, help="Optional precomputed embedding matrix")
    parser.add_argument(
        "--embedding-inventory",
        type=Path,
        help="CSV mapping text_sha256 to embedding_index for --embeddings-npy",
    )
    parser.add_argument(
        "--embedding-cache",
        type=Path,
        help="Optional resumable exact-text SQLite cache; missing vectors use OPENAI_API_KEY",
    )
    parser.add_argument("--backend", choices=("auto", "cpu", "gpu"), default="auto")
    parser.add_argument("--block-size", type=int, default=PAIRWISE_BLOCK_SIZE)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    arguments = parse_args()
    if arguments.embeddings_npy is not None and arguments.embedding_inventory is None:
        raise SystemExit("--embedding-inventory is required with --embeddings-npy")
    run_analysis(arguments)


if __name__ == "__main__":
    main()
