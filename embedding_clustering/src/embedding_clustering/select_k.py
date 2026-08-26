"""Repeated-start fitting and geometry-only K selection."""

from __future__ import annotations

import json
from dataclasses import dataclass
from itertools import combinations
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, silhouette_score

from .config import (
    K_VALUES,
    MINIMUM_CLUSTER_SIZE,
    SEEDS,
    SILHOUETTE_TIE_TOLERANCE,
    STABILITY_REFERENCE_ARI,
)
from .spherical_kmeans import (
    SphericalKMeansResult,
    relabel_by_size,
    spherical_kmeans,
)


@dataclass(frozen=True)
class FamilySelection:
    model_family: str
    metrics: pd.DataFrame
    stability: pd.DataFrame
    summary: pd.DataFrame
    best_by_k: dict[int, SphericalKMeansResult]
    best_seed_by_k: dict[int, int]
    selected_k: int
    selected_seed: int
    labels: np.ndarray
    centers: np.ndarray
    cosine_loss: float
    assigned_similarity: np.ndarray
    label_mapping: dict[int, int]
    stability_screen_relaxed: bool


def choose_k_from_summary(
    summary: pd.DataFrame,
) -> tuple[int, bool, pd.DataFrame]:
    """Apply the prespecified K rule to a per-K geometry summary.

    Keeping this decision rule separate allows verification to apply the same
    implementation to saved summaries without repeating every fit.
    """

    required = {
        "k",
        "mean_cosine_silhouette",
        "best_solution_minimum_cluster_size",
        "median_pairwise_ari",
    }
    missing = sorted(required - set(summary.columns))
    if missing:
        raise ValueError(f"K summary is missing columns: {missing}")
    annotated = summary.copy()
    annotated["nonpathological_minimum_size"] = (
        annotated["best_solution_minimum_cluster_size"] >= MINIMUM_CLUSTER_SIZE
    )
    eligible = annotated.loc[annotated["nonpathological_minimum_size"]]
    if eligible.empty:
        raise RuntimeError("Every candidate K failed the minimum-cluster guard")
    maximum_mean_silhouette = float(eligible["mean_cosine_silhouette"].max())
    annotated["within_silhouette_tie_band"] = (
        annotated["nonpathological_minimum_size"]
        & (
            maximum_mean_silhouette - annotated["mean_cosine_silhouette"]
            <= SILHOUETTE_TIE_TOLERANCE
        )
    )
    annotated["stable_reference_ari"] = (
        annotated["median_pairwise_ari"] >= STABILITY_REFERENCE_ARI
    )
    tied = annotated.loc[annotated["within_silhouette_tie_band"]]
    stable_tied = tied.loc[tied["stable_reference_ari"]]
    stability_screen_relaxed = stable_tied.empty
    supported = tied if stability_screen_relaxed else stable_tied
    selected_k = int(supported["k"].min())
    annotated["selected_primary_k"] = annotated["k"].eq(selected_k)
    return selected_k, stability_screen_relaxed, annotated


def _cosine_distance_matrix(matrix: np.ndarray) -> np.ndarray:
    distances = 1.0 - matrix @ matrix.T
    distances = np.maximum(distances, 0.0).astype(np.float64)
    np.fill_diagonal(distances, 0.0)
    return distances


def select_k_for_family(
    model_family: str,
    matrix: np.ndarray,
    *,
    progress: Callable[[str], None] | None = None,
) -> FamilySelection:
    """Evaluate all prespecified K/seed combinations and select the retained fit."""

    if matrix.shape[0] != 300:
        raise ValueError(f"{model_family}: expected 300 vectors, found {len(matrix)}")
    matrix = np.ascontiguousarray(matrix, dtype=np.float32)
    distances = _cosine_distance_matrix(matrix)
    metric_rows: list[dict[str, object]] = []
    stability_rows: list[dict[str, object]] = []
    best_by_k: dict[int, SphericalKMeansResult] = {}
    best_seed_by_k: dict[int, int] = {}

    for k in K_VALUES:
        seed_solutions: list[tuple[int, SphericalKMeansResult]] = []
        best_result: SphericalKMeansResult | None = None
        best_seed: int | None = None
        for seed in SEEDS:
            result = spherical_kmeans(matrix, k, seed)
            sizes = np.bincount(result.labels, minlength=k)
            silhouette = float(
                silhouette_score(distances, result.labels, metric="precomputed")
            )
            metric_rows.append(
                {
                    "model_family": model_family,
                    "k": k,
                    "seed": seed,
                    "cosine_silhouette": silhouette,
                    "cosine_loss": result.cosine_loss,
                    "mean_assigned_cosine": 1.0 - result.cosine_loss / len(matrix),
                    "cluster_sizes": json.dumps([int(value) for value in sizes]),
                    "minimum_cluster_size": int(sizes.min()),
                    "maximum_cluster_size": int(sizes.max()),
                    "cluster_size_ratio_max_min": float(sizes.max() / sizes.min()),
                    "cluster_size_cv": float(sizes.std(ddof=0) / sizes.mean()),
                    "iterations": result.iterations,
                    "converged": result.converged,
                }
            )
            seed_solutions.append((seed, result))
            if best_result is None or result.cosine_loss < best_result.cosine_loss:
                best_result = result
                best_seed = seed

        for (seed_a, first), (seed_b, second) in combinations(seed_solutions, 2):
            stability_rows.append(
                {
                    "model_family": model_family,
                    "k": k,
                    "seed_a": seed_a,
                    "seed_b": seed_b,
                    "adjusted_rand_index": float(
                        adjusted_rand_score(first.labels, second.labels)
                    ),
                }
            )
        assert best_result is not None and best_seed is not None
        best_by_k[k] = best_result
        best_seed_by_k[k] = best_seed
        if progress is not None:
            progress(
                f"{model_family}: K={k}, best seed={best_seed}, "
                f"loss={best_result.cosine_loss:.6f}"
            )

    metrics = pd.DataFrame(metric_rows)
    stability = pd.DataFrame(stability_rows)
    summary_rows: list[dict[str, object]] = []
    for k in K_VALUES:
        fits = metrics.loc[metrics["k"].eq(k)]
        pairwise_ari = stability.loc[
            stability["k"].eq(k), "adjusted_rand_index"
        ]
        best_seed = best_seed_by_k[k]
        best_fit = fits.loc[fits["seed"].eq(best_seed)].iloc[0]
        summary_rows.append(
            {
                "model_family": model_family,
                "k": k,
                "mean_cosine_silhouette": float(fits["cosine_silhouette"].mean()),
                "median_cosine_silhouette": float(
                    fits["cosine_silhouette"].median()
                ),
                "best_fit_cosine_silhouette": float(
                    fits["cosine_silhouette"].max()
                ),
                "worst_fit_cosine_silhouette": float(
                    fits["cosine_silhouette"].min()
                ),
                "mean_cosine_loss": float(fits["cosine_loss"].mean()),
                "best_cosine_loss": float(best_fit["cosine_loss"]),
                "best_seed": int(best_seed),
                "best_solution_silhouette": float(best_fit["cosine_silhouette"]),
                "best_solution_minimum_cluster_size": int(
                    best_fit["minimum_cluster_size"]
                ),
                "best_solution_maximum_cluster_size": int(
                    best_fit["maximum_cluster_size"]
                ),
                "best_solution_size_ratio": float(
                    best_fit["cluster_size_ratio_max_min"]
                ),
                "best_solution_size_cv": float(best_fit["cluster_size_cv"]),
                "median_pairwise_ari": float(pairwise_ari.median()),
                "mean_pairwise_ari": float(pairwise_ari.mean()),
                "minimum_pairwise_ari": float(pairwise_ari.min()),
                "maximum_pairwise_ari": float(pairwise_ari.max()),
                "all_seeds_converged": bool(fits["converged"].all()),
            }
        )

    selected_k, stability_screen_relaxed, summary = choose_k_from_summary(
        pd.DataFrame(summary_rows)
    )
    selected_seed = best_seed_by_k[selected_k]
    selected_fit = best_by_k[selected_k]
    labels, centers, mapping = relabel_by_size(
        selected_fit.labels, selected_fit.centers
    )
    similarities = np.sum(matrix * centers[labels], axis=1, dtype=np.float64)
    return FamilySelection(
        model_family=model_family,
        metrics=metrics,
        stability=stability,
        summary=summary,
        best_by_k=best_by_k,
        best_seed_by_k=best_seed_by_k,
        selected_k=selected_k,
        selected_seed=selected_seed,
        labels=labels,
        centers=centers,
        cosine_loss=selected_fit.cosine_loss,
        assigned_similarity=similarities,
        label_mapping=mapping,
        stability_screen_relaxed=stability_screen_relaxed,
    )
