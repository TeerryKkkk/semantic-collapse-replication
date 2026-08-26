"""Deterministic spherical K-means for the embedding-clustering analysis."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import kmeans_plusplus

from .config import CONVERGENCE_TOLERANCE, MAX_ITERATIONS


@dataclass(frozen=True)
class SphericalKMeansResult:
    labels: np.ndarray
    centers: np.ndarray
    cosine_loss: float
    iterations: int
    converged: bool


def spherical_kmeans(
    matrix: np.ndarray,
    k: int,
    seed: int,
    *,
    max_iterations: int = MAX_ITERATIONS,
    tolerance: float = CONVERGENCE_TOLERANCE,
) -> SphericalKMeansResult:
    """Fit spherical K-means using cosine assignment and unit centroids.

    The input must already be L2-normalized. Empty clusters are recovered using
    the least-well-represented points, without consulting metadata.
    """

    centers, _ = kmeans_plusplus(matrix, n_clusters=k, random_state=seed)
    centers = centers.astype(np.float32, copy=False)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    previous_labels: np.ndarray | None = None
    previous_loss = np.inf
    converged = False

    for iteration in range(1, max_iterations + 1):
        similarities = matrix @ centers.T
        labels = np.argmax(similarities, axis=1).astype(np.int32)
        assigned = similarities[np.arange(len(matrix)), labels]
        counts = np.bincount(labels, minlength=k)
        if np.any(counts == 0):
            least_represented = np.argsort(assigned)
            used: set[int] = set()
            for empty_cluster in np.flatnonzero(counts == 0):
                replacement = next(
                    int(index)
                    for index in least_represented
                    if int(index) not in used
                )
                used.add(replacement)
                labels[replacement] = int(empty_cluster)

        new_centers = np.zeros((k, matrix.shape[1]), dtype=np.float32)
        np.add.at(new_centers, labels, matrix)
        center_norms = np.linalg.norm(new_centers, axis=1, keepdims=True)
        if np.any(center_norms == 0):
            raise RuntimeError("An empty cluster remained after deterministic recovery")
        new_centers /= center_norms

        new_similarities = matrix @ new_centers.T
        loss = float(
            np.sum(
                1.0 - new_similarities[np.arange(len(matrix)), labels],
                dtype=np.float64,
            )
        )
        labels_unchanged = previous_labels is not None and np.array_equal(
            labels, previous_labels
        )
        objective_stable = abs(previous_loss - loss) <= tolerance
        centers = new_centers
        if labels_unchanged or objective_stable:
            converged = True
            break
        previous_labels = labels.copy()
        previous_loss = loss

    final_similarities = matrix @ centers.T
    final_labels = np.argmax(final_similarities, axis=1).astype(np.int32)
    final_loss = float(
        np.sum(
            1.0 - final_similarities[np.arange(len(matrix)), final_labels],
            dtype=np.float64,
        )
    )
    return SphericalKMeansResult(
        labels=final_labels,
        centers=centers,
        cosine_loss=final_loss,
        iterations=iteration,
        converged=converged,
    )


def relabel_by_size(
    labels: np.ndarray, centers: np.ndarray
) -> tuple[np.ndarray, np.ndarray, dict[int, int]]:
    """Relabel clusters deterministically from largest to smallest."""

    counts = np.bincount(labels, minlength=centers.shape[0])
    order = sorted(range(len(counts)), key=lambda value: (-int(counts[value]), value))
    mapping = {old: new for new, old in enumerate(order)}
    relabeled = np.array([mapping[int(value)] for value in labels], dtype=np.int32)
    return relabeled, centers[order], mapping
