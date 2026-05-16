"""
Cache-only cross-run semantic support / concentration analysis.

This script intentionally does not import OpenAI, read an API key, or recompute
embeddings. It uses the message-level metadata CSV and embedding NPY cache files
created by compute_embedding_diversity_vendi.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd


# =====================
# ====== CONFIG =======
# =====================

OUTPUT_DIR = Path("EMBEDDING_DIVERSITY_RESULTS")
CACHE_DIR = OUTPUT_DIR / "cache"
CROSS_RUN_DIR = OUTPUT_DIR / "cross_run"

EXPECTED_DIM = 3072
EXPECTED_EMBEDDING_MODEL = "text-embedding-3-large"

WINDOW_SIZE = 10
HOP = 10
EXPECTED_MAX_ROUND = 1000
EXPECTED_WINDOWS = 100

REQUIRE_ALL_EXPECTED_CACHES = True
MAKE_FIGURES = True

MODEL_FAMILY_SPECS: Sequence[Tuple[str, str, Sequence[int]]] = (
    ("DeepSeek", "3_deepseek_1000_v", range(1, 8)),
    ("GPT", "3_gpt_1000_v", range(1, 6)),
    ("Phi-4", "3_phi-4_1000_v", range(1, 6)),
)

REQUIRED_METADATA_COLUMNS = ["source_file", "family", "run_id", "round", "embedding_model"]


@dataclass(frozen=True)
class CacheSpec:
    model_family: str
    stem: str
    source_file: str
    run_id: str
    metadata_path: Path
    embeddings_path: Path


def build_cache_specs() -> List[CacheSpec]:
    specs: List[CacheSpec] = []
    for model_family, prefix, versions in MODEL_FAMILY_SPECS:
        for version in versions:
            stem = f"{prefix}{version}"
            specs.append(
                CacheSpec(
                    model_family=model_family,
                    stem=stem,
                    source_file=f"{stem}.txt",
                    run_id=f"v{version}",
                    metadata_path=CACHE_DIR / f"{stem}_metadata.csv",
                    embeddings_path=CACHE_DIR / f"{stem}_embeddings.npy",
                )
            )
    return specs


def expected_window_count(max_round: int) -> int:
    if max_round < WINDOW_SIZE:
        return 0
    return ((max_round - WINDOW_SIZE) // HOP) + 1


def load_metadata_minimal(metadata_path: Path) -> pd.DataFrame:
    return pd.read_csv(metadata_path, usecols=REQUIRED_METADATA_COLUMNS)


def audit_cache(specs: Sequence[CacheSpec]) -> Tuple[pd.DataFrame, List[CacheSpec]]:
    rows: List[Dict[str, object]] = []
    selected_specs: List[CacheSpec] = []

    for spec in specs:
        metadata_exists = spec.metadata_path.exists()
        embeddings_exists = spec.embeddings_path.exists()
        found = bool(metadata_exists and embeddings_exists)

        row: Dict[str, object] = {
            "source_file": spec.source_file,
            "model_family": spec.model_family,
            "run_id": spec.run_id,
            "metadata_path": str(spec.metadata_path),
            "embeddings_path": str(spec.embeddings_path),
            "metadata_exists": metadata_exists,
            "embeddings_exists": embeddings_exists,
            "found": found,
            "metadata_rows": np.nan,
            "embedding_shape": "",
            "embedding_dtype": "",
            "row_counts_match": False,
            "embedding_dim_is_3072": False,
            "embedding_model_is_expected": False,
            "max_parsed_round": np.nan,
            "expected_number_of_windows": np.nan,
            "valid": False,
            "issue": "",
        }

        issues: List[str] = []
        metadata_rows = 0
        embedding_shape: Tuple[int, ...] = tuple()
        metadata: pd.DataFrame | None = None
        embeddings: np.ndarray | None = None

        if not metadata_exists:
            issues.append("missing metadata CSV")
        if not embeddings_exists:
            issues.append("missing embeddings NPY")

        if found:
            try:
                metadata = load_metadata_minimal(spec.metadata_path)
                metadata_rows = int(len(metadata))
                row["metadata_rows"] = metadata_rows
            except Exception as exc:  # pragma: no cover - reports exact runtime issue
                issues.append(f"metadata read failed: {exc}")

            try:
                embeddings = np.load(spec.embeddings_path, mmap_mode="r", allow_pickle=False)
                embedding_shape = tuple(int(x) for x in embeddings.shape)
                row["embedding_shape"] = str(embedding_shape)
                row["embedding_dtype"] = str(embeddings.dtype)
            except Exception as exc:  # pragma: no cover - reports exact runtime issue
                issues.append(f"embedding read failed: {exc}")

        if metadata is not None:
            rounds = pd.to_numeric(metadata["round"], errors="coerce")
            if rounds.isna().any():
                issues.append("metadata round column contains non-numeric values")
            else:
                max_round = int(rounds.max()) if len(rounds) else 0
                n_windows = expected_window_count(max_round)
                row["max_parsed_round"] = max_round
                row["expected_number_of_windows"] = n_windows
                if max_round != EXPECTED_MAX_ROUND:
                    issues.append(f"max parsed round is {max_round}, expected {EXPECTED_MAX_ROUND}")
                if n_windows != EXPECTED_WINDOWS:
                    issues.append(f"expected windows is {n_windows}, expected {EXPECTED_WINDOWS}")

            model_values = metadata["embedding_model"].dropna().astype(str).unique().tolist()
            if model_values == [EXPECTED_EMBEDDING_MODEL]:
                row["embedding_model_is_expected"] = True
            else:
                issues.append(f"embedding_model values are {model_values}, expected {[EXPECTED_EMBEDDING_MODEL]}")

            source_values = metadata["source_file"].dropna().astype(str).unique().tolist()
            if source_values and source_values != [spec.source_file]:
                issues.append(f"source_file values are {source_values}, expected {[spec.source_file]}")

            run_values = metadata["run_id"].dropna().astype(str).unique().tolist()
            if run_values and run_values != [spec.run_id]:
                issues.append(f"run_id values are {run_values}, expected {[spec.run_id]}")

        if embeddings is not None:
            if len(embedding_shape) != 2:
                issues.append(f"embedding matrix is {len(embedding_shape)}D, expected 2D")
            elif embedding_shape[1] != EXPECTED_DIM:
                issues.append(f"embedding dimension is {embedding_shape[1]}, expected {EXPECTED_DIM}")
            else:
                row["embedding_dim_is_3072"] = True

            if not np.issubdtype(embeddings.dtype, np.floating):
                issues.append(f"embedding dtype is {embeddings.dtype}, expected floating")

        if metadata is not None and embeddings is not None:
            row_counts_match = metadata_rows == int(embedding_shape[0]) if len(embedding_shape) >= 1 else False
            row["row_counts_match"] = row_counts_match
            if not row_counts_match:
                issues.append(f"metadata rows {metadata_rows} != embedding rows {embedding_shape[0]}")

        valid = found and not issues
        row["valid"] = valid
        row["issue"] = "; ".join(issues)
        rows.append(row)

        if found and (valid or not REQUIRE_ALL_EXPECTED_CACHES):
            selected_specs.append(spec)

    audit_df = pd.DataFrame(rows)
    return audit_df, selected_specs


def require_valid_audit(audit_df: pd.DataFrame) -> None:
    if audit_df["valid"].all():
        return

    invalid = audit_df.loc[~audit_df["valid"], ["source_file", "metadata_path", "embeddings_path", "issue"]]
    print("[ERROR] Cache audit failed. No analysis was run.")
    print(invalid.to_string(index=False))
    raise SystemExit(1)


def normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise ValueError("Encountered zero-norm message embedding")
    return x / norms


def normalize_vector(x: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(x))
    if norm <= 0:
        raise ValueError("Encountered zero-norm centroid")
    return x / norm


def compute_window_centroids(specs: Sequence[CacheSpec]) -> Tuple[np.ndarray, pd.DataFrame]:
    centroids: List[np.ndarray] = []
    metadata_rows: List[Dict[str, object]] = []

    for spec in specs:
        metadata = load_metadata_minimal(spec.metadata_path)
        rounds = pd.to_numeric(metadata["round"], errors="raise").to_numpy(dtype=np.int64)
        embeddings = np.load(spec.embeddings_path, mmap_mode="r", allow_pickle=False)

        for window_index in range(1, EXPECTED_WINDOWS + 1):
            round_start = 1 + (window_index - 1) * HOP
            round_end = round_start + WINDOW_SIZE - 1
            positions = np.flatnonzero((rounds >= round_start) & (rounds <= round_end))
            n_messages = int(len(positions))
            if n_messages == 0:
                raise ValueError(f"No messages found for {spec.source_file} window {window_index}")

            message_vectors = np.asarray(embeddings[positions], dtype=np.float32)
            message_vectors = normalize_rows(message_vectors)
            centroid = normalize_vector(message_vectors.mean(axis=0))
            centroids.append(centroid.astype(np.float32, copy=False))
            metadata_rows.append(
                {
                    "source_file": spec.source_file,
                    "model_family": spec.model_family,
                    "run_id": spec.run_id,
                    "window_index": window_index,
                    "round_start": round_start,
                    "round_end": round_end,
                    "n_messages": n_messages,
                }
            )

    centroid_matrix = np.vstack(centroids).astype(np.float32, copy=False)
    centroid_metadata = pd.DataFrame(metadata_rows)
    return centroid_matrix, centroid_metadata


def vendi_from_cosine_kernel(c: np.ndarray) -> Tuple[float, float]:
    r = int(c.shape[0])
    kernel = c @ c.T
    scaled_kernel = kernel / float(r)
    eigvals = np.linalg.eigvalsh(scaled_kernel).astype(np.float64)
    eigvals = np.clip(eigvals, 0.0, None)
    eig_sum = float(eigvals.sum())
    if eig_sum <= 0:
        raise ValueError("Cross-run kernel eigenvalues sum to zero")
    if not np.isclose(eig_sum, 1.0, rtol=1e-6, atol=1e-8):
        eigvals = eigvals / eig_sum
    positive = eigvals[eigvals > 0]
    entropy = -float(np.sum(positive * np.log(positive)))
    vendi = float(np.exp(entropy))
    return vendi, vendi / float(r)


def upper_triangle_values(matrix: np.ndarray) -> np.ndarray:
    row_idx, col_idx = np.triu_indices(matrix.shape[0], k=1)
    return matrix[row_idx, col_idx]


def compute_cross_run_metrics(
    centroids: np.ndarray,
    centroid_metadata: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows: List[Dict[str, object]] = []
    pair_rows: List[Dict[str, object]] = []

    for model_family in [spec[0] for spec in MODEL_FAMILY_SPECS]:
        family_meta = centroid_metadata[centroid_metadata["model_family"] == model_family].copy()
        run_order = sorted(family_meta["run_id"].unique(), key=lambda x: int(str(x).lstrip("v")))

        for window_index in range(1, EXPECTED_WINDOWS + 1):
            window_meta = family_meta[family_meta["window_index"] == window_index].copy()
            window_meta["run_order"] = window_meta["run_id"].map({run_id: i for i, run_id in enumerate(run_order)})
            window_meta = window_meta.sort_values("run_order")
            centroid_indices = window_meta.index.to_numpy()
            c = centroids[centroid_indices]
            r = int(c.shape[0])
            if r < 2:
                raise ValueError(f"Need at least 2 runs for {model_family} window {window_index}; found {r}")

            cosine_matrix = c @ c.T
            pairwise_similarity = upper_triangle_values(cosine_matrix)
            pairwise_distance = 1.0 - pairwise_similarity
            vendi, vendi_norm = vendi_from_cosine_kernel(c)

            round_start = int(window_meta["round_start"].iloc[0])
            round_end = int(window_meta["round_end"].iloc[0])
            metric_rows.append(
                {
                    "model_family": model_family,
                    "window_index": window_index,
                    "round_start": round_start,
                    "round_end": round_end,
                    "n_runs": r,
                    "n_run_pairs": int(len(pairwise_similarity)),
                    "mean_cross_cosine_similarity": float(np.mean(pairwise_similarity)),
                    "mean_cross_cosine_distance": float(np.mean(pairwise_distance)),
                    "cross_vendi": vendi,
                    "cross_vendi_norm": vendi_norm,
                }
            )

            run_ids = window_meta["run_id"].tolist()
            for run_a_idx, run_b_idx in combinations(range(r), 2):
                sim = float(cosine_matrix[run_a_idx, run_b_idx])
                pair_rows.append(
                    {
                        "model_family": model_family,
                        "run_a": run_ids[run_a_idx],
                        "run_b": run_ids[run_b_idx],
                        "window_index": window_index,
                        "cosine_similarity": sim,
                        "cosine_distance": 1.0 - sim,
                    }
                )

    return pd.DataFrame(metric_rows), pd.DataFrame(pair_rows)


def summarize_family_metrics(metrics: pd.DataFrame, centroid_metadata: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for model_family in [spec[0] for spec in MODEL_FAMILY_SPECS]:
        family_metrics = metrics[metrics["model_family"] == model_family].copy()
        family_centroids = centroid_metadata[centroid_metadata["model_family"] == model_family]
        early = family_metrics[family_metrics["window_index"].between(1, 10)]
        late = family_metrics[family_metrics["window_index"].between(91, 100)]

        early_distance = float(early["mean_cross_cosine_distance"].mean())
        late_distance = float(late["mean_cross_cosine_distance"].mean())
        early_vendi = float(early["cross_vendi_norm"].mean())
        late_vendi = float(late["cross_vendi_norm"].mean())

        rows.append(
            {
                "model_family": model_family,
                "n_runs": int(family_centroids["run_id"].nunique()),
                "n_windows": int(family_metrics["window_index"].nunique()),
                "mean_cross_cosine_similarity_all_windows": float(
                    family_metrics["mean_cross_cosine_similarity"].mean()
                ),
                "sd_cross_cosine_similarity_all_windows": float(
                    family_metrics["mean_cross_cosine_similarity"].std(ddof=1)
                ),
                "mean_cross_cosine_distance_all_windows": float(
                    family_metrics["mean_cross_cosine_distance"].mean()
                ),
                "sd_cross_cosine_distance_all_windows": float(
                    family_metrics["mean_cross_cosine_distance"].std(ddof=1)
                ),
                "mean_cross_vendi_norm_all_windows": float(family_metrics["cross_vendi_norm"].mean()),
                "sd_cross_vendi_norm_all_windows": float(family_metrics["cross_vendi_norm"].std(ddof=1)),
                "early_mean_cross_distance": early_distance,
                "late_mean_cross_distance": late_distance,
                "late_minus_early_cross_distance": late_distance - early_distance,
                "early_mean_cross_vendi_norm": early_vendi,
                "late_mean_cross_vendi_norm": late_vendi,
                "late_minus_early_cross_vendi_norm": late_vendi - early_vendi,
            }
        )

    return pd.DataFrame(rows)


def summarize_pairs(pair_distribution: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for model_family, group in pair_distribution.groupby("model_family", sort=False):
        early = group[group["window_index"].between(1, 10)]
        late = group[group["window_index"].between(91, 100)]
        rows.append(
            {
                "model_family": model_family,
                "n_pair_window_rows": int(len(group)),
                "mean_same_family_run_pair_similarity": float(group["cosine_similarity"].mean()),
                "sd_same_family_run_pair_similarity": float(group["cosine_similarity"].std(ddof=1)),
                "mean_same_family_run_pair_distance": float(group["cosine_distance"].mean()),
                "sd_same_family_run_pair_distance": float(group["cosine_distance"].std(ddof=1)),
                "early_mean_same_family_run_pair_similarity": float(early["cosine_similarity"].mean()),
                "late_mean_same_family_run_pair_similarity": float(late["cosine_similarity"].mean()),
                "early_mean_same_family_run_pair_distance": float(early["cosine_distance"].mean()),
                "late_mean_same_family_run_pair_distance": float(late["cosine_distance"].mean()),
            }
        )
    return pd.DataFrame(rows)


def summarize_reference_comparisons(centroids: np.ndarray, centroid_metadata: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    same_records: List[Dict[str, object]] = []
    different_records: List[Dict[str, object]] = []

    for window_index in range(1, EXPECTED_WINDOWS + 1):
        window_meta = centroid_metadata[centroid_metadata["window_index"] == window_index].copy()

        for model_family, family_meta in window_meta.groupby("model_family", sort=False):
            idx = family_meta.index.to_numpy()
            c = centroids[idx]
            sims = c @ c.T
            run_ids = family_meta["run_id"].tolist()
            for a, b in combinations(range(len(run_ids)), 2):
                sim = float(sims[a, b])
                same_records.append(
                    {
                        "comparison_group": "same_model_same_condition",
                        "window_index": window_index,
                        "cosine_similarity": sim,
                        "cosine_distance": 1.0 - sim,
                    }
                )

        family_names = sorted(window_meta["model_family"].unique().tolist())
        for family_a, family_b in combinations(family_names, 2):
            meta_a = window_meta[window_meta["model_family"] == family_a]
            meta_b = window_meta[window_meta["model_family"] == family_b]
            sims = centroids[meta_a.index.to_numpy()] @ centroids[meta_b.index.to_numpy()].T
            for sim in sims.ravel():
                sim_float = float(sim)
                different_records.append(
                    {
                        "comparison_group": "different_model_same_window",
                        "window_index": window_index,
                        "cosine_similarity": sim_float,
                        "cosine_distance": 1.0 - sim_float,
                    }
                )

    def summarize_records(records: List[Dict[str, object]]) -> Dict[str, object]:
        df = pd.DataFrame(records)
        early = df[df["window_index"].between(1, 10)]
        late = df[df["window_index"].between(91, 100)]
        return {
            "comparison_group": str(df["comparison_group"].iloc[0]),
            "n_centroid_pairs": int(len(df)),
            "mean_cosine_similarity": float(df["cosine_similarity"].mean()),
            "sd_cosine_similarity": float(df["cosine_similarity"].std(ddof=1)),
            "mean_cosine_distance": float(df["cosine_distance"].mean()),
            "sd_cosine_distance": float(df["cosine_distance"].std(ddof=1)),
            "early_mean_cosine_similarity": float(early["cosine_similarity"].mean()),
            "late_mean_cosine_similarity": float(late["cosine_similarity"].mean()),
            "early_mean_cosine_distance": float(early["cosine_distance"].mean()),
            "late_mean_cosine_distance": float(late["cosine_distance"].mean()),
        }

    rows.append(summarize_records(same_records))
    rows.append(summarize_records(different_records))
    return pd.DataFrame(rows)


def write_figures(metrics: pd.DataFrame, family_summary: pd.DataFrame) -> None:
    if not MAKE_FIGURES:
        return

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {
        "DeepSeek": "#2563eb",
        "GPT": "#16a34a",
        "Phi-4": "#dc2626",
    }

    plt.figure(figsize=(9, 5))
    for model_family in [spec[0] for spec in MODEL_FAMILY_SPECS]:
        group = metrics[metrics["model_family"] == model_family]
        plt.plot(
            group["window_index"],
            group["mean_cross_cosine_distance"],
            label=model_family,
            linewidth=1.7,
            color=colors.get(model_family),
        )
    plt.xlabel("Window index")
    plt.ylabel("Mean cross-run cosine distance")
    plt.title("Cross-run semantic distance by window")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(CROSS_RUN_DIR / "cross_run_distance_trajectory.png", dpi=180)
    plt.close()

    plt.figure(figsize=(9, 5))
    for model_family in [spec[0] for spec in MODEL_FAMILY_SPECS]:
        group = metrics[metrics["model_family"] == model_family]
        plt.plot(
            group["window_index"],
            group["cross_vendi_norm"],
            label=model_family,
            linewidth=1.7,
            color=colors.get(model_family),
        )
    plt.xlabel("Window index")
    plt.ylabel("Cross-run normalized Vendi")
    plt.title("Cross-run semantic support by window")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(CROSS_RUN_DIR / "cross_run_vendi_norm_trajectory.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 5))
    ordered = family_summary.set_index("model_family").loc[[spec[0] for spec in MODEL_FAMILY_SPECS]].reset_index()
    bar_colors = [colors.get(name) for name in ordered["model_family"]]
    plt.bar(
        ordered["model_family"],
        ordered["mean_cross_cosine_distance_all_windows"],
        color=bar_colors,
        width=0.62,
    )
    plt.xlabel("Model family")
    plt.ylabel("Mean cross-run cosine distance")
    plt.title("Mean cross-run semantic distance")
    plt.grid(True, axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(CROSS_RUN_DIR / "cross_run_family_summary_bar.png", dpi=180)
    plt.close()


def write_outputs(
    audit_df: pd.DataFrame,
    centroids: np.ndarray,
    centroid_metadata: pd.DataFrame,
    metrics: pd.DataFrame,
    family_summary: pd.DataFrame,
    pair_distribution: pd.DataFrame,
    pair_summary: pd.DataFrame,
    reference_summary: pd.DataFrame,
) -> None:
    CROSS_RUN_DIR.mkdir(parents=True, exist_ok=True)
    audit_df.to_csv(CROSS_RUN_DIR / "cross_run_cache_audit.csv", index=False)
    np.save(CROSS_RUN_DIR / "window_centroids.npy", centroids.astype(np.float32, copy=False))
    centroid_metadata.to_csv(CROSS_RUN_DIR / "window_centroid_metadata.csv", index=False)
    metrics.to_csv(CROSS_RUN_DIR / "cross_run_window_metrics.csv", index=False)
    family_summary.to_csv(CROSS_RUN_DIR / "cross_run_family_summary.csv", index=False)
    pair_distribution.to_csv(CROSS_RUN_DIR / "cross_run_pair_distribution.csv", index=False)
    pair_summary.to_csv(CROSS_RUN_DIR / "cross_run_pair_summary.csv", index=False)
    reference_summary.to_csv(CROSS_RUN_DIR / "cross_run_reference_summary.csv", index=False)
    write_figures(metrics, family_summary)


def print_audit_summary(audit_df: pd.DataFrame) -> None:
    print("[AUDIT] selected 1000-round baseline caches")
    cols = [
        "source_file",
        "model_family",
        "run_id",
        "metadata_rows",
        "embedding_shape",
        "embedding_dtype",
        "row_counts_match",
        "embedding_dim_is_3072",
        "max_parsed_round",
        "expected_number_of_windows",
        "valid",
    ]
    print(audit_df[cols].to_string(index=False))


def print_final_summary(
    specs: Sequence[CacheSpec],
    family_summary: pd.DataFrame,
    reference_summary: pd.DataFrame,
) -> None:
    counts = pd.DataFrame(
        [{"model_family": family, "n_runs": int(sum(spec.model_family == family for spec in specs))} for family, _, _ in MODEL_FAMILY_SPECS]
    )
    print("[SUMMARY] included runs")
    print(counts.to_string(index=False))

    summary_cols = [
        "model_family",
        "mean_cross_cosine_similarity_all_windows",
        "mean_cross_cosine_distance_all_windows",
        "mean_cross_vendi_norm_all_windows",
        "early_mean_cross_distance",
        "late_mean_cross_distance",
        "late_minus_early_cross_distance",
        "early_mean_cross_vendi_norm",
        "late_mean_cross_vendi_norm",
        "late_minus_early_cross_vendi_norm",
    ]
    print("[SUMMARY] cross-run concentration")
    print(family_summary[summary_cols].to_string(index=False))

    print("[SUMMARY] reference comparison")
    print(reference_summary.to_string(index=False))


def main() -> None:
    CROSS_RUN_DIR.mkdir(parents=True, exist_ok=True)

    specs = build_cache_specs()
    audit_df, selected_specs = audit_cache(specs)
    audit_df.to_csv(CROSS_RUN_DIR / "cross_run_cache_audit.csv", index=False)
    print_audit_summary(audit_df)
    require_valid_audit(audit_df)

    centroids, centroid_metadata = compute_window_centroids(selected_specs)
    metrics, pair_distribution = compute_cross_run_metrics(centroids, centroid_metadata)
    family_summary = summarize_family_metrics(metrics, centroid_metadata)
    pair_summary = summarize_pairs(pair_distribution)
    reference_summary = summarize_reference_comparisons(centroids, centroid_metadata)

    write_outputs(
        audit_df,
        centroids,
        centroid_metadata,
        metrics,
        family_summary,
        pair_distribution,
        pair_summary,
        reference_summary,
    )
    print_final_summary(selected_specs, family_summary, reference_summary)

    print(f"[OK] wrote outputs to {CROSS_RUN_DIR}")


if __name__ == "__main__":
    main()
