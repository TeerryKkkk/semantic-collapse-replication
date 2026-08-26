"""Fit the shared longitudinal LDA model and compute topic-diversity summaries."""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS

from config import (
    BOOTSTRAP_DRAWS,
    BOOTSTRAP_SCHEME,
    BOOTSTRAP_SEED,
    EARLY_INTERVALS,
    FAMILY_ORDER,
    LATE_INTERVALS,
    LDA_EVALUATE_EVERY,
    LDA_MAX_DOC_UPDATE_ITER,
    LDA_MAX_ITER,
    LDA_MEAN_CHANGE_TOL,
    LDA_PERP_TOL,
    MAX_DF,
    MIN_DF,
    MODEL_IDENTIFIER_STOPWORDS,
    NGRAM_RANGE,
    PRIMARY_K,
    PRIMARY_SEED,
    ROBUSTNESS_K,
    ROBUSTNESS_SEEDS,
    TOKEN_PATTERN,
)
from parse_data import load_baseline_corpus


LOGGER = logging.getLogger(__name__)
SUMMARY_METRICS = (
    "effective_topics",
    "normalized_entropy",
    "dominant_topic_probability",
    "adjacent_js_distance",
    "document_length",
)


def build_vectorizer(structural_stopwords: set[str]) -> CountVectorizer:
    stopwords = sorted(set(ENGLISH_STOP_WORDS) | set(structural_stopwords))
    return CountVectorizer(
        lowercase=True,
        stop_words=stopwords,
        token_pattern=TOKEN_PATTERN,
        min_df=MIN_DF,
        max_df=MAX_DF,
        ngram_range=NGRAM_RANGE,
        dtype=np.int64,
    )


def fit_lda(X: object, k: int, seed: int) -> tuple[LatentDirichletAllocation, np.ndarray, dict[str, object]]:
    started = time.perf_counter()
    model = LatentDirichletAllocation(
        n_components=k,
        doc_topic_prior=None,
        topic_word_prior=None,
        learning_method="batch",
        max_iter=LDA_MAX_ITER,
        evaluate_every=LDA_EVALUATE_EVERY,
        perp_tol=LDA_PERP_TOL,
        mean_change_tol=LDA_MEAN_CHANGE_TOL,
        max_doc_update_iter=LDA_MAX_DOC_UPDATE_ITER,
        random_state=seed,
        n_jobs=-1,
        verbose=0,
    )
    theta = model.fit_transform(X)
    return model, theta, {
        "k": k,
        "seed": seed,
        "n_iter": int(model.n_iter_),
        "max_iter": LDA_MAX_ITER,
        "converged_before_max_iter": bool(model.n_iter_ < LDA_MAX_ITER),
        "bound": float(model.bound_),
        "perplexity": float(model.perplexity(X)),
        "runtime_seconds": time.perf_counter() - started,
    }


def calculate_interval_metrics(
    theta: np.ndarray,
    documents: pd.DataFrame,
    document_lengths: np.ndarray,
) -> pd.DataFrame:
    """Calculate posterior breadth and adjacent-interval movement explicitly."""
    if theta.shape[0] != len(documents):
        raise ValueError("Theta/document row mismatch")
    k = theta.shape[1]
    safe = np.clip(theta, np.finfo(float).tiny, 1.0)
    entropy = -np.sum(safe * np.log(safe), axis=1)
    metrics = documents.drop(columns=["document_text"]).copy()
    metrics["topic_entropy"] = entropy
    metrics["normalized_entropy"] = entropy / math.log(k)
    metrics["effective_topics"] = np.exp(entropy)
    metrics["dominant_topic"] = np.argmax(theta, axis=1).astype(int)
    metrics["dominant_topic_probability"] = np.max(theta, axis=1)
    metrics["document_length"] = np.asarray(document_lengths).ravel().astype(int)
    metrics["adjacent_js_distance"] = np.nan

    for _, index in metrics.groupby("run_id", sort=False).groups.items():
        positions = np.asarray(list(index), dtype=int)
        order = positions[np.argsort(metrics.loc[positions, "interval"].to_numpy())]
        for previous, current in zip(order[:-1], order[1:]):
            p = theta[previous] / theta[previous].sum()
            q = theta[current] / theta[current].sum()
            midpoint = 0.5 * (p + q)
            divergence = 0.5 * (
                np.sum(p * np.log(p / midpoint)) + np.sum(q * np.log(q / midpoint))
            )
            metrics.loc[current, "adjacent_js_distance"] = math.sqrt(max(0.0, float(divergence)))
    return metrics


def topic_word_table(
    model: LatentDirichletAllocation,
    vectorizer: CountVectorizer,
    top_n: int = 15,
) -> pd.DataFrame:
    vocabulary = np.asarray(vectorizer.get_feature_names_out())
    rows: list[dict[str, object]] = []
    for topic_id, weights in enumerate(model.components_):
        probabilities = weights / weights.sum()
        order = np.argsort(probabilities)[::-1][:top_n]
        for rank, feature_index in enumerate(order, start=1):
            rows.append(
                {
                    "topic_id": topic_id,
                    "rank": rank,
                    "word": vocabulary[feature_index],
                    "weight": float(weights[feature_index]),
                    "probability": float(probabilities[feature_index]),
                }
            )
    return pd.DataFrame(rows)


def _phase_values(frame: pd.DataFrame, phase: str, metric: str) -> pd.Series:
    if phase == "early":
        intervals = set(EARLY_INTERVALS)
        if metric == "adjacent_js_distance":
            intervals.discard(min(EARLY_INTERVALS))
    elif phase == "late":
        intervals = set(LATE_INTERVALS)
        if metric == "adjacent_js_distance":
            intervals.discard(min(LATE_INTERVALS))
    else:
        raise ValueError(phase)
    return frame.loc[frame["interval"].isin(intervals), metric]


def summarize_runs(metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for run_id, group in metrics.groupby("run_id", sort=False):
        group = group.sort_values("interval")
        row: dict[str, object] = {
            "run_id": run_id,
            "model_family": group["model_family"].iloc[0],
            "version": group["version"].iloc[0],
            "version_number": int(group["version_number"].iloc[0]),
        }
        for metric in SUMMARY_METRICS:
            early = float(_phase_values(group, "early", metric).mean())
            late = float(_phase_values(group, "late", metric).mean())
            row[f"early_{metric}"] = early
            row[f"late_{metric}"] = late
            row[f"early_minus_late_{metric}"] = early - late
        rows.append(row)
    result = pd.DataFrame(rows)
    family_rank = {family: index for index, family in enumerate(FAMILY_ORDER)}
    result["family_order"] = result["model_family"].map(family_rank)
    return result.sort_values(["family_order", "version_number"]).drop(columns="family_order")


def summarize_families(run_table: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for family in FAMILY_ORDER:
        group = run_table.loc[run_table["model_family"] == family]
        row: dict[str, object] = {"model_family": family, "n_runs": len(group)}
        for metric in SUMMARY_METRICS:
            early = group[f"early_{metric}"]
            late = group[f"late_{metric}"]
            difference = group[f"early_minus_late_{metric}"]
            row[f"mean_early_{metric}"] = float(early.mean())
            row[f"sd_early_{metric}"] = float(early.std(ddof=1))
            row[f"mean_late_{metric}"] = float(late.mean())
            row[f"sd_late_{metric}"] = float(late.std(ddof=1))
            row[f"mean_early_minus_late_{metric}"] = float(difference.mean())
            row[f"sd_early_minus_late_{metric}"] = float(difference.std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def stratified_run_bootstrap(
    run_table: pd.DataFrame,
    metric: str,
    *,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, float | int | str]:
    """Bootstrap runs within fixed model families and retain equal family weights.

    The five model families are treated as prespecified experimental conditions.
    Each bootstrap replicate resamples the available independent runs with
    replacement inside every family, computes a family mean, and then averages
    the five family means with equal weight.
    """
    rng = np.random.default_rng(seed)
    columns = [f"early_{metric}", f"late_{metric}", f"early_minus_late_{metric}"]
    arrays: dict[str, np.ndarray] = {}
    for family in FAMILY_ORDER:
        values = run_table.loc[run_table["model_family"] == family, columns].to_numpy(float)
        if len(values) == 0:
            raise ValueError(f"No runs available for model family: {family}")
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite bootstrap input for model family: {family}")
        arrays[family] = values

    observed_families = set(run_table["model_family"].dropna().astype(str))
    unexpected = observed_families.difference(FAMILY_ORDER)
    if unexpected:
        raise ValueError(f"Unexpected model families in run table: {sorted(unexpected)}")

    boot = np.empty((draws, 3), dtype=float)
    for draw in range(draws):
        family_values = []
        for family in FAMILY_ORDER:
            values = arrays[family]
            sampled_runs = rng.integers(0, len(values), size=len(values))
            family_values.append(values[sampled_runs].mean(axis=0))
        boot[draw] = np.mean(family_values, axis=0)

    point = np.mean([arrays[family].mean(axis=0) for family in FAMILY_ORDER], axis=0)
    quantiles = np.quantile(boot, [0.025, 0.975], axis=0)
    return {
        "family_equal_early": float(point[0]),
        "family_equal_late": float(point[1]),
        "family_equal_early_minus_late": float(point[2]),
        "early_ci_lower": float(quantiles[0, 0]),
        "early_ci_upper": float(quantiles[1, 0]),
        "late_ci_lower": float(quantiles[0, 1]),
        "late_ci_upper": float(quantiles[1, 1]),
        "difference_ci_lower": float(quantiles[0, 2]),
        "difference_ci_upper": float(quantiles[1, 2]),
        "bootstrap_draws": draws,
        "bootstrap_seed": seed,
        "bootstrap_scheme": BOOTSTRAP_SCHEME,
    }


def overall_summary(run_table: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "metric": metric,
                **stratified_run_bootstrap(run_table, metric, seed=BOOTSTRAP_SEED + metric_index),
            }
            for metric_index, metric in enumerate(SUMMARY_METRICS)
        ]
    )


def _robustness_summary_from_run_rows(run_rows: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (k, seed), group in run_rows.groupby(["k", "seed"], sort=True):
        run_table = group.drop(columns=["k", "seed"]).copy()
        overall = stratified_run_bootstrap(
            run_table,
            "normalized_entropy",
            seed=BOOTSTRAP_SEED + 1000 * int(k) + int(seed),
        )
        fit_columns = [
            "n_iter",
            "max_iter",
            "converged_before_max_iter",
            "bound",
            "perplexity",
            "runtime_seconds",
        ]
        fit_info = {
            column: group[column].iloc[0]
            for column in fit_columns
            if column in group.columns
        }
        rows.append({"k": int(k), "seed": int(seed), **fit_info, **overall})
    return pd.DataFrame(rows).sort_values(["k", "seed"]).reset_index(drop=True)


def run_k_seed_robustness(
    X: object,
    documents: pd.DataFrame,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run or resume the K-by-seed grid and retain run-level summaries.

    Run-level rows are saved so bootstrap confidence intervals can be refreshed
    later without refitting LDA models. Existing aggregate rows produced under a
    different bootstrap scheme are not treated as complete.
    """
    result_path = output_dir / "k_seed_robustness.csv"
    family_path = output_dir / "k_seed_family.csv"
    run_path = output_dir / "k_seed_run.csv"

    if run_path.exists():
        run_rows_frame = pd.read_csv(run_path)
    else:
        run_rows_frame = pd.DataFrame()

    required_run_count = len(FAMILY_ORDER) * 3
    complete: set[tuple[int, int]] = set()
    if not run_rows_frame.empty:
        counts = run_rows_frame.groupby(["k", "seed"]).size()
        complete = {
            (int(k), int(seed))
            for (k, seed), count in counts.items()
            if int(count) == required_run_count
        }

    lengths = np.asarray(X.sum(axis=1)).ravel()
    new_run_rows: list[dict[str, object]] = []
    for k in ROBUSTNESS_K:
        for seed in ROBUSTNESS_SEEDS:
            if (k, seed) in complete:
                continue
            LOGGER.info("Fitting robustness model K=%d, seed=%d", k, seed)
            _, theta, fit_info = fit_lda(X, k, seed)
            run_table = summarize_runs(calculate_interval_metrics(theta, documents, lengths))
            for row in run_table.to_dict("records"):
                new_run_rows.append({"k": k, "seed": seed, **fit_info, **row})

            combined = pd.concat(
                [run_rows_frame, pd.DataFrame(new_run_rows)],
                ignore_index=True,
            )
            combined = combined.drop_duplicates(["k", "seed", "run_id"], keep="last")
            combined.sort_values(["k", "seed", "model_family", "version_number"]).to_csv(
                run_path, index=False
            )

    run_rows_frame = pd.read_csv(run_path)
    result_frame = _robustness_summary_from_run_rows(run_rows_frame)
    result_frame.to_csv(result_path, index=False)

    family_rows: list[dict[str, object]] = []
    for (k, seed), group in run_rows_frame.groupby(["k", "seed"], sort=True):
        family = summarize_families(group.drop(columns=["k", "seed"]))
        for row in family.itertuples(index=False):
            family_rows.append(
                {
                    "k": int(k),
                    "seed": int(seed),
                    "model_family": row.model_family,
                    "n_runs": row.n_runs,
                    "mean_early_normalized_entropy": row.mean_early_normalized_entropy,
                    "mean_late_normalized_entropy": row.mean_late_normalized_entropy,
                    "early_minus_late_normalized_entropy": row.mean_early_minus_late_normalized_entropy,
                }
            )
    family_frame = pd.DataFrame(family_rows).sort_values(["k", "seed", "model_family"])
    family_frame.to_csv(family_path, index=False)
    return result_frame, family_frame, run_rows_frame


def refresh_bootstrap_outputs(output_dir: Path) -> None:
    """Refresh bootstrap summaries from previously saved run-level tables."""
    lda_dir = output_dir / "lda"
    primary_path = lda_dir / "run_early_late.csv"
    if not primary_path.exists():
        raise FileNotFoundError(f"Missing primary run-level table: {primary_path}")
    run_table = pd.read_csv(primary_path)
    overall_summary(run_table).to_csv(lda_dir / "overall_early_late.csv", index=False)

    configuration_path = lda_dir / "configuration.json"
    if configuration_path.exists():
        configuration = json.loads(configuration_path.read_text(encoding="utf-8"))
        configuration["bootstrap_draws"] = BOOTSTRAP_DRAWS
        configuration["bootstrap_seed"] = BOOTSTRAP_SEED
        configuration["bootstrap_scheme"] = BOOTSTRAP_SCHEME
        configuration_path.write_text(json.dumps(configuration, indent=2) + "\n", encoding="utf-8")

    LOGGER.info("Refreshed primary bootstrap summary using %s", BOOTSTRAP_SCHEME)

    robustness_run_path = lda_dir / "k_seed_run.csv"
    if robustness_run_path.exists():
        run_rows = pd.read_csv(robustness_run_path)
        _robustness_summary_from_run_rows(run_rows).to_csv(
            lda_dir / "k_seed_robustness.csv", index=False
        )
        LOGGER.info("Refreshed K-by-seed bootstrap summaries from saved run-level rows")
    else:
        LOGGER.info(
            "No k_seed_run.csv found; K-by-seed confidence intervals require one robustness rerun "
            "to create run-level summaries under the current analysis pipeline"
        )


def run_analysis(raw_dir: Path, output_dir: Path, skip_robustness: bool) -> None:
    lda_dir = output_dir / "lda"
    model_dir = output_dir / "models"
    lda_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Parsing and auditing baseline logs")
    manifest, _, documents, speaker_stopwords = load_baseline_corpus(raw_dir)
    structural_stopwords = set(speaker_stopwords) | set(MODEL_IDENTIFIER_STOPWORDS)
    manifest.to_csv(lda_dir / "data_manifest.csv", index=False)
    documents.to_csv(lda_dir / "interval_documents.csv", index=False)
    pd.DataFrame({"term": sorted(structural_stopwords)}).to_csv(
        lda_dir / "structural_stopwords.csv", index=False
    )

    vectorizer = build_vectorizer(structural_stopwords)
    X = vectorizer.fit_transform(documents["document_text"])
    terms = vectorizer.get_feature_names_out()
    vocabulary = pd.DataFrame(
        {
            "term": terms,
            "feature_index": np.arange(len(terms)),
            "document_frequency": np.asarray((X > 0).sum(axis=0)).ravel(),
            "corpus_count": np.asarray(X.sum(axis=0)).ravel(),
        }
    )
    vocabulary.to_csv(lda_dir / "vocabulary.csv", index=False)

    LOGGER.info("Fitting shared LDA model K=%d, seed=%d", PRIMARY_K, PRIMARY_SEED)
    model, theta, fit_info = fit_lda(X, PRIMARY_K, PRIMARY_SEED)
    joblib.dump(vectorizer, model_dir / "count_vectorizer.joblib")
    joblib.dump(model, model_dir / "lda_K20_seed42.joblib")
    (model_dir / "lda_K20_seed42_fit_info.json").write_text(
        json.dumps(fit_info, indent=2) + "\n", encoding="utf-8"
    )

    topic_word_table(model, vectorizer).to_csv(lda_dir / "topics.csv", index=False)
    distributions = documents.drop(columns=["document_text"]).copy()
    for topic_id in range(PRIMARY_K):
        distributions[f"topic_{topic_id:02d}"] = theta[:, topic_id]
    distributions["topic_probability_sum"] = theta.sum(axis=1)
    distributions.to_csv(lda_dir / "interval_topic_distributions.csv", index=False)

    metrics = calculate_interval_metrics(theta, documents, np.asarray(X.sum(axis=1)).ravel())
    run_table = summarize_runs(metrics)
    metrics.to_csv(lda_dir / "interval_topic_metrics.csv", index=False)
    run_table.to_csv(lda_dir / "run_early_late.csv", index=False)
    summarize_families(run_table).to_csv(lda_dir / "family_early_late.csv", index=False)
    overall_summary(run_table).to_csv(lda_dir / "overall_early_late.csv", index=False)

    configuration = {
        "document_unit": "non-overlapping 10-round interval",
        "primary_k": PRIMARY_K,
        "primary_seed": PRIMARY_SEED,
        "robustness_k": list(ROBUSTNESS_K),
        "robustness_seeds": list(ROBUSTNESS_SEEDS),
        "early_intervals": list(EARLY_INTERVALS),
        "late_intervals": list(LATE_INTERVALS),
        "vectorizer": {
            "token_pattern": TOKEN_PATTERN,
            "min_df": MIN_DF,
            "max_df": MAX_DF,
            "ngram_range": list(NGRAM_RANGE),
            "weighting": "integer counts",
        },
        "lda": {
            "learning_method": "batch",
            "max_iter": LDA_MAX_ITER,
            "evaluate_every": LDA_EVALUATE_EVERY,
            "perp_tol": LDA_PERP_TOL,
            "mean_change_tol": LDA_MEAN_CHANGE_TOL,
            "max_doc_update_iter": LDA_MAX_DOC_UPDATE_ITER,
        },
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_scheme": BOOTSTRAP_SCHEME,
    }
    (lda_dir / "configuration.json").write_text(
        json.dumps(configuration, indent=2) + "\n", encoding="utf-8"
    )
    if not skip_robustness:
        run_k_seed_robustness(X, documents, lda_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, help="Directory containing simulation .txt logs")
    parser.add_argument("--output-dir", type=Path, required=True, help="Destination for generated analysis files")
    parser.add_argument(
        "--skip-robustness",
        action="store_true",
        help="Fit only the primary model and omit the K-by-seed grid",
    )
    parser.add_argument(
        "--refresh-bootstrap-only",
        action="store_true",
        help="Recalculate bootstrap summaries from existing run-level output tables without refitting the primary LDA model",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    arguments = parse_args()
    if arguments.refresh_bootstrap_only:
        refresh_bootstrap_outputs(arguments.output_dir)
        return
    if arguments.raw_dir is None:
        raise SystemExit("--raw-dir is required unless --refresh-bootstrap-only is used")
    run_analysis(arguments.raw_dir, arguments.output_dir, arguments.skip_robustness)


if __name__ == "__main__":
    main()
