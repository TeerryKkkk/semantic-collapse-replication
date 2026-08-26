from __future__ import annotations

"""Run the five-class independent-reference classifier."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression

from .embed_texts import EMBEDDING_DIMENSION, EMBEDDING_MODEL, EmbeddingCache, ensure_embeddings
from .pipeline import (
    CLASS_ORDER,
    FAMILY_ORDER,
    IndependentReferenceData,
    build_independent_reference,
    collect_independent_inputs,
    load_independent_manifest,
    parse_independent_runs,
)


SEGMENTS = {"overall": (1, 1000), "early": (1, 200), "middle": (401, 600), "late": (801, 1000)}
PROBABILITY_COLUMNS = {
    "Claude Sonnet": "p_claude_sonnet",
    "DeepSeek-V3": "p_deepseek_v3",
    "GPT-4o-mini": "p_gpt_4o_mini",
    "GPT-5.6 Terra": "p_gpt_5_6_terra",
    "Phi-4": "p_phi_4",
}


def make_classifier() -> LogisticRegression:
    return LogisticRegression(
        penalty="l2",
        C=4.0,
        solver="lbfgs",
        max_iter=5000,
        class_weight=None,
        fit_intercept=True,
        tol=1e-4,
        random_state=None,
    )


def _matrix(hashes: list[str], vectors: dict[str, bytes]) -> np.ndarray:
    matrix = np.empty((len(hashes), EMBEDDING_DIMENSION), dtype=np.float32)
    for index, digest in enumerate(hashes):
        vector = np.frombuffer(vectors[digest], dtype="<f4")
        if vector.shape != (EMBEDDING_DIMENSION,):
            raise RuntimeError(f"Invalid embedding dimension for {digest}")
        matrix[index] = vector
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    matrix /= norms
    if not np.isfinite(matrix).all():
        raise RuntimeError("Non-finite normalized embeddings")
    return matrix


def _metrics(rows: list[dict]) -> dict[str, float | int]:
    confusion = np.zeros((len(CLASS_ORDER), len(CLASS_ORDER)), dtype=np.int64)
    p_sum = 0.0
    for row in rows:
        true_index = CLASS_ORDER.index(row["true_class"])
        predicted_index = CLASS_ORDER.index(row["predicted_class"])
        confusion[true_index, predicted_index] += 1
        p_sum += float(row["p_true_model"])
    n = len(rows)
    if not n:
        return {"round_documents": 0, "accuracy": float("nan"), "balanced_accuracy": float("nan"), "macro_f1": float("nan"), "mean_p_true_model": float("nan")}
    true_support = confusion.sum(axis=1).astype(float)
    predicted_support = confusion.sum(axis=0).astype(float)
    true_positive = np.diag(confusion).astype(float)
    recalls = np.divide(true_positive, true_support, out=np.zeros_like(true_positive), where=true_support > 0)
    f1 = np.divide(2 * true_positive, true_support + predicted_support, out=np.zeros_like(true_positive), where=(true_support + predicted_support) > 0)
    return {
        "round_documents": n,
        "accuracy": float(true_positive.sum() / n),
        "balanced_accuracy": float(recalls[true_support > 0].mean()),
        "macro_f1": float(f1[true_support > 0].mean()),
        "mean_p_true_model": float(p_sum / n),
    }


def _segment(rows: list[dict], name: str) -> list[dict]:
    low, high = SEGMENTS[name]
    return [row for row in rows if low <= int(row["round"]) <= high]


def _fit_and_predict(data: IndependentReferenceData, cache: EmbeddingCache, output_dir: Path, save_model: bool) -> tuple[list[dict], dict]:
    train_hashes = [record.sha256 for record in data.training]
    train_vectors = cache.vectors(train_hashes)
    x_train = _matrix(train_hashes, train_vectors)
    y_train = np.asarray([record.model_family for record in data.training])
    classifier = make_classifier()
    classifier.fit(x_train, y_train, sample_weight=data.sample_weights)
    if list(classifier.classes_) != CLASS_ORDER:
        raise RuntimeError(f"Unexpected class order: {list(classifier.classes_)}")
    fit_diagnostics = {
        "training_utterances": len(data.training),
        "unique_training_texts": len(set(train_hashes)),
        "reference_runs": len(data.reference_run_counts),
        "mean_sample_weight": float(data.sample_weights.mean()),
        "n_iter": int(np.max(classifier.n_iter_)),
        "converged_before_max_iter": bool(np.max(classifier.n_iter_) < 5000),
        "class_order": CLASS_ORDER,
    }
    if save_model:
        model_dir = output_dir / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "classifier": classifier,
                "class_order": CLASS_ORDER,
                "embedding_model": EMBEDDING_MODEL,
                "embedding_dimension": EMBEDDING_DIMENSION,
                "normalization": "row-wise L2",
                "sample_weight": "equal total contribution per each of 15 reference runs; mean one",
                "design": "independent_reference",
            },
            model_dir / "independent_reference_classifier.joblib",
            compress=3,
        )
    del x_train, train_vectors

    test_hashes = [record.sha256 for record in data.test]
    test_vectors = cache.vectors(test_hashes)
    x_test = _matrix(test_hashes, test_vectors)
    probabilities = classifier.predict_proba(x_test)
    predicted = probabilities.argmax(axis=1)
    rows: list[dict] = []
    for index, record in enumerate(data.test):
        true_index = CLASS_ORDER.index(record.model_family)
        predicted_index = int(predicted[index])
        row = {
            "model_family": record.model_family,
            "run_id": f"run_{record.run_id}",
            "round": record.round_id,
            "true_class": record.model_family,
            "predicted_class": CLASS_ORDER[predicted_index],
            "p_true_model": float(probabilities[index, true_index]),
            "correct": int(predicted_index == true_index),
            "valid_utterance_count": record.valid_utterance_count,
            "document_character_count": record.character_count,
            "document_whitespace_word_count": record.whitespace_word_count,
            "input_sha256": record.sha256,
        }
        for class_index, label in enumerate(CLASS_ORDER):
            row[PROBABILITY_COLUMNS[label]] = float(probabilities[index, class_index])
        rows.append(row)
    rows.sort(key=lambda row: (FAMILY_ORDER.index(row["model_family"]), row["run_id"], row["round"]))
    return rows, fit_diagnostics


def run(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    specs = load_independent_manifest(args.manifest)
    reference_runs, test_runs = parse_independent_runs(specs)
    data = build_independent_reference(reference_runs, test_runs)
    unique_inputs = collect_independent_inputs(data)
    preparation = {
        "design": "independent_reference",
        "model_families": FAMILY_ORDER,
        "class_order": CLASS_ORDER,
        "reference_runs": 15,
        "test_runs": 15,
        "training_utterances": len(data.training),
        "test_round_documents": len(data.test),
        "unique_embedding_inputs": len(unique_inputs),
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dimension": EMBEDDING_DIMENSION,
        "character_truncation": None,
        "mean_sample_weight": float(data.sample_weights.mean()),
    }
    (args.output_dir / "preparation_summary.json").write_text(json.dumps(preparation, indent=2) + "\n", encoding="utf-8")
    if args.prepare_only:
        return
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY must be set in the process environment")
    cache_path = args.cache_path or (args.output_dir / "embedding_cache.sqlite3")
    with EmbeddingCache(cache_path) as cache:
        ensure_embeddings(cache, unique_inputs, api_key, batch_size=args.batch_size)
        rows, fit_diagnostics = _fit_and_predict(data, cache, args.output_dir, not args.no_save_model)
    fieldnames = [
        "model_family", "run_id", "round", "true_class", "predicted_class", "p_true_model", "correct",
        *[PROBABILITY_COLUMNS[label] for label in CLASS_ORDER],
        "valid_utterance_count", "document_character_count", "document_whitespace_word_count", "input_sha256",
    ]
    with (args.output_dir / "raw_predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "design": "independent_reference",
        "overall": _metrics(rows),
        "segments": {name: _metrics(_segment(rows, name)) for name in ("early", "middle", "late")},
        "fit": fit_diagnostics,
        "classifier": {"solver": "lbfgs", "regularization": "L2", "C": 4.0, "max_iter": 5000, "tol": 1e-4, "fit_intercept": True, "class_weight": None, "calibration": None},
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dimension": EMBEDDING_DIMENSION,
        "scikit_learn": sklearn.__version__,
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the primary five-class independent-reference model-attribution classifier.")
    parser.add_argument("--manifest", type=Path, required=True, help="CSV with role, model_family, run_id, and transcript_path columns.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for generated cache, model, predictions, and summaries.")
    parser.add_argument("--cache-path", type=Path, help="Optional exact-text resumable SQLite embedding cache path.")
    parser.add_argument("--batch-size", type=int, default=64, help="Embedding request batch size (default: 64).")
    parser.add_argument("--prepare-only", action="store_true", help="Validate manifests/parsing/weights without API calls or model fitting.")
    parser.add_argument("--no-save-model", action="store_true", help="Do not save the fitted classifier artifact.")
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be positive")
    run(args)


if __name__ == "__main__":
    main()
