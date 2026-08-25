from __future__ import annotations

"""Run the optional frozen five-class strict leave-one-run-out robustness analysis."""

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
from .pipeline import CLASS_ORDER, FAMILY_ORDER, FOLDS, FoldData, build_fold, collect_unique_inputs, load_run_manifest, parse_runs


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


def _manifest_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_preparation_summary(output_dir: Path, runs, unique_inputs: dict[str, str]) -> None:
    folds = [build_fold(runs, fold) for fold in sorted(FOLDS)]
    summary = {
        "model_families": FAMILY_ORDER,
        "class_order": CLASS_ORDER,
        "run_count": len(runs),
        "unique_embedding_inputs": len(unique_inputs),
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dimension": EMBEDDING_DIMENSION,
        "character_truncation": None,
        "folds": [
            {
                "fold": data.fold,
                "held_out_run_id": data.held_out_run_id,
                "training_utterances": len(data.training),
                "test_round_documents": len(data.test),
                "mean_sample_weight": float(data.sample_weights.mean()),
            }
            for data in folds
        ],
    }
    (output_dir / "preparation_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def _fit_fold(data: FoldData, cache: EmbeddingCache, output_dir: Path, save_models: bool) -> list[dict]:
    train_hashes = [record.sha256 for record in data.training]
    test_hashes = [record.sha256 for record in data.test]
    vectors = cache.vectors(train_hashes + test_hashes)
    x_train = _matrix(train_hashes, vectors)
    x_test = _matrix(test_hashes, vectors)
    y_train = np.asarray([record.model_family for record in data.training])
    classifier = make_classifier()
    classifier.fit(x_train, y_train, sample_weight=data.sample_weights)
    if list(classifier.classes_) != CLASS_ORDER:
        raise RuntimeError(f"Unexpected class order: {list(classifier.classes_)}")
    probabilities = classifier.predict_proba(x_test)
    predictions = probabilities.argmax(axis=1)
    rows: list[dict] = []
    for index, record in enumerate(data.test):
        true_index = CLASS_ORDER.index(record.model_family)
        predicted_index = int(predictions[index])
        row = {
            "model_family": record.model_family,
            "run_id": f"run_{record.run_id}",
            "fold": data.fold,
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
    if save_models:
        model_dir = output_dir / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "classifier": classifier,
                "fold": data.fold,
                "held_out_run_id": data.held_out_run_id,
                "class_order": CLASS_ORDER,
                "embedding_model": EMBEDDING_MODEL,
                "embedding_dimension": EMBEDDING_DIMENSION,
                "normalization": "row-wise L2",
                "sample_weight": "equal total contribution per training run; fold mean one",
            },
            model_dir / f"fold_{data.fold}_classifier.joblib",
            compress=3,
        )
    return rows


def run(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    specs = load_run_manifest(args.manifest)
    runs = parse_runs(specs)
    unique_inputs = collect_unique_inputs(runs)
    _write_preparation_summary(args.output_dir, runs, unique_inputs)
    if args.prepare_only:
        return
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY must be set in the environment")
    cache_path = args.cache_path or (args.output_dir / "embedding_cache.sqlite3")
    all_rows: list[dict] = []
    with EmbeddingCache(cache_path) as cache:
        ensure_embeddings(cache, unique_inputs, api_key, batch_size=args.batch_size)
        for fold in sorted(FOLDS):
            all_rows.extend(_fit_fold(build_fold(runs, fold), cache, args.output_dir, not args.no_save_models))
    all_rows.sort(key=lambda row: (row["fold"], FAMILY_ORDER.index(row["model_family"]), row["round"]))
    fieldnames = [
        "model_family", "run_id", "fold", "round", "true_class", "predicted_class",
        "p_true_model", "correct", *[PROBABILITY_COLUMNS[label] for label in CLASS_ORDER],
        "valid_utterance_count", "document_character_count", "document_whitespace_word_count", "input_sha256",
    ]
    with (args.output_dir / "raw_fold_predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(all_rows)
    summary = {
        "held_out_round_documents": len(all_rows),
        "overall_accuracy": float(np.mean([row["correct"] for row in all_rows])),
        "mean_p_true_model": float(np.mean([row["p_true_model"] for row in all_rows])),
        "folds": FOLDS,
        "class_order": CLASS_ORDER,
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dimension": EMBEDDING_DIMENSION,
        "classifier": {
            "solver": "lbfgs",
            "regularization": "L2",
            "C": 4.0,
            "max_iter": 5000,
            "tol": 1e-4,
            "fit_intercept": True,
            "class_weight": None,
            "calibration": None,
        },
        "scikit_learn": sklearn.__version__,
        "manifest_sha256": _manifest_sha256(args.manifest),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the optional canonical five-class strict leave-one-run-out robustness classifier.")
    parser.add_argument("--manifest", type=Path, required=True, help="CSV with model_family, run_id, and transcript_path columns.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for generated cache, models, predictions, and summaries.")
    parser.add_argument("--cache-path", type=Path, help="Optional resumable SQLite embedding cache path.")
    parser.add_argument("--batch-size", type=int, default=64, help="Embedding request batch size (default: 64).")
    parser.add_argument("--prepare-only", action="store_true", help="Validate parsing/folds and write preparation_summary.json without API calls.")
    parser.add_argument("--no-save-models", action="store_true", help="Do not save fitted fold classifiers.")
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be positive")
    run(args)


if __name__ == "__main__":
    main()
