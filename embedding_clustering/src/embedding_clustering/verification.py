"""Lightweight verification of the embedding-clustering results."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score

from .characterize import assignment_label_hash, association_summary
from .config import K_VALUES, MODEL_SPECS, SEEDS
from .io import load_inputs, write_json
from .select_k import choose_k_from_summary
from .spherical_kmeans import relabel_by_size, spherical_kmeans


def _close(actual: float, expected: float, tolerance: float) -> bool:
    return bool(np.isclose(actual, expected, rtol=0.0, atol=tolerance))


def verify_release(
    embeddings_path: Path,
    manifest_path: Path,
    results_dir: Path,
    *,
    refit_selected: bool = False,
    tolerance: float = 1e-12,
) -> dict[str, Any]:
    """Check fixed inputs, summaries, assignments, AMI values, and selected fits."""

    matrix, manifest, input_validation = load_inputs(
        embeddings_path, manifest_path, check_hashes=True
    )
    results_dir = results_dir.resolve()
    assignments = pd.read_csv(results_dir / "primary_cluster_assignments.csv")
    family_summary = pd.read_csv(results_dir / "family_summary.csv")
    k_summary = pd.read_csv(results_dir / "k_selection_summary.csv")
    association_table = pd.read_csv(results_dir / "cluster_association_summary.csv")

    checks: dict[str, bool] = {
        "input_has_1500_embeddings": matrix.shape == (1500, 3072),
        "input_has_five_families": manifest["model_family"].nunique() == 5,
        "input_has_v1_v2_v3": set(manifest["run_version"]) == {"V1", "V2", "V3"},
        "assignments_have_1500_rows": len(assignments) == 1500,
        "assignments_have_unique_intervals": not assignments["interval_id"].duplicated().any(),
        "summary_has_five_families": len(family_summary) == 5,
        "k_summary_has_45_rows": len(k_summary) == len(MODEL_SPECS) * len(K_VALUES),
    }
    manifest_key = [
        "final_row_index_0_based",
        "interval_id",
        "model_family",
        "run_version",
        "interval_number",
        "start_round",
        "end_round",
    ]
    checks["assignment_rows_match_manifest"] = bool(
        assignments[manifest_key].reset_index(drop=True).equals(
            manifest[manifest_key].reset_index(drop=True)
        )
    )

    family_results: dict[str, Any] = {}
    for spec in MODEL_SPECS:
        family_assignments = assignments.loc[
            assignments["model_family"].eq(spec.name)
        ].copy()
        family_row = family_summary.loc[
            family_summary["model_family"].eq(spec.name)
        ].iloc[0]
        family_k = k_summary.loc[k_summary["model_family"].eq(spec.name)].copy()
        selected_k, stability_relaxed, _ = choose_k_from_summary(family_k)
        selected_row = family_k.loc[family_k["k"].eq(selected_k)].iloc[0]
        selected_seed = int(family_row["selected_seed"])
        associations = association_summary(family_assignments)
        label_hash = assignment_label_hash(family_assignments)

        run_saved = float(
            association_table.loc[
                association_table["model_family"].eq(spec.name)
                & association_table["variable"].eq("run_identity"),
                "value",
            ].iloc[0]
        )
        phase_saved = float(
            association_table.loc[
                association_table["model_family"].eq(spec.name)
                & association_table["variable"].eq(
                    "interaction_phase_1_33_34_66_67_100"
                ),
                "value",
            ].iloc[0]
        )

        family_checks = {
            "selected_k": selected_k == spec.selected_k,
            "selected_seed": selected_seed == spec.selected_seed,
            "300_assignments": len(family_assignments) == 300,
            "30_initializations_per_k": bool(
                family_k["n_initializations"].eq(len(SEEDS)).all()
            ),
            "435_ari_pairs_per_k": bool(
                family_k["n_pairwise_ari"]
                .eq(len(SEEDS) * (len(SEEDS) - 1) // 2)
                .all()
            ),
            "one_selected_k_flag": int(family_k["selected_primary_k"].sum()) == 1,
            "mean_silhouette": _close(
                float(selected_row["mean_cosine_silhouette"]),
                spec.mean_silhouette,
                tolerance,
            ),
            "solution_silhouette": _close(
                float(selected_row["best_solution_silhouette"]),
                spec.solution_silhouette,
                tolerance,
            ),
            "stability_median_ari": _close(
                float(selected_row["median_pairwise_ari"]),
                spec.stability_median_ari,
                tolerance,
            ),
            "run_ami": _close(associations["run_ami"], spec.run_ami, tolerance),
            "phase_ami": _close(
                associations["phase_ami"], spec.phase_ami, tolerance
            ),
            "saved_run_ami": _close(run_saved, spec.run_ami, tolerance),
            "saved_phase_ami": _close(phase_saved, spec.phase_ami, tolerance),
            "assignment_label_hash": label_hash == spec.label_sha256_int32,
        }
        refit_details: dict[str, Any] | None = None
        if refit_selected:
            mask = manifest["model_family"].eq(spec.name).to_numpy()
            family_matrix = np.ascontiguousarray(matrix[mask], dtype=np.float32)
            refit = spherical_kmeans(family_matrix, selected_k, selected_seed)
            refit_labels, _, _ = relabel_by_size(refit.labels, refit.centers)
            saved_labels = (
                family_assignments.sort_values("family_row_index_0_based")["cluster"]
                .to_numpy(np.int32)
                - 1
            )
            distances = np.maximum(1.0 - family_matrix @ family_matrix.T, 0.0)
            np.fill_diagonal(distances, 0.0)
            refit_silhouette = float(
                silhouette_score(
                    distances.astype(np.float64), refit.labels, metric="precomputed"
                )
            )
            family_checks["selected_refit_exact_labels"] = bool(
                np.array_equal(refit_labels, saved_labels)
            )
            family_checks["selected_refit_silhouette"] = _close(
                refit_silhouette, spec.solution_silhouette, tolerance
            )
            refit_details = {
                "cosine_loss": refit.cosine_loss,
                "cosine_silhouette": refit_silhouette,
                "iterations": refit.iterations,
                "converged": refit.converged,
            }

        family_results[spec.name] = {
            "checks": family_checks,
            "selected_k": selected_k,
            "selected_seed": selected_seed,
            "mean_silhouette": float(selected_row["mean_cosine_silhouette"]),
            "solution_silhouette": float(
                selected_row["best_solution_silhouette"]
            ),
            "stability_median_ari": float(selected_row["median_pairwise_ari"]),
            "stability_screen_relaxed": stability_relaxed,
            "run_ami": associations["run_ami"],
            "phase_ami": associations["phase_ami"],
            "label_sha256_int32": label_hash,
            "selected_refit": refit_details,
        }
        checks[f"{spec.slug}_all_checks"] = all(family_checks.values())

    return {
        "passed": all(checks.values()),
        "refit_selected": refit_selected,
        "tolerance": tolerance,
        "input_validation": input_validation,
        "checks": checks,
        "families": family_results,
        "verification_scope": "saved summaries, saved assignments, and selected fits",
    }


def verify_and_write(
    embeddings_path: Path,
    manifest_path: Path,
    results_dir: Path,
    report_path: Path,
    *,
    refit_selected: bool = False,
    tolerance: float = 1e-12,
) -> dict[str, Any]:
    report = verify_release(
        embeddings_path,
        manifest_path,
        results_dir,
        refit_selected=refit_selected,
        tolerance=tolerance,
    )
    write_json(report_path.resolve(), report)
    if not report["passed"]:
        failed = [name for name, passed in report["checks"].items() if not passed]
        raise RuntimeError(f"Release verification failed: {failed}")
    return report
