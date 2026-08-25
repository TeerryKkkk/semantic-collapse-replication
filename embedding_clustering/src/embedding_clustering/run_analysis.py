"""End-to-end within-model embedding-clustering workflow."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from .characterize import association_summary
from .config import (
    K_VALUES,
    MINIMUM_CLUSTER_SIZE,
    MODEL_SPECS,
    PHASE_BREAKS,
    SEEDS,
    SILHOUETTE_TIE_TOLERANCE,
    STABILITY_REFERENCE_ARI,
)
from .io import load_inputs, write_json
from .select_k import FamilySelection, select_k_for_family


LOGGER = logging.getLogger(__name__)


ASSIGNMENT_COLUMNS = (
    "final_row_index_0_based",
    "interval_id",
    "model_family",
    "run_version",
    "interval_number",
    "start_round",
    "end_round",
)


def _save_family_outputs(
    output_dir: Path,
    family_manifest: pd.DataFrame,
    selection: FamilySelection,
) -> tuple[pd.DataFrame, dict[str, object], dict[str, float]]:
    family_dir = output_dir / next(
        spec.slug for spec in MODEL_SPECS if spec.name == selection.model_family
    )
    solution_dir = family_dir / "k_solutions"
    solution_dir.mkdir(parents=True, exist_ok=True)

    selection.metrics.to_csv(family_dir / "k_selection_metrics.csv", index=False)
    selection.stability.to_csv(family_dir / "clustering_stability.csv", index=False)
    selection.summary.to_csv(family_dir / "k_selection_summary.csv", index=False)
    global_indices = family_manifest["final_row_index_0_based"].to_numpy(np.int32)
    for k in K_VALUES:
        best = selection.best_by_k[k]
        seed = selection.best_seed_by_k[k]
        metric = selection.metrics.loc[
            selection.metrics["k"].eq(k) & selection.metrics["seed"].eq(seed)
        ].iloc[0]
        np.savez_compressed(
            solution_dir / f"k_{k:02d}_best_geometry_only.npz",
            labels=best.labels.astype(np.int32),
            centroids=best.centers.astype(np.float32),
            family_row_indices=np.arange(300, dtype=np.int32),
            global_row_indices=global_indices,
            k=np.int32(k),
            seed=np.int32(seed),
            cosine_loss=np.float64(best.cosine_loss),
            cosine_silhouette=np.float64(metric["cosine_silhouette"]),
        )

    keep = [column for column in ASSIGNMENT_COLUMNS if column in family_manifest.columns]
    assignments = family_manifest[keep].copy()
    assignments["family_row_index_0_based"] = np.arange(300)
    assignments["cluster"] = selection.labels + 1
    assignments["cosine_similarity_to_assigned_centroid"] = (
        selection.assigned_similarity
    )
    assignments["cosine_distance_to_assigned_centroid"] = (
        1.0 - selection.assigned_similarity
    )
    assignments.to_csv(family_dir / "primary_cluster_assignments.csv", index=False)
    np.save(family_dir / "cluster_centroids.npy", selection.centers.astype(np.float32))
    np.savez_compressed(
        family_dir / "cluster_centroids.npz",
        centroids=selection.centers.astype(np.float32),
        cluster_labels=np.arange(1, selection.selected_k + 1, dtype=np.int32),
        selected_k=np.int32(selection.selected_k),
        selected_seed=np.int32(selection.selected_seed),
    )

    selected_summary = selection.summary.loc[
        selection.summary["k"].eq(selection.selected_k)
    ].iloc[0]
    decision: dict[str, object] = {
        "model_family": selection.model_family,
        "selected_k": selection.selected_k,
        "selected_seed": selection.selected_seed,
        "selected_mean_cosine_silhouette": float(
            selected_summary["mean_cosine_silhouette"]
        ),
        "selected_best_solution_silhouette": float(
            selected_summary["best_solution_silhouette"]
        ),
        "selected_median_pairwise_ari": float(
            selected_summary["median_pairwise_ari"]
        ),
        "selected_minimum_cluster_size": int(
            selected_summary["best_solution_minimum_cluster_size"]
        ),
        "selected_maximum_cluster_size": int(
            selected_summary["best_solution_maximum_cluster_size"]
        ),
        "geometry_only_k_selection": True,
        "seed_selection_rule": "minimum spherical cosine loss among seeds 0-29",
        "k_selection_rule": {
            "candidate_k": list(K_VALUES),
            "mean_silhouette_tie_band": SILHOUETTE_TIE_TOLERANCE,
            "minimum_cluster_size": MINIMUM_CLUSTER_SIZE,
            "stability_reference_median_ari": STABILITY_REFERENCE_ARI,
            "parsimony": (
                "smallest K after nonpathology, silhouette tie, and stability screens"
            ),
            "stability_screen_relaxed_because_no_tied_k_met_reference": (
                selection.stability_screen_relaxed
            ),
        },
        "fit_space": "original 3072-dimensional L2-normalized embedding space",
        "used_text_or_metadata_for_fit_or_selection": False,
        "label_mapping_after_selection": {
            str(old): int(new + 1) for old, new in selection.label_mapping.items()
        },
    }
    write_json(family_dir / "k_selection_decision.json", decision)

    associations = association_summary(assignments)
    pd.DataFrame(
        [
            {
                "model_family": selection.model_family,
                "variable": "run_identity",
                "measure": "adjusted_mutual_information",
                "value": associations["run_ami"],
            },
            {
                "model_family": selection.model_family,
                "variable": "interaction_phase_1_33_34_66_67_100",
                "measure": "adjusted_mutual_information",
                "value": associations["phase_ami"],
            },
        ]
    ).to_csv(family_dir / "cluster_association_summary.csv", index=False)
    return assignments, decision, associations


def run_analysis(
    embeddings_path: Path,
    manifest_path: Path,
    output_dir: Path,
    *,
    check_hashes: bool = True,
) -> pd.DataFrame:
    """Run the exact family-specific clustering workflow and save its outputs."""

    matrix, manifest, input_validation = load_inputs(
        embeddings_path, manifest_path, check_hashes=check_hashes
    )
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "input_validation.json", input_validation)
    write_json(
        output_dir / "analysis_config.json",
        {
            "candidate_k": list(K_VALUES),
            "seeds": list(SEEDS),
            "silhouette_tie_tolerance": SILHOUETTE_TIE_TOLERANCE,
            "minimum_cluster_size": MINIMUM_CLUSTER_SIZE,
            "stability_reference_median_ari": STABILITY_REFERENCE_ARI,
            "phase_breaks": list(PHASE_BREAKS),
            "model_order": [spec.name for spec in MODEL_SPECS],
            "clustering_uses_embeddings_only": True,
        },
    )

    all_assignments: list[pd.DataFrame] = []
    decisions: list[dict[str, object]] = []
    family_summaries: list[dict[str, object]] = []
    k_summaries: list[pd.DataFrame] = []
    for spec in MODEL_SPECS:
        mask = manifest["model_family"].eq(spec.name).to_numpy()
        family_manifest = manifest.loc[mask].reset_index(drop=True)
        family_matrix = np.ascontiguousarray(matrix[mask], dtype=np.float32)
        LOGGER.info("Fitting %s", spec.name)
        selection = select_k_for_family(
            spec.name, family_matrix, progress=LOGGER.info
        )
        assignments, decision, associations = _save_family_outputs(
            output_dir, family_manifest, selection
        )
        all_assignments.append(assignments)
        decisions.append(decision)
        k_summary = selection.summary.copy()
        k_summary["n_initializations"] = len(SEEDS)
        k_summary["n_pairwise_ari"] = len(SEEDS) * (len(SEEDS) - 1) // 2
        k_summaries.append(k_summary)
        family_summaries.append(
            {
                "model_family": spec.name,
                "selected_k": selection.selected_k,
                "selected_seed": selection.selected_seed,
                "mean_silhouette": decision["selected_mean_cosine_silhouette"],
                "best_solution_silhouette": decision[
                    "selected_best_solution_silhouette"
                ],
                "stability_median_ari": decision["selected_median_pairwise_ari"],
                "run_ami": associations["run_ami"],
                "phase_ami": associations["phase_ami"],
            }
        )

    combined = pd.concat(all_assignments, ignore_index=True)
    combined.to_csv(output_dir / "all_primary_cluster_assignments.csv", index=False)
    summary = pd.DataFrame(family_summaries)
    summary.to_csv(output_dir / "family_summary.csv", index=False)
    pd.concat(k_summaries, ignore_index=True).to_csv(
        output_dir / "k_selection_summary.csv", index=False
    )
    pd.DataFrame(
        [
            {
                "model_family": row["model_family"],
                "variable": variable,
                "measure": "adjusted_mutual_information",
                "value": row[value_key],
            }
            for row in family_summaries
            for variable, value_key in (
                ("run_identity", "run_ami"),
                ("interaction_phase_1_33_34_66_67_100", "phase_ami"),
            )
        ]
    ).to_csv(output_dir / "cluster_association_summary.csv", index=False)
    pd.DataFrame(decisions).to_csv(
        output_dir / "family_k_selection_summary.csv", index=False
    )
    write_json(output_dir / "family_k_selection_decisions.json", {"families": decisions})

    write_json(
        output_dir / "run_summary.json",
        {
            "families": family_summaries,
            "fit_space": "original 3072-dimensional L2-normalized embedding space",
        },
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run family-specific spherical clustering of fixed embeddings."
    )
    parser.add_argument(
        "--embeddings",
        type=Path,
        default=Path("data/embeddings_l2.npy"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/embedding_manifest.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/embedding_clustering/public"),
    )
    parser.add_argument(
        "--no-hash-check",
        action="store_true",
        help="Allow a structurally valid input other than the fixed reference files.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate the fixed inputs and exit without fitting or writing outputs.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if args.validate_only:
        _, _, validation = load_inputs(
            args.embeddings, args.manifest, check_hashes=not args.no_hash_check
        )
        print(json.dumps(validation, indent=2))
        return
    summary = run_analysis(
        args.embeddings,
        args.manifest,
        args.output,
        check_hashes=not args.no_hash_check,
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
