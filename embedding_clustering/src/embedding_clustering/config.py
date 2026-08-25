"""Scientific constants for the reviewer-response clustering analysis."""

from __future__ import annotations

from dataclasses import dataclass


ANALYSIS_NAME = "within-model embedding clustering"
EXPECTED_EMBEDDING_SHAPE = (1500, 3072)
EXPECTED_MATRIX_SHA256 = "2fd164b9425b2c1792a4c585ff8613f90fabecf17d11f129ebee53dab000496b"
EXPECTED_MANIFEST_SHA256 = "6a9fa8166ecf8cccf867d7aca2fe6686c1c2777469f2468ae3d4b329077b654f"

K_VALUES = tuple(range(2, 11))
SEEDS = tuple(range(30))
MAX_ITERATIONS = 200
CONVERGENCE_TOLERANCE = 1e-8
SILHOUETTE_TIE_TOLERANCE = 0.005
MINIMUM_CLUSTER_SIZE = 6  # 2% of each 300-interval family dataset
STABILITY_REFERENCE_ARI = 0.90

RUN_ORDER = ("V1", "V2", "V3")
PHASE_BREAKS = (33, 66)
PHASE_LABELS = ("Early (1-33)", "Middle (34-66)", "Late (67-100)")


@dataclass(frozen=True)
class ModelSpec:
    """Stable model metadata plus expected final results used for QA."""

    name: str
    slug: str
    order: int
    selected_k: int
    selected_seed: int
    mean_silhouette: float
    solution_silhouette: float
    stability_median_ari: float
    run_ami: float
    phase_ami: float
    label_sha256_int32: str


MODEL_SPECS = (
    ModelSpec(
        name="DeepSeek-V3",
        slug="deepseek_v3",
        order=1,
        selected_k=4,
        selected_seed=29,
        mean_silhouette=0.3259566178849473,
        solution_silhouette=0.3764611230071076,
        stability_median_ari=0.736982953822724,
        run_ami=0.3872986151277688,
        phase_ami=0.1384723111163488,
        label_sha256_int32="710c76e72a3e3da65e36f032372c58928350492ac7410964985b33def4c13656",
    ),
    ModelSpec(
        name="GPT-4o-mini",
        slug="gpt_4o_mini",
        order=2,
        selected_k=2,
        selected_seed=0,
        mean_silhouette=0.5408407967396199,
        solution_silhouette=0.54084079673962,
        stability_median_ari=1.0,
        run_ami=0.3307677054600124,
        phase_ami=0.0836030524102764,
        label_sha256_int32="cd05768ed43220302ec329ad6cf0548356d9a6431d55e5740dcf3c52962d6afc",
    ),
    ModelSpec(
        name="Phi-4",
        slug="phi_4",
        order=3,
        selected_k=2,
        selected_seed=0,
        mean_silhouette=0.5675724963746929,
        solution_silhouette=0.5675724963746928,
        stability_median_ari=1.0,
        run_ami=0.2024712044954375,
        phase_ami=0.0043819818389112,
        label_sha256_int32="bdae4f8761be7c38632f1dcac8c358b653acd21b6153884509ce5421d9c83ef9",
    ),
    ModelSpec(
        name="GPT-5.6 Terra",
        slug="gpt_5_6_terra",
        order=4,
        selected_k=3,
        selected_seed=0,
        mean_silhouette=0.4530695790636781,
        solution_silhouette=0.4530695790636781,
        stability_median_ari=1.0,
        run_ami=1.0,
        phase_ami=-0.0061611620395816,
        label_sha256_int32="65a4e2e7b28f56187a1ed4eae26a3621cb59dd2efe40a14c41f5837dff67374f",
    ),
    ModelSpec(
        name="Claude Sonnet",
        slug="claude_sonnet",
        order=5,
        selected_k=2,
        selected_seed=0,
        mean_silhouette=0.3558324411105159,
        solution_silhouette=0.3558324411105158,
        stability_median_ari=1.0,
        run_ami=0.7326451023952699,
        phase_ami=-0.0038877949347162,
        label_sha256_int32="2910933eb5732fd1735a2c17fcc2f7aa63d5c2cb3575ab7a54b3cbc48c58c2af",
    ),
)

MODEL_NAMES = tuple(spec.name for spec in MODEL_SPECS)
MODEL_BY_NAME = {spec.name: spec for spec in MODEL_SPECS}
