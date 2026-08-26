"""Scientific constants for the embedding-clustering analysis."""

from __future__ import annotations

from dataclasses import dataclass


ANALYSIS_NAME = "within-model embedding clustering"
EXPECTED_EMBEDDING_SHAPE = (1500, 3072)

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
    """Stable model metadata plus reference values used for verification."""

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
    ),
)

MODEL_NAMES = tuple(spec.name for spec in MODEL_SPECS)
MODEL_BY_NAME = {spec.name: spec for spec in MODEL_SPECS}
