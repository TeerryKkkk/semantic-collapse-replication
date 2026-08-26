"""Post-selection metadata summaries for fixed embedding clusters."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_mutual_info_score

from .config import PHASE_BREAKS, PHASE_LABELS


def interaction_phase(interval: int) -> str:
    """Map interval 1-100 to the fixed three-phase diagnostic convention."""

    if interval <= PHASE_BREAKS[0]:
        return PHASE_LABELS[0]
    if interval <= PHASE_BREAKS[1]:
        return PHASE_LABELS[1]
    return PHASE_LABELS[2]


def association_summary(assignments: pd.DataFrame) -> dict[str, float]:
    """Compute diagnostic AMI values after clustering is fixed."""

    required = {"cluster", "run_version", "interval_number"}
    missing = sorted(required - set(assignments.columns))
    if missing:
        raise ValueError(f"Assignments are missing columns: {missing}")
    phases = assignments["interval_number"].astype(int).map(interaction_phase)
    return {
        "run_ami": float(
            adjusted_mutual_info_score(
                assignments["cluster"].astype(int), assignments["run_version"]
            )
        ),
        "phase_ami": float(
            adjusted_mutual_info_score(assignments["cluster"].astype(int), phases)
        ),
    }


def assignment_label_hash(assignments: pd.DataFrame) -> str:
    """Hash labels in family-row order for result verification."""

    import hashlib

    ordered = assignments.sort_values("family_row_index_0_based")
    labels = ordered["cluster"].to_numpy(np.int32)
    return hashlib.sha256(labels.tobytes()).hexdigest()
