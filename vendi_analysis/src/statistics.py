from __future__ import annotations

import numpy as np
import pandas as pd

from config import BOOTSTRAP_DRAWS, EXPECTED_FAMILIES, EXPECTED_RUNS_PER_FAMILY
from src.vendi import deterministic_seed


VALUE_COLUMNS = ["early", "middle", "late", "late_minus_early"]


def hierarchical_family_run_bootstrap(frame: pd.DataFrame) -> np.ndarray:
    """Resample model families, then runs within each sampled family."""
    families = sorted(frame["family"].astype(str).unique())
    if families != sorted(EXPECTED_FAMILIES):
        raise RuntimeError(f"Unexpected model families: {families}")
    arrays = {
        family: frame[frame["family"].astype(str).eq(family)][VALUE_COLUMNS].to_numpy(dtype=float)
        for family in families
    }
    if any(len(array) != EXPECTED_RUNS_PER_FAMILY for array in arrays.values()):
        raise RuntimeError("Each model family must contribute exactly three baseline runs")

    rng = np.random.default_rng(deterministic_seed("hierarchical_bootstrap"))
    output = np.empty((BOOTSTRAP_DRAWS, len(VALUE_COLUMNS)), dtype=np.float64)
    for draw in range(BOOTSTRAP_DRAWS):
        sampled_families = rng.choice(families, size=len(families), replace=True)
        family_means = []
        for family in sampled_families:
            array = arrays[str(family)]
            sampled_runs = rng.integers(0, len(array), size=len(array))
            family_means.append(array[sampled_runs].mean(axis=0))
        output[draw] = np.mean(family_means, axis=0)
    return output


def analyze(frame: pd.DataFrame) -> pd.DataFrame:
    point = frame.groupby("family")[VALUE_COLUMNS].mean().mean(axis=0)
    draws = hierarchical_family_run_bootstrap(frame)
    rows: list[dict[str, object]] = []
    for column_index, metric in enumerate(VALUE_COLUMNS):
        low, high = np.quantile(draws[:, column_index], [0.025, 0.975])
        rows.append(
            {
                "metric": metric,
                "estimate": float(point[metric]),
                "ci95_low": float(low),
                "ci95_high": float(high),
                "model_families": len(EXPECTED_FAMILIES),
                "runs_per_family": EXPECTED_RUNS_PER_FAMILY,
                "total_runs": len(frame),
                "bootstrap_draws": BOOTSTRAP_DRAWS,
            }
        )
    return pd.DataFrame(rows)
