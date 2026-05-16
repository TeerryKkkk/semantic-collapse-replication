import os
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

# =========================
# Config (edit here)
# =========================
WITHIN_PATH = "WITHIN_final_version.csv"
CROSS_PATH = "CROSS_final_version.csv"
OUT_DIR = "factorwise_window_regression_out_v6_factorwise_holm_finalversion"

WINDOW_SPECS = ["late5", "global"]
LATE5 = list(range(16, 21))
ALPHA = 0.05

# Holm family is all coefficient-level regression p-values within each metric x window_spec x factor x analysis_model_group.
# This groups by model rather than only by factor, because some factors lack model_control or have missing baseline levels.
HOLM_GROUP_COLS = ["metric", "window_spec", "factor", "analysis_model_group"]
WITHIN_FACTOR_BASELINE_LEVEL = {
    "TEMPERATURE": "0.9",
    "MAXTOKEN": "200",
    "RAG": "REGULAR",
    "PROMPT": "t0.9",
    "STEERING": "baseline",
}

EXTERNAL_BASELINE_RULES = {
    "AUTOGEN": {
        "baseline_factor": "TEMPERATURE",
        "baseline_level": "0.9",
        "match_on": "model_control",
        "analysis_type": "per_model_matched_baseline",
    },
}

EXPLORATORY_POOLED_BASELINE_FACTORS = {
    "MIXAGENT": {
        "baseline_factor": "TEMPERATURE",
        "baseline_level": "0.9",
        "analysis_type": "exploratory_pooled_baseline_no_model_control",
    },
    "UNREGULATED": {
        "baseline_factor": "TEMPERATURE",
        "baseline_level": "0.9",
        "analysis_type": "exploratory_pooled_baseline_no_model_control",
    },
}

FACTOR_ORDER = [
    "TEMPERATURE",
    "MAXTOKEN",
    "RAG",
    "PROMPT",
    "AUTOGEN",
    "STEERING",
    "UNREGULATED",
    "MIXAGENT",
]

MODEL_GROUP_ALIASES = {
    "llama8b_steering": "llama8b",
    "llama8b_baseline": "llama8b",
}


def holm_adjust(pvals):
    pvals = np.asarray(pvals, dtype=float)
    out = np.full_like(pvals, np.nan, dtype=float)
    mask = np.isfinite(pvals)
    if mask.sum() == 0:
        return out

    x = pvals[mask]
    m = len(x)
    order = np.argsort(x)
    adj = np.empty(m, dtype=float)
    prev = 0.0

    for i, idx in enumerate(order):
        adj_p = min(1.0, (m - i) * x[idx])
        prev = max(prev, adj_p)
        adj[idx] = prev

    out[mask] = adj
    return out


def _sort_key(x):
    s = str(x)
    try:
        return (0, float(s))
    except Exception:
        return (1, s)


def factor_sort_key(x):
    s = str(x)
    try:
        return (0, FACTOR_ORDER.index(s))
    except ValueError:
        return (1, s)


def sort_levels(values, baseline=None):
    vals = sorted(pd.Series(values).astype(str).unique().tolist(), key=_sort_key)
    if baseline is not None and baseline in vals:
        vals = [baseline] + [x for x in vals if x != baseline]
    return vals


def filter_window_spec(df, window_col, window_spec):
    out = df.copy()
    if window_spec == "late5":
        out = out[out[window_col].isin(LATE5)].copy()
    elif window_spec == "global":
        out = out.copy()
    else:
        raise ValueError(f"Unknown window_spec: {window_spec}")
    out["window_spec"] = window_spec
    return out


def canonical_model_group(model_name):
    s = str(model_name).strip()
    return MODEL_GROUP_ALIASES.get(s, s)


def _scalar(x):
    arr = np.asarray(x)
    if arr.size == 0:
        return np.nan
    return float(arr.reshape(-1)[0])


def prepare_within_window(within_df, window_spec):
    df = within_df.copy()
    df["factor"] = df["factor"].astype(str)
    df["experiment_value"] = df["experiment_value"].astype(str)
    df["window_index"] = df["window_index"].astype(int)
    df["source"] = df["source"].astype(str)
    df["model"] = df["model"].astype(str)
    df = filter_window_spec(df, "window_index", window_spec)

    keep = ["model", "factor", "experiment_value", "source", "window_index", "sim_vs_first", "window_spec"]
    out = df[keep].rename(columns={"source": "cluster_id"})
    out["y"] = -pd.to_numeric(out["sim_vs_first"], errors="coerce")
    out = out.drop(columns=["sim_vs_first"])
    out["model_control"] = out["model"].map(canonical_model_group)
    out["metric"] = "WITHIN_diversity_from_sim_vs_first"
    return out


def prepare_cross_window(cross_df, window_spec):
    df = cross_df.copy()
    df["factor"] = df["factor"].astype(str)
    df = df[df["metric"] == "embed_cosine"].copy()
    df["experiment_value"] = df["experiment_value"].astype(str)
    df["window_id"] = df["window_id"].astype(int)
    df["file_i"] = df["file_i"].astype(str)
    df["file_j"] = df["file_j"].astype(str)
    df["model"] = df["model"].astype(str)
    df = filter_window_spec(df, "window_id", window_spec)

    left = df["file_i"].to_numpy()
    right = df["file_j"].to_numpy()
    pair_id = np.where(left <= right, left + "||" + right, right + "||" + left)

    keep = ["model", "factor", "experiment_value", "window_id", "value", "window_spec"]
    out = df[keep].rename(columns={"window_id": "window_index"})
    out["y"] = -pd.to_numeric(out["value"], errors="coerce")
    out = out.drop(columns=["value"])
    out["cluster_id"] = pair_id
    out["model_control"] = out["model"].map(canonical_model_group)
    out["metric"] = "CROSS_diversity_from_embed_cosine"
    return out


def _status_row(metric_label, window_spec, factor, analysis_model_group, status, **extra):
    row = {
        "metric": metric_label,
        "window_spec": window_spec,
        "factor": factor,
        "analysis_model_group": analysis_model_group,
        "status": status,
    }
    row.update(extra)
    return row


def get_analysis_baseline_level(factor):
    if factor in WITHIN_FACTOR_BASELINE_LEVEL:
        return str(WITHIN_FACTOR_BASELINE_LEVEL[factor])
    if factor in EXTERNAL_BASELINE_RULES:
        return str(EXTERNAL_BASELINE_RULES[factor]["baseline_level"])
    if factor in EXPLORATORY_POOLED_BASELINE_FACTORS:
        return str(EXPLORATORY_POOLED_BASELINE_FACTORS[factor]["baseline_level"])
    return None


def get_factor_analysis_type(factor):
    if factor in EXPLORATORY_POOLED_BASELINE_FACTORS:
        return EXPLORATORY_POOLED_BASELINE_FACTORS[factor]["analysis_type"]
    if factor in EXTERNAL_BASELINE_RULES:
        return EXTERNAL_BASELINE_RULES[factor]["analysis_type"]
    return "per_model_internal_baseline"


def _finalize_combined(combined, factor, analysis_model_group):
    out = combined.copy()
    out["analysis_model_group"] = str(analysis_model_group)
    out["analysis_type"] = get_factor_analysis_type(factor)
    return out


def build_analysis_dataset(df, metric_label):
    assembled = []
    statuses = []

    for window_spec in WINDOW_SPECS:
        df_ws = df[df["window_spec"] == window_spec].copy()
        factors_ws = sorted(df_ws["factor"].astype(str).unique().tolist(), key=factor_sort_key)

        for factor in factors_ws:
            target = df_ws[df_ws["factor"] == factor].copy()
            if target.empty:
                statuses.append(_status_row(metric_label, window_spec, factor, "ALL", "skip_no_target_rows"))
                continue

            if factor in WITHIN_FACTOR_BASELINE_LEVEL:
                base_level = str(WITHIN_FACTOR_BASELINE_LEVEL[factor])
                for mg in sorted(target["model_control"].astype(str).unique().tolist()):
                    sub = target[target["model_control"].astype(str) == mg].copy()
                    levels_present = sort_levels(sub["experiment_value"].unique(), baseline=base_level)
                    if base_level not in set(map(str, sub["experiment_value"].unique())):
                        statuses.append(_status_row(
                            metric_label, window_spec, factor, mg,
                            "skip_missing_within_factor_baseline",
                            baseline_level=base_level,
                            levels_found=",".join(map(str, sorted(sub["experiment_value"].astype(str).unique().tolist()))),
                        ))
                        continue
                    if len(levels_present) < 2:
                        statuses.append(_status_row(
                            metric_label, window_spec, factor, mg,
                            "skip_only_baseline_level_present",
                            baseline_level=base_level,
                            levels_found=",".join(levels_present),
                        ))
                        continue
                    assembled.append(_finalize_combined(sub, factor, mg))
                    statuses.append(_status_row(
                        metric_label, window_spec, factor, mg,
                        "ok_within_factor_baseline",
                        analysis_type=get_factor_analysis_type(factor),
                        baseline_level=base_level,
                        n_rows=int(len(sub)),
                        n_models_in_data=int(sub["model_control"].nunique()),
                        levels_included=",".join(levels_present),
                    ))
                continue

            if factor in EXTERNAL_BASELINE_RULES:
                rule = EXTERNAL_BASELINE_RULES[factor]
                base_factor = str(rule["baseline_factor"])
                base_level = str(rule["baseline_level"])
                match_on = str(rule.get("match_on", "model_control"))
                baseline_all = df_ws[(df_ws["factor"] == base_factor) & (df_ws["experiment_value"].astype(str) == base_level)].copy()
                if baseline_all.empty:
                    statuses.append(_status_row(
                        metric_label, window_spec, factor, "ALL",
                        "skip_missing_external_baseline_rows",
                        baseline_factor=base_factor,
                        baseline_level=base_level,
                    ))
                    continue
                for mg in sorted(target["model_control"].astype(str).unique().tolist()):
                    sub_target = target[target["model_control"].astype(str) == mg].copy()
                    sub_base = baseline_all[baseline_all[match_on].astype(str) == mg].copy()
                    if sub_target.empty:
                        statuses.append(_status_row(metric_label, window_spec, factor, mg, "skip_no_target_rows_for_model_group"))
                        continue
                    if sub_base.empty:
                        statuses.append(_status_row(
                            metric_label, window_spec, factor, mg,
                            "skip_no_matching_external_baseline_for_model_group",
                            baseline_factor=base_factor,
                            baseline_level=base_level,
                            match_on=match_on,
                        ))
                        continue
                    sub_base = sub_base.copy()
                    sub_base["factor"] = factor
                    sub_base["experiment_value"] = base_level
                    combined = pd.concat([sub_base, sub_target], ignore_index=True, sort=False)
                    levels_present = sort_levels(combined["experiment_value"].unique(), baseline=base_level)
                    if len(levels_present) < 2:
                        statuses.append(_status_row(
                            metric_label, window_spec, factor, mg,
                            "skip_only_baseline_level_present",
                            baseline_level=base_level,
                            levels_found=",".join(levels_present),
                        ))
                        continue
                    assembled.append(_finalize_combined(combined, factor, mg))
                    statuses.append(_status_row(
                        metric_label, window_spec, factor, mg,
                        "ok_external_baseline_matched_per_model",
                        analysis_type=get_factor_analysis_type(factor),
                        baseline_factor=base_factor,
                        baseline_level=base_level,
                        match_on=match_on,
                        n_rows=int(len(combined)),
                        n_models_in_data=int(combined["model_control"].nunique()),
                        levels_included=",".join(levels_present),
                    ))
                continue

            if factor in EXPLORATORY_POOLED_BASELINE_FACTORS:
                rule = EXPLORATORY_POOLED_BASELINE_FACTORS[factor]
                base_factor = str(rule["baseline_factor"])
                base_level = str(rule["baseline_level"])
                baseline_all = df_ws[(df_ws["factor"] == base_factor) & (df_ws["experiment_value"].astype(str) == base_level)].copy()
                if baseline_all.empty:
                    statuses.append(_status_row(
                        metric_label, window_spec, factor, "ALL",
                        "skip_missing_pooled_baseline_rows",
                        baseline_factor=base_factor,
                        baseline_level=base_level,
                    ))
                    continue
                for mg in sorted(target["model_control"].astype(str).unique().tolist()):
                    sub_target = target[target["model_control"].astype(str) == mg].copy()
                    if sub_target.empty:
                        statuses.append(_status_row(metric_label, window_spec, factor, mg, "skip_no_target_rows_for_model_group"))
                        continue
                    sub_base = baseline_all.copy()
                    sub_base["factor"] = factor
                    sub_base["experiment_value"] = base_level
                    combined = pd.concat([sub_base, sub_target], ignore_index=True, sort=False)
                    levels_present = sort_levels(combined["experiment_value"].unique(), baseline=base_level)
                    if len(levels_present) < 2:
                        statuses.append(_status_row(
                            metric_label, window_spec, factor, mg,
                            "skip_only_baseline_level_present",
                            baseline_level=base_level,
                            levels_found=",".join(levels_present),
                        ))
                        continue
                    assembled.append(_finalize_combined(combined, factor, mg))
                    statuses.append(_status_row(
                        metric_label, window_spec, factor, mg,
                        "ok_exploratory_pooled_baseline_per_model",
                        analysis_type=get_factor_analysis_type(factor),
                        baseline_factor=base_factor,
                        baseline_level=base_level,
                        n_rows=int(len(combined)),
                        n_models_in_data=int(combined["model_control"].nunique()),
                        levels_included=",".join(levels_present),
                    ))
                continue

            levels = sorted(target["experiment_value"].astype(str).unique().tolist(), key=_sort_key)
            statuses.append(_status_row(
                metric_label, window_spec, factor, "ALL",
                "skip_no_baseline_rule",
                levels_found=",".join(levels),
                n_rows=int(len(target)),
                n_models_in_data=int(target["model_control"].nunique()),
            ))

    out = pd.concat(assembled, ignore_index=True, sort=False) if assembled else pd.DataFrame(columns=df.columns)
    status_df = pd.DataFrame(statuses)
    return out, status_df


def fit_clustered_ols(sub, base_level):
    reference_term = f"Treatment(reference={repr(base_level)})"
    factor_term = f"C(experiment_value, {reference_term})"
    formula = f"y ~ {factor_term} + C(window_index)"
    fit = smf.ols(formula, data=sub).fit(
        cov_type="cluster",
        cov_kwds={"groups": sub["cluster_id"]},
        use_t=True,
    )
    return fit, factor_term, formula


def get_nonbaseline_coef_names(fit, factor_term):
    prefix = factor_term + "[T."
    return [name for name in fit.params.index if name.startswith(prefix)]


def analyze_factor_window_model(sub, metric_label, window_spec, factor, analysis_model_group):
    base_level = get_analysis_baseline_level(factor)
    if base_level is None:
        return pd.DataFrame()

    sub = sub.copy()
    sub = sub[np.isfinite(sub["y"])].copy()
    if sub.empty:
        return pd.DataFrame()

    levels_present = sort_levels(sub["experiment_value"].unique(), baseline=base_level)
    if base_level not in levels_present or len(levels_present) < 2:
        return pd.DataFrame()

    sub = sub[sub["experiment_value"].isin(levels_present)].copy()
    sub["experiment_value"] = pd.Categorical(sub["experiment_value"], categories=levels_present, ordered=True)
    sub["cluster_id"] = sub["cluster_id"].astype(str)

    n_clusters = sub["cluster_id"].nunique()
    if n_clusters < 2:
        return pd.DataFrame()

    try:
        fit, factor_term, formula = fit_clustered_ols(sub, base_level)
    except Exception:
        return pd.DataFrame()

    coef_names = get_nonbaseline_coef_names(fit, factor_term)
    if len(coef_names) == 0:
        return pd.DataFrame()

    rows = []
    eye = np.eye(len(fit.params))
    for name in coef_names:
        level = name.split("[T.", 1)[1][:-1]
        idx = fit.params.index.get_loc(name)
        t1 = fit.t_test(eye[idx])
        coef = _scalar(t1.effect)
        p_raw = _scalar(t1.pvalue)
        rows.append({
            "metric": metric_label,
            "window_spec": window_spec,
            "factor": factor,
            "analysis_model_group": analysis_model_group,
            "analysis_type": sub["analysis_type"].iloc[0],
            "baseline_level": base_level,
            "level": level,
            "coef_name": name,
            "formula": formula,
            "n_obs": int(len(sub)),
            "n_clusters": int(n_clusters),
            "n_models_in_data": int(sub["model_control"].nunique()),
            "n_windows": int(sub["window_index"].nunique()),
            "n_obs_level": int((sub["experiment_value"] == level).sum()),
            "coefficient": coef,
            "se": _scalar(t1.sd),
            "t": _scalar(t1.tvalue),
            "df": _scalar(getattr(t1, "df_denom", np.nan)),
            "raw_p": p_raw,
            "direction": "increase_diversity" if np.isfinite(coef) and coef > 0 else (
                "decrease_diversity" if np.isfinite(coef) and coef < 0 else "no_change"
            ),
            "status": "ok",
        })

    return pd.DataFrame(rows)


def analyze_window_level(df, metric_label):
    level_tables = []
    group_cols = ["window_spec", "factor", "analysis_model_group"]

    for (window_spec, factor, analysis_model_group), sub in df.groupby(group_cols, sort=True):
        level_df = analyze_factor_window_model(sub, metric_label, window_spec, factor, analysis_model_group)
        if not level_df.empty:
            level_tables.append(level_df)

    level_df = pd.concat(level_tables, ignore_index=True) if level_tables else pd.DataFrame()
    if level_df.empty:
        return level_df

    level_df["holm_p"] = np.nan
    for _, idx in level_df.groupby(HOLM_GROUP_COLS).groups.items():
        ii = np.array(list(idx))
        level_df.loc[ii, "holm_p"] = holm_adjust(level_df.loc[ii, "raw_p"].to_numpy())

    level_df["sig_raw_0.05"] = level_df["raw_p"] < ALPHA
    level_df["sig_holm_0.05"] = level_df["holm_p"] < ALPHA

    level_df = level_df.sort_values(
        ["metric", "window_spec", "factor", "analysis_model_group", "level"],
        key=lambda s: s.map(factor_sort_key) if s.name == "factor" else s,
    ).reset_index(drop=True)

    return level_df


def build_summary_table(level_df):
    if level_df.empty:
        return pd.DataFrame()

    summary = (
        level_df
        .groupby(["metric", "window_spec"], as_index=False)
        .agg(
            n_coefficients=("coef_name", "size"),
            n_sig_raw=("sig_raw_0.05", "sum"),
            n_sig_holm=("sig_holm_0.05", "sum"),
            n_sig_holm_positive=("coefficient", lambda s: int(((s > 0) & level_df.loc[s.index, "sig_holm_0.05"]).sum())),
            n_sig_holm_negative=("coefficient", lambda s: int(((s < 0) & level_df.loc[s.index, "sig_holm_0.05"]).sum())),
        )
    )
    return summary


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    within = pd.read_csv(WITHIN_PATH)
    cross = pd.read_csv(CROSS_PATH)

    within_prepared = pd.concat([prepare_within_window(within, ws) for ws in WINDOW_SPECS], ignore_index=True)
    cross_prepared = pd.concat([prepare_cross_window(cross, ws) for ws in WINDOW_SPECS], ignore_index=True)

    within_all, within_build_status = build_analysis_dataset(within_prepared, "WITHIN_diversity_from_sim_vs_first")
    cross_all, cross_build_status = build_analysis_dataset(cross_prepared, "CROSS_diversity_from_embed_cosine")

    within_levels = analyze_window_level(within_all, "WITHIN_diversity_from_sim_vs_first")
    cross_levels = analyze_window_level(cross_all, "CROSS_diversity_from_embed_cosine")
    levels_all = pd.concat([within_levels, cross_levels], ignore_index=True)

    summary_df = build_summary_table(levels_all)

    levels_all.to_csv(os.path.join(OUT_DIR, "per_factor_per_model_level_regression.csv"), index=False)
    summary_df.to_csv(os.path.join(OUT_DIR, "regression_significance_summary.csv"), index=False)
    within_all.to_csv(os.path.join(OUT_DIR, "within_window_level_used.csv"), index=False)
    cross_all.to_csv(os.path.join(OUT_DIR, "cross_window_level_used.csv"), index=False)
    within_build_status.to_csv(os.path.join(OUT_DIR, "within_factor_model_build_status.csv"), index=False)
    cross_build_status.to_csv(os.path.join(OUT_DIR, "cross_factor_model_build_status.csv"), index=False)

    print("Wrote outputs to:", OUT_DIR)
    print("Coefficient-level regression:", os.path.join(OUT_DIR, "per_factor_per_model_level_regression.csv"))
    print("Summary:", os.path.join(OUT_DIR, "regression_significance_summary.csv"))


if __name__ == "__main__":
    main()
