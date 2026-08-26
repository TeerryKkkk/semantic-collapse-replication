#!/usr/bin/env python3
"""Complete-message Reddit-versus-LLM semantic-similarity analysis.

The analysis has two commands.

``compute`` expects a UTF-8 CSV with these columns:

    source,unit_id,model_family,message_index,text

``source`` is ``Reddit`` or ``LLM``. A Reddit ``unit_id`` is a thread ID; an
LLM ``unit_id`` is a run ID. Reddit ``model_family`` is blank. Input text must
already be cleaned and ordered, with one complete comment/utterance per row.
The command embeds complete messages without splitting, truncating, joining,
or discarding text and writes one C1-vs-Ci or U1-vs-Ui row per message.

``summarize`` consumes that unit-level CSV, recreates the frozen 2--92 curves,
bootstrap intervals, family summaries, token diagnostics, the manuscript's pooled
message-length sensitivity analysis, and a three-panel figure. The fixed horizon of 92 is the pre-established >=50% Reddit-cohort
retention limit in the 2,000-thread reference cohort.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import os
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


MODEL = "text-embedding-3-large"
TOKENIZER = "cl100k_base"
DIMENSION = 3072
MAX_INDEX = 92
BOOTSTRAP_RESAMPLES = 10_000
LENGTH_BOOTSTRAP_RESAMPLES = 5_000
EXPECTED_PAIR_COUNT = 157_694
API_BATCH_SIZE = 256
EXPECTED_FAMILIES = frozenset({
    "DeepSeek-V3",
    "GPT-4-mini",
    "GPT-5.6 Luna",
    "Phi-4",
})
MODEL_ALIASES = {
    "GPT-4o-mini": "GPT-4-mini",
    "GPT-4-mini": "GPT-4-mini",
}
REDDIT_SEED_BASE = 20260810
REDDIT_VERSION = "reddit_segmentation_comparison_v1_2026-08-10"
LLM_SEED_BASE = 20260730
LLM_VERSION = "complete_message_200round_v1_2026-08-10"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise RuntimeError(f"Refusing to write empty output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def stable_seed(base: int, *parts: object) -> int:
    digest = hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).digest()
    return (base + int.from_bytes(digest[:8], "big")) % (2**32)


def token_hash(token_ids: list[int]) -> str:
    values = np.asarray(token_ids, dtype="<i4")
    digest = hashlib.sha256()
    digest.update(TOKENIZER.encode("ascii") + b"\0le32\0")
    digest.update(values.tobytes())
    return digest.hexdigest()


def canonical_model_family(value: object) -> str:
    name = str(value).strip()
    return MODEL_ALIASES.get(name, name)


def validate_source_counts(rows: list[dict[str, object]]) -> None:
    reddit = {str(row["unit_id"]) for row in rows if row["source"] == "Reddit"}
    llm = {str(row["unit_id"]) for row in rows if row["source"] == "LLM"}
    families = {
        canonical_model_family(row["model_family"])
        for row in rows
        if row["source"] == "LLM"
    }
    if len(reddit) != 2_000:
        raise RuntimeError(f"Expected 2,000 Reddit threads, found {len(reddit)}")
    if not llm:
        raise RuntimeError("No LLM runs were found")
    if families != EXPECTED_FAMILIES:
        raise RuntimeError(f"Unexpected LLM model families: {sorted(families)}")


def validate_sequences(rows: list[dict[str, object]], require_full_llm: bool = True) -> None:
    grouped: dict[tuple[str, str], list[int]] = defaultdict(list)
    seen: set[tuple[str, str, int]] = set()
    for row in rows:
        source = str(row["source"])
        unit_id = str(row["unit_id"])
        index = int(row["message_index"])
        if source not in {"Reddit", "LLM"} or index < 1:
            raise RuntimeError("Invalid source or message index")
        key = source, unit_id, index
        if key in seen:
            raise RuntimeError(f"Duplicate message row: {key}")
        seen.add(key)
        grouped[(source, unit_id)].append(index)
    for (source, unit_id), indices in grouped.items():
        ordered = sorted(indices)
        if ordered != list(range(1, max(ordered) + 1)):
            raise RuntimeError(f"Non-contiguous message sequence: {source} {unit_id}")
        if require_full_llm and source == "LLM" and max(ordered) < MAX_INDEX:
            raise RuntimeError(f"LLM run lacks U{MAX_INDEX}: {unit_id}")


def embed_token_sequences(sequences: list[list[int]], api_key: str) -> np.ndarray:
    from openai import OpenAI

    client = OpenAI(api_key=api_key, timeout=180.0)
    vectors = np.empty((len(sequences), DIMENSION), dtype=np.float32)
    for start in range(0, len(sequences), API_BATCH_SIZE):
        stop = min(start + API_BATCH_SIZE, len(sequences))
        delay = 2.0
        for attempt in range(8):
            try:
                response = client.embeddings.create(model=MODEL, input=sequences[start:stop])
                batch = np.asarray([item.embedding for item in response.data], dtype=np.float32)
                if batch.shape != (stop - start, DIMENSION):
                    raise RuntimeError(f"Unexpected embedding shape: {batch.shape}")
                norms = np.linalg.norm(batch, axis=1, keepdims=True)
                if np.any(~np.isfinite(norms)) or np.any(norms <= 0):
                    raise RuntimeError("Invalid embedding norm")
                vectors[start:stop] = batch / norms
                break
            except Exception:
                if attempt == 7:
                    raise
                time.sleep(delay)
                delay = min(60.0, delay * 2.0)
    return vectors


def compute(args: argparse.Namespace) -> None:
    import tiktoken

    raw_rows = read_csv(args.messages)
    required = {"source", "unit_id", "model_family", "message_index", "text"}
    if not raw_rows or not required.issubset(raw_rows[0]):
        raise RuntimeError(f"Input must contain columns: {sorted(required)}")
    rows: list[dict[str, object]] = []
    for row in raw_rows:
        text = row["text"]
        if not text or not text.strip():
            raise RuntimeError("Message text must be nonempty and pre-cleaned")
        rows.append(
            {
                "source": row["source"],
                "unit_id": row["unit_id"],
                "model_family": canonical_model_family(row["model_family"]) if row["source"] == "LLM" else "",
                "message_index": int(row["message_index"]),
                "text": text,
            }
        )
    validate_source_counts(rows)
    validate_sequences(rows)
    selected = [row for row in rows if int(row["message_index"]) <= MAX_INDEX]
    selected.sort(key=lambda row: (str(row["source"]), str(row["unit_id"]), int(row["message_index"])))

    encoding = tiktoken.get_encoding(TOKENIZER)
    unique_hash_to_index: dict[str, int] = {}
    unique_sequences: list[list[int]] = []
    occurrence_to_unique: list[int] = []
    lengths: list[int] = []
    hashes: list[str] = []
    for row in selected:
        sequence = encoding.encode(str(row["text"]))
        if not sequence or len(sequence) > 8191:
            raise RuntimeError(f"Invalid complete-message length: {row['source']} {row['unit_id']} {row['message_index']}")
        digest = token_hash(sequence)
        unique_index = unique_hash_to_index.get(digest)
        if unique_index is None:
            unique_index = len(unique_sequences)
            unique_hash_to_index[digest] = unique_index
            unique_sequences.append(sequence)
        occurrence_to_unique.append(unique_index)
        lengths.append(len(sequence))
        hashes.append(digest)

    api_key = os.environ.get(args.api_key_env, "").strip()
    if not api_key:
        raise RuntimeError(f"Set the API key in environment variable {args.api_key_env}")
    unique_vectors = embed_token_sequences(unique_sequences, api_key)
    vectors = unique_vectors[np.asarray(occurrence_to_unique)]
    output: list[dict[str, object]] = []
    cursor = 0
    grouped_rows: dict[tuple[str, str], list[tuple[dict[str, object], int]]] = defaultdict(list)
    for row in selected:
        grouped_rows[(str(row["source"]), str(row["unit_id"]))].append((row, cursor))
        cursor += 1
    for key in sorted(grouped_rows):
        items = sorted(grouped_rows[key], key=lambda item: int(item[0]["message_index"]))
        anchor_vector = vectors[items[0][1]]
        for row, occurrence in items:
            index = int(row["message_index"])
            cosine = 1.0 if index == 1 else float(np.dot(anchor_vector, vectors[occurrence]))
            output.append(
                {
                    "source": row["source"],
                    "unit_id": row["unit_id"],
                    "model_family": canonical_model_family(row["model_family"]) if row["source"] == "LLM" else "",
                    "message_index": index,
                    "comparison": f"{'C' if row['source'] == 'Reddit' else 'U'}1_vs_{'C' if row['source'] == 'Reddit' else 'U'}{index}",
                    "visualization_only": int(index == 1),
                    "token_count": lengths[occurrence],
                    "token_hash": hashes[occurrence],
                    "cosine_similarity": cosine,
                    "independent_unit": "thread" if row["source"] == "Reddit" else "LLM run",
                    "tokenizer": TOKENIZER,
                    "embedding_model": MODEL,
                }
            )
    write_csv(args.output, output)


def run_weights(n_runs: int, seed: int, draws: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(n_runs, np.full(n_runs, 1.0 / n_runs), size=draws)
    return counts.astype(np.float64) / n_runs


def family_weights(families: list[str], seed: int, draws: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    names = sorted(set(families))
    family_array = np.asarray(families)
    indices = {name: np.flatnonzero(family_array == name) for name in names}
    weights = np.zeros((draws, len(families)), dtype=np.float64)
    for draw in range(draws):
        chosen_families = rng.integers(0, len(names), size=len(names))
        for family_index in chosen_families:
            run_indices = indices[names[int(family_index)]]
            chosen_runs = rng.integers(0, len(run_indices), size=len(run_indices))
            counts = np.bincount(chosen_runs, minlength=len(run_indices)).astype(np.float64)
            weights[draw, run_indices] += counts / (len(run_indices) * len(names))
    return weights


def bootstrap_means(values: np.ndarray, seed: int, draws: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    output = np.empty(draws, dtype=np.float64)
    for start in range(0, draws, 500):
        stop = min(start + 500, draws)
        selected = rng.integers(0, len(values), size=(stop - start, len(values)))
        output[start:stop] = values[selected].mean(axis=1)
    return output


def percentile_ci(values: np.ndarray) -> tuple[float, float]:
    low, high = np.quantile(values, [0.025, 0.975])
    return float(low), float(high)


def family_point(values: np.ndarray, families: list[str]) -> float:
    array = np.asarray(families)
    return float(np.mean([np.mean(values[array == family]) for family in sorted(set(families))]))


def distribution(values: list[int]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "n": len(values),
        "min": int(array.min()),
        "q25": float(np.quantile(array, 0.25)),
        "median": float(np.median(array)),
        "mean": float(array.mean()),
        "q75": float(np.quantile(array, 0.75)),
        "q90": float(np.quantile(array, 0.90)),
        "q95": float(np.quantile(array, 0.95)),
        "max": int(array.max()),
    }


def _ols_source_coefficient(source: np.ndarray, length_term: np.ndarray, y: np.ndarray) -> float:
    design = np.column_stack(
        [np.ones(len(y), dtype=np.float64), source.astype(np.float64), length_term.astype(np.float64)]
    )
    beta, *_ = np.linalg.lstsq(design, y.astype(np.float64), rcond=None)
    return float(beta[1])


def _cluster_sufficient_statistics(
    source: np.ndarray,
    length_term: np.ndarray,
    y: np.ndarray,
    unit_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    ids = unit_ids.astype(str)
    names_array, inverse = np.unique(ids, return_inverse=True)
    design = np.column_stack(
        [
            np.ones(len(y), dtype=np.float64),
            source.astype(np.float64),
            length_term.astype(np.float64),
        ]
    )
    xtx = np.zeros((len(names_array), 3, 3), dtype=np.float64)
    xty = np.zeros((len(names_array), 3), dtype=np.float64)
    np.add.at(xtx, inverse, np.einsum("ni,nj->nij", design, design))
    np.add.at(xty, inverse, design * y.astype(np.float64)[:, None])
    return xtx, xty, names_array.tolist()


def length_sensitivity(
    rows: list[dict[str, object]],
    *,
    bootstrap_resamples: int,
) -> list[dict[str, object]]:
    """Reproduce Supplementary Note 2.8's pooled message-length sensitivity test."""

    anchors: dict[tuple[str, str], int] = {}
    for row in rows:
        if int(row["message_index"]) == 1:
            anchors[(str(row["source"]), str(row["unit_id"]))] = int(row["token_count"])

    pair_rows: list[tuple[str, str, float, float, float]] = []
    for row in rows:
        index = int(row["message_index"])
        if index < 2 or index > MAX_INDEX:
            continue
        key = (str(row["source"]), str(row["unit_id"]))
        if key not in anchors:
            raise RuntimeError(f"Missing message-1 token count for {key}")
        combined_tokens = anchors[key] + int(row["token_count"])
        pair_rows.append(
            (
                str(row["source"]),
                str(row["unit_id"]),
                1.0 if row["source"] == "LLM" else 0.0,
                float(combined_tokens),
                float(row["cosine_similarity"]),
            )
        )

    if len(pair_rows) != EXPECTED_PAIR_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_PAIR_COUNT:,} anchor-target pairs for indices 2-{MAX_INDEX}, "
            f"found {len(pair_rows):,}"
        )

    domains = np.asarray([item[0] for item in pair_rows], dtype=object)
    unit_ids = np.asarray([item[1] for item in pair_rows], dtype=object)
    source = np.asarray([item[2] for item in pair_rows], dtype=np.float64)
    combined = np.asarray([item[3] for item in pair_rows], dtype=np.float64)
    y = np.asarray([item[4] for item in pair_rows], dtype=np.float64)
    if np.any(combined <= 0):
        raise RuntimeError("Combined anchor-target token counts must be positive")

    reddit_mean = float(y[domains == "Reddit"].mean())
    llm_mean = float(y[domains == "LLM"].mean())
    unadjusted = llm_mean - reddit_mean
    linear_point = _ols_source_coefficient(source, combined, y)
    log_point = _ols_source_coefficient(source, np.log(combined), y)

    # Trajectory-level bootstrap: sample Reddit threads and LLM runs independently
    # with replacement, retaining every pair belonging to a sampled trajectory.
    stats: dict[str, dict[str, tuple[np.ndarray, np.ndarray, list[str]]]] = {}
    for spec, term in (("linear", combined), ("log", np.log(combined))):
        stats[spec] = {}
        for domain in ("Reddit", "LLM"):
            keep = domains == domain
            stats[spec][domain] = _cluster_sufficient_statistics(
                source[keep], term[keep], y[keep], unit_ids[keep]
            )

    rng = np.random.default_rng(stable_seed(REDDIT_SEED_BASE, "length_sensitivity", "trajectory_bootstrap"))
    linear_draws = np.empty(bootstrap_resamples, dtype=np.float64)
    log_draws = np.empty(bootstrap_resamples, dtype=np.float64)
    for draw in range(bootstrap_resamples):
        sampled: dict[str, np.ndarray] = {}
        for domain in ("Reddit", "LLM"):
            n_units = len(stats["linear"][domain][2])
            sampled[domain] = rng.integers(0, n_units, size=n_units)
        for spec, target in (("linear", linear_draws), ("log", log_draws)):
            total_xtx = np.zeros((3, 3), dtype=np.float64)
            total_xty = np.zeros(3, dtype=np.float64)
            for domain in ("Reddit", "LLM"):
                xtx, xty, _ = stats[spec][domain]
                idx = sampled[domain]
                total_xtx += xtx[idx].sum(axis=0)
                total_xty += xty[idx].sum(axis=0)
            try:
                beta = np.linalg.solve(total_xtx, total_xty)
            except np.linalg.LinAlgError:
                beta = np.linalg.lstsq(total_xtx, total_xty, rcond=None)[0]
            target[draw] = float(beta[1])

    linear_low, linear_high = percentile_ci(linear_draws)
    log_low, log_high = percentile_ci(log_draws)
    return [
        {
            "specification": "unadjusted",
            "n_anchor_target_pairs": len(pair_rows),
            "llm_minus_reddit": unadjusted,
            "ci95_low": "",
            "ci95_high": "",
            "length_variable": "none",
            "bootstrap_resamples": "",
        },
        {
            "specification": "linear_length",
            "n_anchor_target_pairs": len(pair_rows),
            "llm_minus_reddit": linear_point,
            "ci95_low": linear_low,
            "ci95_high": linear_high,
            "length_variable": "anchor_tokens + target_tokens",
            "bootstrap_resamples": bootstrap_resamples,
        },
        {
            "specification": "log_length",
            "n_anchor_target_pairs": len(pair_rows),
            "llm_minus_reddit": log_point,
            "ci95_low": log_low,
            "ci95_high": log_high,
            "length_variable": "log(anchor_tokens + target_tokens)",
            "bootstrap_resamples": bootstrap_resamples,
        },
    ]


def plot_figure(
    curve_rows: list[dict[str, object]],
    family_rows: list[dict[str, object]],
    reddit_lengths: list[int],
    llm_lengths: dict[str, list[int]],
    output_stem: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7,
            "axes.labelsize": 7,
            "axes.titlesize": 8,
            "xtick.labelsize": 6.3,
            "ytick.labelsize": 6.3,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.7,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    palette = {
        "DeepSeek-V3": "#2563EB",
        "GPT-4-mini": "#0F766E",
        "GPT-5.6 Luna": "#D97706",
        "Phi-4": "#7C3AED",
    }
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(183 / 25.4, 2.55),
        gridspec_kw={"width_ratios": [1.08, 1.04, 0.88]},
    )
    x = np.asarray([int(row["message_index"]) for row in curve_rows])
    reddit = np.asarray([float(row["reddit_mean"]) for row in curve_rows])
    reddit_low = np.asarray([1.0 if row["reddit_ci95_low"] == "" else float(row["reddit_ci95_low"]) for row in curve_rows])
    reddit_high = np.asarray([1.0 if row["reddit_ci95_high"] == "" else float(row["reddit_ci95_high"]) for row in curve_rows])
    llm = np.asarray([float(row["llm_family_equal_mean"]) for row in curve_rows])
    llm_low = np.asarray([1.0 if row["llm_family_equal_ci95_low"] == "" else float(row["llm_family_equal_ci95_low"]) for row in curve_rows])
    llm_high = np.asarray([1.0 if row["llm_family_equal_ci95_high"] == "" else float(row["llm_family_equal_ci95_high"]) for row in curve_rows])
    axes[0].fill_between(x, reddit_low, reddit_high, color="#444444", alpha=0.12, linewidth=0)
    axes[0].plot(x, reddit, color="#333333", lw=1.3, label="Reddit complete comments")
    axes[0].fill_between(x, llm_low, llm_high, color="#D97706", alpha=0.15, linewidth=0)
    axes[0].plot(x, llm, color="#D97706", lw=1.3, label="LLM complete utterances")
    axes[0].set(xlim=(1, MAX_INDEX), ylim=(0.2, 1.02), xlabel="Complete-message index", ylabel="Cosine similarity to message 1")
    axes[0].set_title("a  Natural-message comparison", loc="left", fontweight="bold")
    axes[0].grid(axis="y", color="#dddddd", linewidth=0.45)
    axes[0].legend(loc="best")

    for family, color in palette.items():
        selected = [row for row in family_rows if row["model_family"] == family]
        fx = np.asarray([int(row["message_index"]) for row in selected])
        mean = np.asarray([float(row["mean_similarity"]) for row in selected])
        low = np.asarray([1.0 if row["ci95_low"] == "" else float(row["ci95_low"]) for row in selected])
        high = np.asarray([1.0 if row["ci95_high"] == "" else float(row["ci95_high"]) for row in selected])
        axes[1].fill_between(fx, low, high, color=color, alpha=0.08, linewidth=0)
        axes[1].plot(fx, mean, color=color, lw=1.05, label=family)
    axes[1].set(xlim=(1, MAX_INDEX), ylim=(0.2, 1.02), xlabel="Complete-utterance index", ylabel="Cosine similarity to U1")
    axes[1].set_title("b  LLM model families", loc="left", fontweight="bold")
    axes[1].grid(axis="y", color="#dddddd", linewidth=0.45)
    axes[1].legend(loc="best", fontsize=5.7)

    labels = ["Reddit"] + list(palette)
    data = [reddit_lengths] + [llm_lengths[family] for family in palette]
    colors = ["#777777"] + [palette[family] for family in palette]
    box = axes[2].boxplot(data, patch_artist=True, showfliers=False, widths=0.62, medianprops={"color": "white", "lw": 1.0})
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.82)
    axes[2].set_yscale("log")
    axes[2].set_xticklabels(["Reddit", "DeepSeek", "GPT-4-mini", "Luna", "Phi-4"], rotation=28, ha="right")
    axes[2].set_ylabel("Complete-message tokens (log scale)")
    axes[2].set_title("c  Message-length mismatch", loc="left", fontweight="bold")
    axes[2].grid(axis="y", color="#dddddd", linewidth=0.45)
    fig.tight_layout(w_pad=1.25)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix, kwargs in ((".svg", {}), (".pdf", {}), (".tiff", {"dpi": 600}), (".png", {"dpi": 300})):
        fig.savefig(output_stem.with_suffix(suffix), bbox_inches="tight", **kwargs)
    plt.close(fig)


def summarize(args: argparse.Namespace) -> None:
    rows_raw = read_csv(args.units)
    required = {"source", "unit_id", "model_family", "message_index", "token_count", "cosine_similarity"}
    if not rows_raw or not required.issubset(rows_raw[0]):
        raise RuntimeError(f"Unit table must contain columns: {sorted(required)}")
    rows: list[dict[str, object]] = []
    for row in rows_raw:
        index = int(row["message_index"])
        if index <= MAX_INDEX:
            rows.append(
                {
                    "source": row["source"],
                    "unit_id": row["unit_id"],
                    "model_family": canonical_model_family(row["model_family"]) if row["source"] == "LLM" else "",
                    "message_index": index,
                    "token_count": int(row["token_count"]),
                    "cosine_similarity": float(row["cosine_similarity"]),
                }
            )
    validate_source_counts(rows)
    validate_sequences(rows)
    for row in rows:
        if int(row["message_index"]) == 1 and abs(float(row["cosine_similarity"]) - 1.0) > 1e-7:
            raise RuntimeError("Index 1 must be visualization-only self-similarity")

    reddit_rows = [row for row in rows if row["source"] == "Reddit"]
    llm_rows = [row for row in rows if row["source"] == "LLM"]
    reddit_lengths = [int(row["token_count"]) for row in reddit_rows]
    llm_lengths: dict[str, list[int]] = defaultdict(list)
    for row in llm_rows:
        llm_lengths[str(row["model_family"])].append(int(row["token_count"]))

    reddit_values: dict[int, np.ndarray] = {}
    for index in range(1, MAX_INDEX + 1):
        selected = sorted(
            (row for row in reddit_rows if int(row["message_index"]) == index),
            key=lambda row: str(row["unit_id"]),
        )
        reddit_values[index] = np.asarray([float(row["cosine_similarity"]) for row in selected], dtype=np.float64)
    if len(reddit_values[1]) != 2_000 or len(reddit_values[MAX_INDEX]) != 1_008:
        raise RuntimeError("Reddit eligibility does not reproduce the frozen 2--92 horizon")

    run_ids = sorted({str(row["unit_id"]) for row in llm_rows})
    family_by_run = {
        run_id: next(str(row["model_family"]) for row in llm_rows if str(row["unit_id"]) == run_id)
        for run_id in run_ids
    }
    families = [family_by_run[run_id] for run_id in run_ids]
    similarity = np.empty((len(run_ids), MAX_INDEX), dtype=np.float64)
    for run_index, run_id in enumerate(run_ids):
        selected = sorted(
            (row for row in llm_rows if str(row["unit_id"]) == run_id),
            key=lambda row: int(row["message_index"]),
        )
        if [int(row["message_index"]) for row in selected] != list(range(1, MAX_INDEX + 1)):
            raise RuntimeError(f"Incomplete LLM trajectory: {run_id}")
        similarity[run_index] = [float(row["cosine_similarity"]) for row in selected]

    draws = int(args.bootstrap_resamples)
    reddit_curve: dict[int, dict[str, object]] = {}
    for index in range(1, MAX_INDEX + 1):
        values = reddit_values[index]
        if index == 1:
            low = high = ""
        else:
            low, high = percentile_ci(
                bootstrap_means(
                    values,
                    stable_seed(REDDIT_SEED_BASE, REDDIT_VERSION, "single_comment", index),
                    draws,
                )
            )
        reddit_curve[index] = {
            "eligible_threads": len(values),
            "mean_similarity": float(values.mean()),
            "ci95_low": low,
            "ci95_high": high,
        }

    family_array = np.asarray(families)
    family_rows: list[dict[str, object]] = []
    for family in sorted(set(families)):
        selected_runs = np.flatnonzero(family_array == family)
        for index in range(1, MAX_INDEX + 1):
            values = similarity[selected_runs, index - 1]
            if index == 1:
                low = high = ""
            else:
                low, high = percentile_ci(
                    run_weights(
                        len(values),
                        stable_seed(LLM_SEED_BASE, LLM_VERSION, "family", family, index),
                        draws,
                    )
                    @ values
                )
            family_rows.append(
                {
                    "model_family": family,
                    "message_index": index,
                    "comparison": f"U1_vs_U{index}",
                    "visualization_only": int(index == 1),
                    "eligible_runs": len(values),
                    "mean_similarity": float(values.mean()),
                    "ci95_low": low,
                    "ci95_high": high,
                    "reddit_mean": reddit_curve[index]["mean_similarity"],
                    "family_minus_reddit": float(values.mean()) - float(reddit_curve[index]["mean_similarity"]),
                    "bootstrap_resamples": "" if index == 1 else draws,
                }
            )

    curve_rows: list[dict[str, object]] = []
    for index in range(1, MAX_INDEX + 1):
        values = similarity[:, index - 1]
        reddit = reddit_curve[index]
        run_mean = float(values.mean())
        preferred_mean = family_point(values, families)
        if index == 1:
            run_low = run_high = family_low = family_high = ""
            run_diff_low = run_diff_high = family_diff_low = family_diff_high = ""
        else:
            run_draws = run_weights(
                len(run_ids), stable_seed(LLM_SEED_BASE, LLM_VERSION, "run", index), draws
            ) @ values
            family_draws = family_weights(
                families,
                stable_seed(LLM_SEED_BASE, LLM_VERSION, "family_equal", index),
                draws,
            ) @ values
            run_low, run_high = percentile_ci(run_draws)
            family_low, family_high = percentile_ci(family_draws)
            reddit_run = bootstrap_means(
                reddit_values[index],
                stable_seed(LLM_SEED_BASE, LLM_VERSION, "reddit_run", index),
                draws,
            )
            reddit_family = bootstrap_means(
                reddit_values[index],
                stable_seed(LLM_SEED_BASE, LLM_VERSION, "reddit_family", index),
                draws,
            )
            run_diff_low, run_diff_high = percentile_ci(run_draws - reddit_run)
            family_diff_low, family_diff_high = percentile_ci(family_draws - reddit_family)
        curve_rows.append(
            {
                "message_index": index,
                "comparison": f"Message_1_vs_Message_{index}",
                "visualization_only": int(index == 1),
                "reddit_eligible_threads": reddit["eligible_threads"],
                "reddit_mean": reddit["mean_similarity"],
                "reddit_ci95_low": reddit["ci95_low"],
                "reddit_ci95_high": reddit["ci95_high"],
                "llm_eligible_runs": len(run_ids),
                "llm_model_families": len(set(families)),
                "llm_run_equal_mean": run_mean,
                "llm_run_equal_ci95_low": run_low,
                "llm_run_equal_ci95_high": run_high,
                "llm_family_equal_mean": preferred_mean,
                "llm_family_equal_ci95_low": family_low,
                "llm_family_equal_ci95_high": family_high,
                "run_equal_llm_minus_reddit": run_mean - float(reddit["mean_similarity"]),
                "run_equal_difference_ci95_low": run_diff_low,
                "run_equal_difference_ci95_high": run_diff_high,
                "family_equal_llm_minus_reddit": preferred_mean - float(reddit["mean_similarity"]),
                "family_equal_difference_ci95_low": family_diff_low,
                "family_equal_difference_ci95_high": family_diff_high,
                "preferred_summary": "model-family-equal",
                "bootstrap_resamples": "" if index == 1 else draws,
            }
        )

    diagnostics = [
        {"domain": "Reddit", "model_family": "Reddit", "scope": "complete comments at analyzed indices 1-92", **distribution(reddit_lengths)},
        {"domain": "LLM", "model_family": "All LLM", "scope": "complete utterances U1-U92", **distribution([value for family in llm_lengths.values() for value in family])},
    ]
    diagnostics.extend(
        {"domain": "LLM", "model_family": family, "scope": "complete utterances U1-U92", **distribution(llm_lengths[family])}
        for family in sorted(llm_lengths)
    )

    output_dir = args.output_dir
    length_rows = length_sensitivity(
        rows, bootstrap_resamples=int(args.length_bootstrap_resamples)
    )
    write_csv(output_dir / "complete_message_curve.csv", curve_rows)
    write_csv(output_dir / "complete_message_family_curves.csv", family_rows)
    write_csv(output_dir / "complete_message_token_length_diagnostics.csv", diagnostics)
    write_csv(output_dir / "message_length_sensitivity.csv", length_rows)
    plot_figure(curve_rows, family_rows, reddit_lengths, llm_lengths, output_dir / "complete_message_reddit_vs_llm")


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    subparsers = root.add_subparsers(dest="command", required=True)
    compute_parser = subparsers.add_parser("compute", help="Embed complete messages and calculate anchor similarities")
    compute_parser.add_argument("--messages", type=Path, required=True)
    compute_parser.add_argument("--output", type=Path, required=True)
    compute_parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    compute_parser.set_defaults(func=compute)
    summary_parser = subparsers.add_parser("summarize", help="Aggregate unit similarities and create the summary figure")
    summary_parser.add_argument("--units", type=Path, required=True)
    summary_parser.add_argument("--output-dir", type=Path, required=True)
    summary_parser.add_argument("--bootstrap-resamples", type=int, default=BOOTSTRAP_RESAMPLES)
    summary_parser.add_argument(
        "--length-bootstrap-resamples",
        type=int,
        default=LENGTH_BOOTSTRAP_RESAMPLES,
        help="Trajectory-level bootstrap draws for the linear/log message-length sensitivity models.",
    )
    summary_parser.set_defaults(func=summarize)
    return root


def main() -> None:
    args = parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
