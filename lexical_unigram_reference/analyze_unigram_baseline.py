#!/usr/bin/env python3
"""Empirical-frequency IID unigram reference for lexical accumulation.

The analysis reads the frozen cohort in ``cohort_manifest.csv``. For each run,
it applies the same lexical preprocessing to the observed trajectory and to the
complete token stream used to estimate the empirical unigram probabilities.
It then evaluates the analytical IID expectation at each 10-round interval.

This module contains no plotting code and does not download or embed data.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import pathlib
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd


WINDOW_SIZE = 10
HOP = 10
EXPECTED_ROUNDS = list(range(1, 1001))
EXPECTED_INTERVALS = 100
EXPECTED_RUNS = 15
EARLY_INTERVALS = tuple(range(1, 21))
LATE_INTERVALS = tuple(range(81, 101))

FAMILY_ORDER = [
    "DeepSeek-V3",
    "GPT-4-mini",
    "Phi-4",
    "GPT-5.6 Terra",
    "Claude Sonnet 5",
]
EXPECTED_FAMILY_COUNTS = {family: 3 for family in FAMILY_ORDER}

TOKEN_RE = re.compile(r"[a-z]+")
STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "of",
    "to",
    "in",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "for",
    "on",
    "with",
    "as",
    "at",
    "by",
    "from",
    "that",
    "this",
    "it",
    "its",
    "into",
    "about",
    "than",
}

MESSAGE_PATTERNS = [
    re.compile(
        r"\[Round\s+(?P<round>\d+)\]\s*\((?P<role>[^)]+)\)\s*"
        r"(?P<agent>[A-Za-z0-9_\-]+)\s+said:\s*'(?P<text>.*?)'\s*(?:->|$)",
        re.DOTALL | re.IGNORECASE,
    ),
    re.compile(
        r'\[Round\s+(?P<round>\d+)\]\s*\((?P<role>[^)]+)\)\s*'
        r'(?P<agent>[A-Za-z0-9_\-]+)\s+said:\s*"(?P<text>.*?)"\s*(?:->|$)',
        re.DOTALL | re.IGNORECASE,
    ),
    re.compile(
        r"\[Round\s+(?P<round>\d+)\]\s*\((?P<role>[^)]+)\)\s*"
        r"(?P<agent>[A-Za-z0-9_\-]+)\s*:\s*['\"]?(?P<text>.*?)['\"]?\s*(?:->|$)",
        re.DOTALL | re.IGNORECASE,
    ),
]
ROUND_HEAD = re.compile(r"\[Round\s+(?P<round>\d+)\]", re.IGNORECASE)
SIMULATOR_ROUND_HEAD = re.compile(
    r"^\s*=+\s*Round\s+(?P<round>\d+)\s+order\s*:",
    re.IGNORECASE | re.MULTILINE,
)


@dataclass(frozen=True)
class ManifestEntry:
    model_family: str
    run_id: str
    filename: str


@dataclass
class RunData:
    family: str
    run_id: str
    source_file: pathlib.Path
    parser_mode: str
    parser_hits: int
    tokens_per_round: Dict[int, List[str]]
    windows: Dict[int, List[str]]
    token_counts: Counter[str]


def tokenise(text: str) -> list[str]:
    """Apply the paper's lexical unigram preprocessing."""
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    tokens = TOKEN_RE.findall(text)
    return [token for token in tokens if 2 <= len(token) <= 20 and token not in STOPWORDS]


def parse_log_fuzzy(path: pathlib.Path) -> tuple[Dict[int, Dict[str, List[str]]], str, int]:
    """Parse generated utterances using the historical transcript formats."""
    text = path.read_text(encoding="utf-8", errors="ignore")
    per_round: Dict[int, Dict[str, List[str]]] = {}
    hits = 0
    parser_mode = "none"

    for pattern_index, pattern in enumerate(MESSAGE_PATTERNS, start=1):
        for match in pattern.finditer(text):
            round_number = int(match["round"])
            tokens = tokenise(match["text"])
            if not tokens:
                continue
            per_round.setdefault(round_number, {"ALL": []})["ALL"].extend(tokens)
            hits += 1
        if hits:
            parser_mode = f"structured_pattern_{pattern_index}"
            break

    if not hits:
        parser_mode = "round_header_fallback"
        blocks = list(ROUND_HEAD.finditer(text))
        for index, block_match in enumerate(blocks):
            round_number = int(block_match["round"])
            start = block_match.end()
            end = blocks[index + 1].start() if index + 1 < len(blocks) else len(text)
            chunk = text[start:end]
            chunk = re.split(r"\n\s*->.*?$", chunk, flags=re.MULTILINE)[0]
            tokens = tokenise(chunk)
            if tokens:
                per_round.setdefault(round_number, {"ALL": []})["ALL"].extend(tokens)
                hits += 1

    return {key: per_round[key] for key in sorted(per_round)}, parser_mode, hits


def raw_round_ids(path: pathlib.Path) -> list[int]:
    """Read the simulator timeline, including rounds without lexical tokens."""
    text = path.read_text(encoding="utf-8", errors="ignore")
    round_ids = {int(match["round"]) for match in SIMULATOR_ROUND_HEAD.finditer(text)}
    round_ids.update(int(match["round"]) for match in ROUND_HEAD.finditer(text))
    return sorted(round_ids)


def rounds_to_windows(
    tokens_per_round: Dict[int, List[str]], window_size: int, hop: int
) -> Dict[int, List[str]]:
    """Construct non-overlapping windows from the ordered simulator rounds."""
    round_ids = sorted(tokens_per_round)
    windows: Dict[int, List[str]] = {}
    start_index = 0
    window_index = 1
    while start_index < len(round_ids):
        block = round_ids[start_index : start_index + window_size]
        if not block:
            break
        tokens: list[str] = []
        for round_number in block:
            tokens.extend(tokens_per_round[round_number])
        if tokens:
            windows[window_index] = tokens
        window_index += 1
        start_index += hop
    return windows


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: pathlib.Path) -> list[ManifestEntry]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"model_family", "run_id", "filename"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"Manifest must contain columns: {sorted(required)}")
        entries = [
            ManifestEntry(
                model_family=row["model_family"].strip(),
                run_id=row["run_id"].strip(),
                filename=row["filename"].strip(),
            )
            for row in reader
        ]

    if len(entries) != EXPECTED_RUNS:
        raise ValueError(f"Expected {EXPECTED_RUNS} manifest rows; found {len(entries)}")
    if len({entry.filename.casefold() for entry in entries}) != EXPECTED_RUNS:
        raise ValueError("Manifest filenames must be unique")
    if len({entry.run_id.casefold() for entry in entries}) != EXPECTED_RUNS:
        raise ValueError("Manifest run IDs must be unique")
    if any(pathlib.Path(entry.filename).name != entry.filename for entry in entries):
        raise ValueError("Manifest filenames must not contain directory components")

    counts = Counter(entry.model_family for entry in entries)
    if dict(counts) != EXPECTED_FAMILY_COUNTS:
        raise ValueError(
            f"Manifest family counts must be {EXPECTED_FAMILY_COUNTS}; found {dict(counts)}"
        )
    return entries


def discover_and_parse(
    input_dir: pathlib.Path, manifest_path: pathlib.Path
) -> tuple[list[RunData], pd.DataFrame]:
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Transcript directory not found: {input_dir}")
    entries = load_manifest(manifest_path)
    run_data: list[RunData] = []
    manifest_rows: list[dict] = []

    for entry in entries:
        source_file = input_dir / entry.filename
        if not source_file.is_file():
            raise FileNotFoundError(f"Manifest transcript not found: {entry.filename}")

        parsed, parser_mode, parser_hits = parse_log_fuzzy(source_file)
        round_ids = raw_round_ids(source_file)
        if round_ids != EXPECTED_ROUNDS:
            missing = sorted(set(EXPECTED_ROUNDS) - set(round_ids))
            extra = sorted(set(round_ids) - set(EXPECTED_ROUNDS))
            raise ValueError(
                f"{entry.filename}: expected rounds 1-1000; "
                f"missing={missing[:20]}, extra={extra[:20]}"
            )

        tokens_per_round = {
            round_number: list(parsed.get(round_number, {"ALL": []})["ALL"])
            for round_number in round_ids
        }
        windows = rounds_to_windows(tokens_per_round, WINDOW_SIZE, HOP)
        if len(windows) != EXPECTED_INTERVALS:
            raise ValueError(
                f"{entry.filename}: expected {EXPECTED_INTERVALS} nonempty intervals; "
                f"found {len(windows)}"
            )

        all_tokens = [
            token
            for round_number in round_ids
            for token in tokens_per_round[round_number]
        ]
        token_counts = Counter(all_tokens)
        if not token_counts:
            raise ValueError(f"{entry.filename}: no lexical tokens were parsed")

        zero_utterance_rounds = [round_number for round_number in round_ids if round_number not in parsed]
        manifest_rows.append(
            {
                "model_family": entry.model_family,
                "run_id": entry.run_id,
                "source_file": entry.filename,
                "source_sha256": sha256_file(source_file),
                "parser_mode": parser_mode,
                "parsed_message_hits": parser_hits,
                "number_of_rounds": len(round_ids),
                "zero_utterance_round_count": len(zero_utterance_rounds),
                "zero_utterance_round_ids": ";".join(map(str, zero_utterance_rounds)),
                "complete_10_round_intervals": len(windows),
                "lexical_tokens": len(all_tokens),
                "vocabulary_size": len(token_counts),
            }
        )
        run_data.append(
            RunData(
                family=entry.model_family,
                run_id=entry.run_id,
                source_file=source_file,
                parser_mode=parser_mode,
                parser_hits=parser_hits,
                tokens_per_round=tokens_per_round,
                windows=windows,
                token_counts=token_counts,
            )
        )

    return run_data, pd.DataFrame(manifest_rows)


def expected_cumulative_vocabulary(probabilities: np.ndarray, token_exposure: int) -> float:
    """Return sum_v [1 - (1 - p_v)^T]."""
    if token_exposure < 0:
        raise ValueError("Token exposure must be nonnegative")
    with np.errstate(divide="ignore", invalid="ignore"):
        log_q = np.log1p(-probabilities)
        discovery = -np.expm1(token_exposure * log_q)
    if token_exposure == 0:
        discovery = np.zeros_like(probabilities)
    return float(np.sum(discovery))


def expected_new_types(
    probabilities: np.ndarray, preceding_tokens: int, interval_tokens: int
) -> float:
    """Return sum_v (1-p_v)^m [1-(1-p_v)^n]."""
    if preceding_tokens < 0 or interval_tokens < 0:
        raise ValueError("Token exposures must be nonnegative")
    with np.errstate(divide="ignore", invalid="ignore"):
        log_q = np.log1p(-probabilities)
        survived_before = (
            np.ones_like(probabilities)
            if preceding_tokens == 0
            else np.exp(preceding_tokens * log_q)
        )
        discovered_current = -np.expm1(interval_tokens * log_q)
    return float(np.sum(survived_before * discovered_current))


def analyze_runs(run_data: Sequence[RunData]) -> tuple[pd.DataFrame, pd.DataFrame]:
    interval_rows: list[dict] = []
    run_rows: list[dict] = []

    for run in run_data:
        total_tokens = sum(run.token_counts.values())
        support_size = len(run.token_counts)
        probabilities = np.asarray(list(run.token_counts.values()), dtype=float) / total_tokens
        if not math.isclose(float(probabilities.sum()), 1.0, rel_tol=0, abs_tol=1e-12):
            raise AssertionError(f"Probabilities do not sum to one for {run.run_id}")

        seen: set[str] = set()
        cumulative_tokens = 0
        run_result_rows: list[dict] = []
        for interval_index in sorted(run.windows):
            tokens = run.windows[interval_index]
            interval_tokens = len(tokens)
            preceding_tokens = cumulative_tokens
            cumulative_tokens += interval_tokens
            novel_types = set(tokens) - seen
            seen.update(tokens)

            expected_cumulative = expected_cumulative_vocabulary(
                probabilities, cumulative_tokens
            )
            expected_new = expected_new_types(
                probabilities, preceding_tokens, interval_tokens
            )
            expected_difference = expected_cumulative - expected_cumulative_vocabulary(
                probabilities, preceding_tokens
            )
            if abs(expected_new - expected_difference) > 1e-8:
                raise AssertionError(
                    f"Analytical identity failed for {run.run_id}, interval {interval_index}"
                )

            observed_cumulative = len(seen)
            observed_new = len(novel_types)
            observed_rate = 100.0 * observed_new / interval_tokens
            expected_rate = 100.0 * expected_new / interval_tokens
            row = {
                "model_family": run.family,
                "run_id": run.run_id,
                "source_file": run.source_file.name,
                "interval_index": interval_index,
                "round_start": (interval_index - 1) * WINDOW_SIZE + 1,
                "round_end": interval_index * WINDOW_SIZE,
                "interval_lexical_tokens": interval_tokens,
                "cumulative_lexical_tokens_before": preceding_tokens,
                "cumulative_lexical_tokens": cumulative_tokens,
                "observed_cumulative_unique_unigrams": observed_cumulative,
                "unigram_expected_cumulative_unique_unigrams": expected_cumulative,
                "cumulative_observed_minus_unigram": observed_cumulative - expected_cumulative,
                "observed_new_unique_unigrams": observed_new,
                "unigram_expected_new_unique_unigrams": expected_new,
                "observed_new_types_per_100_tokens": observed_rate,
                "unigram_expected_new_types_per_100_tokens": expected_rate,
                "new_rate_observed_minus_unigram": observed_rate - expected_rate,
                "empirical_unigram_support_size": support_size,
            }
            interval_rows.append(row)
            run_result_rows.append(row)

        if cumulative_tokens != total_tokens or len(seen) != support_size:
            raise AssertionError(f"Final token/support consistency check failed for {run.run_id}")

        run_frame = pd.DataFrame(run_result_rows)
        run_summary = {
            "model_family": run.family,
            "run_id": run.run_id,
            "source_file": run.source_file.name,
            "lexical_tokens": total_tokens,
            "vocabulary_size": support_size,
        }
        segments = {
            "trajectory": tuple(range(1, 101)),
            "early_1_20": EARLY_INTERVALS,
            "late_81_100": LATE_INTERVALS,
        }
        for prefix, interval_ids in segments.items():
            part = run_frame[run_frame["interval_index"].isin(interval_ids)]
            denominator = part["interval_lexical_tokens"].sum()
            observed_segment_rate = 100.0 * part["observed_new_unique_unigrams"].sum() / denominator
            expected_segment_rate = (
                100.0 * part["unigram_expected_new_unique_unigrams"].sum() / denominator
            )
            run_summary[f"{prefix}_observed_rate_per_100"] = observed_segment_rate
            run_summary[f"{prefix}_unigram_rate_per_100"] = expected_segment_rate
            run_summary[f"{prefix}_observed_minus_unigram"] = (
                observed_segment_rate - expected_segment_rate
            )
        run_rows.append(run_summary)

    return pd.DataFrame(interval_rows), pd.DataFrame(run_rows)


def make_family_summary(interval_df: pd.DataFrame, run_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    checkpoints = {"round_10": 1, "round_250": 25, "round_500": 50, "round_750": 75, "round_1000": 100}
    for family in FAMILY_ORDER:
        family_intervals = interval_df[interval_df["model_family"] == family]
        family_runs = run_df[run_df["model_family"] == family]
        for label, interval_index in checkpoints.items():
            part = family_intervals[family_intervals["interval_index"] == interval_index]
            rows.append(
                {
                    "summary_type": "cumulative_vocabulary_checkpoint",
                    "model_family": family,
                    "n_runs": family_runs["run_id"].nunique(),
                    "checkpoint_or_segment": label,
                    "interval_index": interval_index,
                    "round_end": interval_index * WINDOW_SIZE,
                    "observed": part["observed_cumulative_unique_unigrams"].mean(),
                    "unigram_expected": part[
                        "unigram_expected_cumulative_unique_unigrams"
                    ].mean(),
                }
            )
        for segment, prefix in {
            "trajectory": "trajectory",
            "early_intervals_1_20": "early_1_20",
            "late_intervals_81_100": "late_81_100",
        }.items():
            rows.append(
                {
                    "summary_type": "new_word_production_rate",
                    "model_family": family,
                    "n_runs": family_runs["run_id"].nunique(),
                    "checkpoint_or_segment": segment,
                    "interval_index": "",
                    "round_end": "",
                    "observed": family_runs[f"{prefix}_observed_rate_per_100"].mean(),
                    "unigram_expected": family_runs[
                        f"{prefix}_unigram_rate_per_100"
                    ].mean(),
                }
            )
    summary = pd.DataFrame(rows)
    summary["observed_minus_unigram"] = summary["observed"] - summary["unigram_expected"]
    return summary


def make_family_balanced_trajectory(interval_df: pd.DataFrame) -> pd.DataFrame:
    value_columns = [
        "observed_cumulative_unique_unigrams",
        "unigram_expected_cumulative_unique_unigrams",
        "observed_new_types_per_100_tokens",
        "unigram_expected_new_types_per_100_tokens",
    ]
    family_means = (
        interval_df.groupby(["model_family", "interval_index", "round_start", "round_end"], as_index=False)[
            value_columns
        ]
        .mean()
    )
    return (
        family_means.groupby(["interval_index", "round_start", "round_end"], as_index=False)[
            value_columns
        ]
        .mean()
        .assign(n_model_families=len(FAMILY_ORDER))
    )


def make_overall_summary(interval_df: pd.DataFrame, run_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    checkpoints = {"round_10": 1, "round_250": 25, "round_500": 50, "round_750": 75, "round_1000": 100}
    for label, interval_index in checkpoints.items():
        part = interval_df[interval_df["interval_index"] == interval_index]
        family_means = part.groupby("model_family")[[
            "observed_cumulative_unique_unigrams",
            "unigram_expected_cumulative_unique_unigrams",
        ]].mean()
        rows.append(
            {
                "aggregation": "family_balanced",
                "summary_type": "cumulative_vocabulary_checkpoint",
                "checkpoint_or_segment": label,
                "interval_index": interval_index,
                "round_end": interval_index * WINDOW_SIZE,
                "n_runs": run_df["run_id"].nunique(),
                "n_model_families": run_df["model_family"].nunique(),
                "observed": family_means["observed_cumulative_unique_unigrams"].mean(),
                "unigram_expected": family_means[
                    "unigram_expected_cumulative_unique_unigrams"
                ].mean(),
            }
        )
    for segment, prefix in {
        "trajectory": "trajectory",
        "early_intervals_1_20": "early_1_20",
        "late_intervals_81_100": "late_81_100",
    }.items():
        family_means = run_df.groupby("model_family")[[
            f"{prefix}_observed_rate_per_100",
            f"{prefix}_unigram_rate_per_100",
        ]].mean()
        rows.append(
            {
                "aggregation": "family_balanced",
                "summary_type": "new_word_production_rate",
                "checkpoint_or_segment": segment,
                "interval_index": "",
                "round_end": "",
                "n_runs": run_df["run_id"].nunique(),
                "n_model_families": run_df["model_family"].nunique(),
                "observed": family_means[f"{prefix}_observed_rate_per_100"].mean(),
                "unigram_expected": family_means[f"{prefix}_unigram_rate_per_100"].mean(),
            }
        )
    summary = pd.DataFrame(rows)
    summary["observed_minus_unigram"] = summary["observed"] - summary["unigram_expected"]
    return summary


def parse_args() -> argparse.Namespace:
    default_manifest = pathlib.Path(__file__).resolve().with_name("cohort_manifest.csv")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=pathlib.Path,
        required=True,
        help="Directory containing the 15 transcript files in the frozen manifest.",
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        required=True,
        help="Directory in which analysis CSV files will be written.",
    )
    parser.add_argument(
        "--manifest",
        type=pathlib.Path,
        default=default_manifest,
        help="Cohort manifest CSV (default: cohort_manifest.csv beside this script).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    run_data, manifest_audit = discover_and_parse(args.input_dir, args.manifest)
    interval_df, run_df = analyze_runs(run_data)
    family_summary = make_family_summary(interval_df, run_df)
    family_balanced_trajectory = make_family_balanced_trajectory(interval_df)
    overall_summary = make_overall_summary(interval_df, run_df)

    manifest_audit.to_csv(args.output_dir / "run_manifest_audit.csv", index=False)
    interval_df.to_csv(args.output_dir / "run_interval_results.csv", index=False)
    run_df.to_csv(args.output_dir / "run_summary.csv", index=False)
    family_summary.to_csv(args.output_dir / "model_family_summary.csv", index=False)
    family_balanced_trajectory.to_csv(
        args.output_dir / "family_balanced_trajectory.csv", index=False
    )
    overall_summary.to_csv(args.output_dir / "overall_summary.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
