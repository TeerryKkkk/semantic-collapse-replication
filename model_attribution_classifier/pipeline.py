from __future__ import annotations

"""Portable preparation for primary independent-reference and optional LORO designs."""

import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .canonical_parser import ParseResult, RoundDocument, Utterance, parse_transcript, sha256_text


FAMILY_ORDER = [
    "GPT-4o-mini",
    "DeepSeek-V3",
    "Phi-4",
    "GPT-5.6 Terra",
    "Claude Sonnet",
]
CLASS_ORDER = sorted(FAMILY_ORDER)
FOLDS = {1: 3, 2: 2, 3: 1}
TRAIN_ROUND_END = 200
TEST_ROUND_END = 1000


@dataclass(frozen=True)
class RunSpec:
    model_family: str
    run_id: int
    transcript_path: Path


@dataclass(frozen=True)
class TrainingRecord:
    model_family: str
    run_id: int
    round_id: int
    sequence_index: int
    text: str
    sha256: str


@dataclass(frozen=True)
class TestRecord:
    model_family: str
    run_id: int
    round_id: int
    text: str
    sha256: str
    valid_utterance_count: int
    character_count: int
    whitespace_word_count: int


@dataclass
class FoldData:
    fold: int
    held_out_run_id: int
    training: list[TrainingRecord]
    test: list[TestRecord]
    sample_weights: np.ndarray


@dataclass(frozen=True)
class IndependentRunSpec:
    role: str
    model_family: str
    run_id: int
    transcript_path: Path


@dataclass
class IndependentReferenceData:
    training: list[TrainingRecord]
    test: list[TestRecord]
    sample_weights: np.ndarray
    reference_run_counts: dict[tuple[str, int], int]


def _parse_run_id(value: str) -> int:
    normalized = value.strip().lower()
    if normalized.startswith("run_"):
        normalized = normalized[4:]
    elif normalized.startswith("run "):
        normalized = normalized[4:]
    run_id = int(normalized)
    if run_id not in {1, 2, 3}:
        raise ValueError(f"run_id must be 1, 2, or 3; received {value!r}")
    return run_id


def load_run_manifest(path: Path) -> list[RunSpec]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"model_family", "run_id", "transcript_path"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"Manifest requires columns {sorted(required)}")
        specs = []
        for row in reader:
            family = row["model_family"].strip()
            if family not in FAMILY_ORDER:
                raise ValueError(f"Unsupported model_family {family!r}")
            run_id = _parse_run_id(row["run_id"])
            transcript = Path(row["transcript_path"].strip())
            if not transcript.is_absolute():
                transcript = (path.parent / transcript).resolve()
            if not transcript.is_file():
                raise FileNotFoundError(transcript)
            specs.append(RunSpec(family, run_id, transcript))
    identities = {(spec.model_family, spec.run_id) for spec in specs}
    expected = {(family, run_id) for family in FAMILY_ORDER for run_id in (1, 2, 3)}
    if len(specs) != 15 or identities != expected:
        missing = sorted(expected - identities)
        extra = sorted(identities - expected)
        raise ValueError(f"Manifest must contain exactly 15 unique family/run rows; missing={missing}, extra={extra}")
    return specs


def load_independent_manifest(path: Path) -> list[IndependentRunSpec]:
    """Load 15 reference and 15 test runs for the primary design."""
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"role", "model_family", "run_id", "transcript_path"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"Independent-reference manifest requires columns {sorted(required)}")
        specs: list[IndependentRunSpec] = []
        for row in reader:
            role = row["role"].strip().lower().replace("-", "_")
            if role in {"training", "train", "training_reference"}:
                role = "reference"
            if role not in {"reference", "test"}:
                raise ValueError(f"role must be reference or test; received {row['role']!r}")
            family = row["model_family"].strip()
            if family not in FAMILY_ORDER:
                raise ValueError(f"Unsupported model_family {family!r}")
            run_id = _parse_run_id(row["run_id"])
            transcript = Path(row["transcript_path"].strip())
            if not transcript.is_absolute():
                transcript = (path.parent / transcript).resolve()
            if not transcript.is_file():
                raise FileNotFoundError(transcript)
            specs.append(IndependentRunSpec(role, family, run_id, transcript))
    identities = {(spec.role, spec.model_family, spec.run_id) for spec in specs}
    expected = {(role, family, run_id) for role in ("reference", "test") for family in FAMILY_ORDER for run_id in (1, 2, 3)}
    if len(specs) != 30 or identities != expected:
        raise ValueError(f"Manifest must contain exactly 30 unique role/family/run rows; missing={sorted(expected-identities)}, extra={sorted(identities-expected)}")
    return specs


def parse_runs(specs: list[RunSpec]) -> dict[tuple[str, int], ParseResult]:
    runs: dict[tuple[str, int], ParseResult] = {}
    for spec in specs:
        parsed = parse_transcript(spec.transcript_path, spec.model_family, spec.run_id)
        parsed.raise_for_structural_errors()
        runs[(spec.model_family, spec.run_id)] = parsed
    return runs


def parse_independent_runs(specs: list[IndependentRunSpec]) -> tuple[dict[tuple[str, int], ParseResult], dict[tuple[str, int], ParseResult]]:
    by_role: dict[str, dict[tuple[str, int], ParseResult]] = {"reference": {}, "test": {}}
    file_hashes: dict[str, set[str]] = {"reference": set(), "test": set()}
    for spec in specs:
        parsed = parse_transcript(spec.transcript_path, spec.model_family, spec.run_id)
        parsed.raise_for_structural_errors()
        expected_horizon = TRAIN_ROUND_END if spec.role == "reference" else TEST_ROUND_END
        headers = set(parsed.header_rounds)
        expected_headers = set(range(1, expected_horizon + 1))
        if headers != expected_headers:
            raise ValueError(f"{spec.role} run {spec.model_family} {spec.run_id} must have headers 1-{expected_horizon}")
        if spec.role == "reference" and set(parsed.documents) != expected_headers:
            missing = sorted(expected_headers - set(parsed.documents))
            raise ValueError(f"Reference run {spec.model_family} {spec.run_id} has missing canonical documents: {missing}")
        by_role[spec.role][(spec.model_family, spec.run_id)] = parsed
        file_hashes[spec.role].add(parsed.file_sha256)
    overlap = file_hashes["reference"] & file_hashes["test"]
    if overlap:
        raise ValueError("Reference and test manifests contain byte-identical transcript files")
    return by_role["reference"], by_role["test"]


def build_fold(runs: dict[tuple[str, int], ParseResult], fold: int) -> FoldData:
    if fold not in FOLDS:
        raise ValueError(f"Unknown fold {fold}")
    held_out_run_id = FOLDS[fold]
    training: list[TrainingRecord] = []
    test: list[TestRecord] = []
    run_counts: dict[tuple[str, int], int] = {}

    for family in FAMILY_ORDER:
        for run_id in (1, 2, 3):
            parsed = runs[(family, run_id)]
            if run_id == held_out_run_id:
                for round_id, document in sorted(parsed.documents.items()):
                    if 1 <= round_id <= TEST_ROUND_END:
                        test.append(
                            TestRecord(
                                model_family=family,
                                run_id=run_id,
                                round_id=round_id,
                                text=document.text,
                                sha256=document.sha256,
                                valid_utterance_count=len(document.utterance_indices),
                                character_count=len(document.text),
                                whitespace_word_count=len(document.text.split()),
                            )
                        )
            else:
                items = [item for item in parsed.utterances if 1 <= item.round_id <= TRAIN_ROUND_END]
                if not items:
                    raise ValueError(f"No training utterances for {family} run {run_id}")
                run_counts[(family, run_id)] = len(items)
                training.extend(
                    TrainingRecord(
                        model_family=family,
                        run_id=run_id,
                        round_id=item.round_id,
                        sequence_index=item.sequence_index,
                        text=item.text,
                        sha256=item.sha256,
                    )
                    for item in items
                )

    if any(record.run_id == held_out_run_id for record in training):
        raise RuntimeError(f"Held-out run leaked into fold {fold} training")
    if len(run_counts) != 10:
        raise RuntimeError(f"Fold {fold} expected 10 training runs, found {len(run_counts)}")
    scale = len(training) / len(run_counts)
    weights = np.asarray(
        [scale / run_counts[(record.model_family, record.run_id)] for record in training],
        dtype=np.float64,
    )
    if not math.isclose(float(weights.mean()), 1.0, rel_tol=0, abs_tol=1e-12):
        raise RuntimeError(f"Fold {fold} sample weights do not have mean one")
    totals: dict[tuple[str, int], float] = {key: 0.0 for key in run_counts}
    for record, weight in zip(training, weights):
        totals[(record.model_family, record.run_id)] += float(weight)
    if max(totals.values()) - min(totals.values()) > 1e-9:
        raise RuntimeError(f"Fold {fold} run-aware weights are unequal")
    family_totals = {
        family: sum(value for (current_family, _run_id), value in totals.items() if current_family == family)
        for family in FAMILY_ORDER
    }
    if max(family_totals.values()) - min(family_totals.values()) > 1e-9:
        raise RuntimeError(f"Fold {fold} model-family weights are unequal")
    if set(record.sha256 for record in training) & set(record.sha256 for record in test):
        raise RuntimeError(f"Fold {fold} contains exact canonical-text train/test overlap")
    return FoldData(fold, held_out_run_id, training, test, weights)


def collect_unique_inputs(runs: dict[tuple[str, int], ParseResult]) -> dict[str, str]:
    texts: dict[str, str] = {}

    def add(text: str) -> None:
        digest = sha256_text(text)
        if digest in texts and texts[digest] != text:
            raise RuntimeError(f"SHA-256 collision for {digest}")
        texts[digest] = text

    for parsed in runs.values():
        for utterance in parsed.utterances:
            if 1 <= utterance.round_id <= TRAIN_ROUND_END:
                add(utterance.text)
        for round_id, document in parsed.documents.items():
            if 1 <= round_id <= TEST_ROUND_END:
                add(document.text)
    return texts


def build_independent_reference(
    reference_runs: dict[tuple[str, int], ParseResult],
    test_runs: dict[tuple[str, int], ParseResult],
) -> IndependentReferenceData:
    training: list[TrainingRecord] = []
    test: list[TestRecord] = []
    run_counts: dict[tuple[str, int], int] = {}
    expected = {(family, run_id) for family in FAMILY_ORDER for run_id in (1, 2, 3)}
    if set(reference_runs) != expected or set(test_runs) != expected:
        raise ValueError("Independent-reference design requires three reference and three test runs per family")
    for family in FAMILY_ORDER:
        for run_id in (1, 2, 3):
            reference = reference_runs[(family, run_id)]
            items = [item for item in reference.utterances if 1 <= item.round_id <= TRAIN_ROUND_END]
            if not items:
                raise ValueError(f"No reference utterances for {family} run {run_id}")
            run_counts[(family, run_id)] = len(items)
            training.extend(
                TrainingRecord(family, run_id, item.round_id, item.sequence_index, item.text, item.sha256)
                for item in items
            )
            long_run = test_runs[(family, run_id)]
            test.extend(
                TestRecord(
                    family, run_id, round_id, document.text, document.sha256,
                    len(document.utterance_indices), len(document.text), len(document.text.split()),
                )
                for round_id, document in sorted(long_run.documents.items())
                if 1 <= round_id <= TEST_ROUND_END
            )
    scale = len(training) / len(run_counts)
    weights = np.asarray([scale / run_counts[(record.model_family, record.run_id)] for record in training], dtype=np.float64)
    if not math.isclose(float(weights.mean()), 1.0, rel_tol=0, abs_tol=1e-12):
        raise RuntimeError("Independent-reference sample weights do not have mean one")
    totals = {key: 0.0 for key in run_counts}
    for record, weight in zip(training, weights):
        totals[(record.model_family, record.run_id)] += float(weight)
    if max(totals.values()) - min(totals.values()) > 1e-9:
        raise RuntimeError("Independent-reference run-aware weights are unequal")
    family_totals = {family: sum(value for (current, _run), value in totals.items() if current == family) for family in FAMILY_ORDER}
    if max(family_totals.values()) - min(family_totals.values()) > 1e-9:
        raise RuntimeError("Independent-reference family weights are unequal")
    return IndependentReferenceData(training, test, weights, run_counts)


def collect_independent_inputs(data: IndependentReferenceData) -> dict[str, str]:
    texts: dict[str, str] = {}
    for record in [*data.training, *data.test]:
        existing = texts.setdefault(record.sha256, record.text)
        if existing != record.text:
            raise RuntimeError(f"SHA-256 collision for {record.sha256}")
    return texts
