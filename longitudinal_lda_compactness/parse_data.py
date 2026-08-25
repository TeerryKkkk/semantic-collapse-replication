"""Authoritative parser and baseline-run selection for simulation logs."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from config import EXPECTED_INTERVALS, EXPECTED_ROUNDS, FAMILY_ORDER, INTERVAL_SIZE


ROUND_ORDER_RE = re.compile(r"^=+\s*Round\s+(\d+)\s+order", re.IGNORECASE)
ROUND_BRACKET_RE = re.compile(r"^\[Round\s+(\d+)\]", re.IGNORECASE)
SAID_LINE_RE = re.compile(
    r"^\[Round\s+(\d+)\]\s*(?:\(([^)]*)\)\s*)?(.*?)\s+said:\s*(.*)$",
    re.IGNORECASE,
)
INFRASTRUCTURE_RE = re.compile(
    r"^\s*(?:progress|retry(?:ing|\s+notice)?|error|exception|traceback|"
    r"system\s+message|referee|routing|router)\s*[:=]",
    re.IGNORECASE,
)
NON_AGENT_MARKERS = (
    "referee",
    "system",
    "routing",
    "router",
    "classifier",
    "classification",
    "summary",
    "metadata",
    "moderator",
    "judge",
    "evaluator",
)
WORD_RE = re.compile(r"(?u)\b[a-zA-Z][a-zA-Z]+\b")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def is_non_agent(tag: str, speaker: str) -> bool:
    probe = f"{tag} {speaker}".lower()
    return any(marker in probe for marker in NON_AGENT_MARKERS)


def parse_log(path: Path) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Extract multiline natural-language ``said:`` payloads from one log."""
    text = path.read_text(encoding="utf-8", errors="strict")
    messages: list[dict[str, object]] = []
    round_headers: list[int] = []
    rounds_seen_in_records: set[int] = set()
    tag_counts: Counter[str] = Counter()
    speakers: set[str] = set()
    excluded_nonagent = 0
    metadata_lines = 0
    structural_lines = 0
    current: dict[str, object] | None = None

    def finalize() -> None:
        nonlocal current
        if current is None:
            return
        chunks = list(current["chunks"])
        while chunks and not str(chunks[-1]).strip():
            chunks.pop()
        if chunks:
            last = str(chunks[-1]).rstrip()
            if last.endswith("'"):
                chunks[-1] = last[:-1]
        content = normalize_text(" ".join(str(chunk) for chunk in chunks))
        if content:
            messages.append(
                {
                    "round": int(current["round"]),
                    "tag": str(current["tag"]),
                    "speaker": str(current["speaker"]),
                    "text": content,
                    "utterance_index": len(messages) + 1,
                }
            )
        current = None

    for raw_line in text.splitlines():
        line = raw_line.rstrip("\r\n")
        order_match = ROUND_ORDER_RE.match(line)
        if order_match:
            finalize()
            round_headers.append(int(order_match.group(1)))
            structural_lines += 1
            continue

        said_match = SAID_LINE_RE.match(line)
        if said_match:
            finalize()
            round_number = int(said_match.group(1))
            tag = (said_match.group(2) or "").strip().upper()
            speaker = normalize_text(said_match.group(3) or "")
            content = (said_match.group(4) or "").strip()
            if content.startswith("'"):
                content = content[1:]
            rounds_seen_in_records.add(round_number)
            tag_counts[tag] += 1
            speakers.add(speaker)
            if is_non_agent(tag, speaker):
                excluded_nonagent += 1
                current = None
            else:
                current = {
                    "round": round_number,
                    "tag": tag,
                    "speaker": speaker,
                    "chunks": [content] if content else [],
                }
            continue

        stripped = line.lstrip()
        if stripped.startswith("->"):
            finalize()
            metadata_lines += 1
            continue
        if ROUND_BRACKET_RE.match(line) or stripped.startswith("====="):
            finalize()
            structural_lines += 1
            continue
        if INFRASTRUCTURE_RE.match(line) or stripped.startswith("Group established by"):
            finalize()
            structural_lines += 1
            continue
        if current is not None:
            current["chunks"].append(stripped)
    finalize()

    rounds = set(round_headers) | rounds_seen_in_records
    audit = {
        "round_headers": round_headers,
        "rounds": sorted(rounds),
        "tag_counts": dict(sorted(tag_counts.items())),
        "speakers": sorted(speakers),
        "excluded_nonagent_records": excluded_nonagent,
        "metadata_lines_excluded": metadata_lines,
        "structural_lines_excluded": structural_lines,
        "raw_file_bytes": path.stat().st_size,
        "raw_line_count": len(text.splitlines()),
    }
    return messages, audit


def classify_source(path: Path) -> dict[str, object]:
    """Map an exact filename pattern to model family, version, and condition."""
    specifications = (
        (r"^3_gpt_1000_v(\d+)\.txt$", "GPT-4o-mini", "standard multi-agent baseline"),
        (r"^3_deepseek_1000_v(\d+)\.txt$", "DeepSeek-V3", "standard multi-agent baseline"),
        (r"^3_phi-4_1000_v(\d+)\.txt$", "Phi-4", "standard multi-agent baseline"),
        (r"^gpt5\.6_1000_v(\d+)\.txt$", "GPT-5.6 Terra", "standard multi-agent baseline"),
        (r"^sonnet_1000_v(\d+)\.txt$", "Claude Sonnet", "standard multi-agent baseline"),
        (r"^gpt5\.6luna_single_agent_v(\d+)\.txt$", "GPT-5.6 Luna", "single-agent condition"),
    )
    for pattern, family, condition in specifications:
        match = re.fullmatch(pattern, path.name, flags=re.IGNORECASE)
        if match:
            version_number = int(match.group(1))
            return {
                "model_family": family,
                "version_number": version_number,
                "version": f"V{version_number}",
                "run_id": path.stem,
                "condition": condition,
            }
    return {
        "model_family": "Unrecognized",
        "version_number": None,
        "version": "",
        "run_id": path.stem,
        "condition": "unrecognized filename/condition",
    }


def inclusion_decision(specification: dict[str, object], rounds: list[int]) -> tuple[bool, str]:
    if specification["condition"] != "standard multi-agent baseline":
        return False, str(specification["condition"])
    if specification["version_number"] not in (1, 2, 3):
        return False, "version outside exact V1-V3 rule"
    if rounds != list(range(1, EXPECTED_ROUNDS + 1)):
        return False, "incomplete/noncontiguous 1000-round sequence"
    return True, "included: standard baseline and exact V1-V3"


def build_interval_documents(utterances: pd.DataFrame) -> pd.DataFrame:
    """Concatenate utterances into chronological non-overlapping 10-round documents."""
    document_rows: list[dict[str, object]] = []
    run_columns = ["run_id", "model_family", "version", "version_number", "source_file"]
    for run_values, run_frame in utterances.groupby(run_columns, sort=False):
        run_meta = dict(zip(run_columns, run_values))
        by_round = {
            round_number: group.sort_values("utterance_index")
            for round_number, group in run_frame.groupby("round", sort=False)
        }
        for interval in range(1, EXPECTED_INTERVALS + 1):
            round_start = (interval - 1) * INTERVAL_SIZE + 1
            round_end = interval * INTERVAL_SIZE
            interval_frames = [
                by_round[round_number]
                for round_number in range(round_start, round_end + 1)
                if round_number in by_round
            ]
            interval_messages = (
                pd.concat(interval_frames, ignore_index=True)
                if interval_frames
                else pd.DataFrame(columns=utterances.columns)
            )
            document = normalize_text("\n".join(interval_messages["text"].astype(str)))
            if not document:
                raise RuntimeError(f"Empty interval document: {run_meta['run_id']}, interval {interval}")
            document_rows.append(
                {
                    **run_meta,
                    "interval": interval,
                    "round_start": round_start,
                    "round_end": round_end,
                    "round_count": INTERVAL_SIZE,
                    "utterance_count": len(interval_messages),
                    "raw_lexical_word_count": len(WORD_RE.findall(document)),
                    "raw_character_count": len(document),
                    "document_text": document,
                }
            )
    documents = pd.DataFrame(document_rows)
    family_rank = {family: index for index, family in enumerate(FAMILY_ORDER)}
    documents["family_order"] = documents["model_family"].map(family_rank)
    return documents.sort_values(["family_order", "version_number", "interval"]).drop(columns="family_order")


def load_baseline_corpus(
    raw_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, set[str]]:
    """Audit all logs and return the included utterances and interval documents."""
    paths = sorted(raw_dir.glob("*.txt"), key=lambda path: path.name.lower())
    if not paths:
        raise FileNotFoundError(f"No .txt files found in {raw_dir}")

    manifest_rows: list[dict[str, object]] = []
    utterance_rows: list[dict[str, object]] = []
    included_versions: defaultdict[str, set[int]] = defaultdict(set)
    structural_speakers: set[str] = set()

    for path in paths:
        specification = classify_source(path)
        messages, audit = parse_log(path)
        rounds = list(audit["rounds"])
        included, reason = inclusion_decision(specification, rounds)
        header_counts = Counter(audit["round_headers"])
        duplicate_headers = sorted(number for number, count in header_counts.items() if count > 1)
        message_rounds = {int(message["round"]) for message in messages}
        zero_content_rounds = sorted(set(rounds) - message_rounds)
        manifest_rows.append(
            {
                **specification,
                "source_file": path.name,
                "included": included,
                "status": "included" if included else "excluded",
                "reason": reason,
                "rounds_detected": len(rounds),
                "round_headers_detected": len(audit["round_headers"]),
                "duplicate_round_headers": ";".join(map(str, duplicate_headers)),
                "rounds_with_no_conversational_utterance": ";".join(map(str, zero_content_rounds)),
                "n_rounds_with_no_conversational_utterance": len(zero_content_rounds),
                "conversational_utterances_detected": len(messages),
                "excluded_nonagent_records": audit["excluded_nonagent_records"],
                "first_round": min(rounds) if rounds else np.nan,
                "last_round": max(rounds) if rounds else np.nan,
                "resulting_10_round_intervals": EXPECTED_INTERVALS if included else 0,
                "raw_lexical_word_count": sum(len(WORD_RE.findall(str(row["text"]))) for row in messages),
                "raw_file_bytes": audit["raw_file_bytes"],
                "raw_line_count": audit["raw_line_count"],
                "metadata_lines_excluded": audit["metadata_lines_excluded"],
                "structural_lines_excluded": audit["structural_lines_excluded"],
                "message_tag_counts_json": json.dumps(audit["tag_counts"], sort_keys=True),
                "sha256": sha256_file(path),
            }
        )
        if not included:
            continue
        family = str(specification["model_family"])
        version_number = int(specification["version_number"])
        included_versions[family].add(version_number)
        structural_speakers.update(str(speaker).lower() for speaker in audit["speakers"] if speaker)
        for message in messages:
            utterance_index = int(message["utterance_index"])
            utterance_rows.append(
                {
                    "model_family": family,
                    "run_id": str(specification["run_id"]),
                    "version": str(specification["version"]),
                    "version_number": version_number,
                    "source_file": path.name,
                    "round": int(message["round"]),
                    "utterance_index": utterance_index,
                    "utterance_id": f"{specification['run_id']}__u{utterance_index:05d}",
                    "tag": str(message["tag"]),
                    "speaker": str(message["speaker"]),
                    "text": str(message["text"]),
                }
            )

    for family in FAMILY_ORDER:
        if included_versions[family] != {1, 2, 3}:
            raise RuntimeError(
                f"Missing required V1-V3 member(s) for {family}: found {sorted(included_versions[family])}"
            )
    manifest = pd.DataFrame(manifest_rows).sort_values(
        ["model_family", "version_number"], na_position="last"
    )
    # Preserve the finalized manifest traversal order used by the compactness
    # analysis so fixed-seed sampling maps to the same utterance pairs.
    utterances = (
        pd.DataFrame(utterance_rows)
        .sort_values(["model_family", "version_number", "utterance_index"])
        .reset_index(drop=True)
    )
    if utterances["utterance_id"].duplicated().any():
        raise RuntimeError("Utterance identifiers are not unique")
    documents = build_interval_documents(utterances)
    expected_documents = len(FAMILY_ORDER) * 3 * EXPECTED_INTERVALS
    if len(documents) != expected_documents:
        raise RuntimeError(f"Expected {expected_documents} interval documents, found {len(documents)}")
    return manifest, utterances, documents, structural_speakers
