from __future__ import annotations

"""Canonical transcript parser for model-attribution data."""

import ast
import hashlib
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


ROUND_DOCUMENT_SEPARATOR = "\n\n"
KNOWN_AGENT_CHANNELS = {
    "MAIN",
    "GROUP_INTERACTION",
    "REACTION",
    "INVITATION_REPLY",
}

ROUND_HEADER_RE = re.compile(r"^=====\s+Round\s+(\d+)\s+order:\s*(\[.*\])\s+=====\s*$")
ROUND_HEADER_PREFIX_RE = re.compile(r"^=====\s+Round\s+(\d+)\s+order:")
SAID_RE = re.compile(r"^\[Round\s+(\d+)\]\s+\(([^)]+)\)\s+(\S+)\s+said:\s?(.*)$")
ROUND_BRACKET_RE = re.compile(r"^\[Round\s+(\d+)\]")
REFEREE_SYSTEM_RE = re.compile(
    r"^(?:\[?(?:REFEREE|SYSTEM|META(?:DATA)?|BOOKKEEPING|INSTRUMENTATION)\]?\s*:|"
    r"\[Round\s+\d+\]\s+\((?:REFEREE|SYSTEM|META(?:DATA)?|BOOKKEEPING|INSTRUMENTATION)[^)]*\))",
    re.IGNORECASE,
)
FORBIDDEN_IN_TEXT = [
    re.compile(r"(?m)^=====\s+Round\s+\d+\s+order:"),
    re.compile(r"(?m)^Group established by\s+"),
    re.compile(r"(?m)^#\s*=====\s*CONTINUATION START\s*=====$"),
    re.compile(r"(?m)^\s*->"),
    REFEREE_SYSTEM_RE,
]


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="strict")).hexdigest()


def normalize_utterance_parts(parts: list[str]) -> str:
    normalized: list[str] = []
    for part in parts:
        value = part.strip()
        if not value:
            if normalized and normalized[-1] != "":
                normalized.append("")
            continue
        normalized.append(value)
    while normalized and normalized[-1] == "":
        normalized.pop()
    while normalized and normalized[0] == "":
        normalized.pop(0)
    return "\n".join(normalized)


@dataclass(frozen=True)
class Utterance:
    model_family: str
    run_id: int
    round_id: int
    channel: str
    speaker: str
    sequence_index: int
    order_in_round: int
    start_line: int
    end_line: int
    continuation_line_count: int
    text: str

    @property
    def sha256(self) -> str:
        return sha256_text(self.text)


@dataclass(frozen=True)
class RoundDocument:
    model_family: str
    run_id: int
    round_id: int
    utterance_indices: tuple[int, ...]
    text: str

    @property
    def sha256(self) -> str:
        return sha256_text(self.text)


@dataclass
class ParserDiagnostics:
    decode_replacement_count: int = 0
    duplicate_round_ids: list[int] = field(default_factory=list)
    invalid_round_header_lines: list[int] = field(default_factory=list)
    malformed_said_lines: list[int] = field(default_factory=list)
    said_rounds_without_header: list[int] = field(default_factory=list)
    unknown_channels: Counter[str] = field(default_factory=Counter)
    unknown_speakers: Counter[str] = field(default_factory=Counter)
    excluded_line_counts: Counter[str] = field(default_factory=Counter)
    invariant_failures: list[str] = field(default_factory=list)

    def structural_errors(self) -> list[str]:
        errors: list[str] = []
        if self.duplicate_round_ids:
            errors.append(f"duplicate_round_ids={self.duplicate_round_ids}")
        if self.invalid_round_header_lines:
            errors.append(f"invalid_round_header_lines={self.invalid_round_header_lines}")
        if self.malformed_said_lines:
            errors.append(f"malformed_said_lines={self.malformed_said_lines}")
        if self.said_rounds_without_header:
            errors.append(f"said_rounds_without_header={self.said_rounds_without_header}")
        if self.unknown_channels:
            errors.append(f"unknown_channels={dict(self.unknown_channels)}")
        if self.unknown_speakers:
            errors.append(f"unknown_speakers={dict(self.unknown_speakers)}")
        errors.extend(self.invariant_failures)
        return errors


@dataclass
class ParseResult:
    path: Path
    model_family: str
    run_id: int
    file_sha256: str
    header_rounds: list[int]
    round_orders: dict[int, list[str]]
    utterances: list[Utterance]
    documents: dict[int, RoundDocument]
    diagnostics: ParserDiagnostics

    def raise_for_structural_errors(self) -> None:
        errors = self.diagnostics.structural_errors()
        if errors:
            raise ValueError(f"Structural parser errors in {self.path}: " + "; ".join(errors))


def _header_inventory(lines: list[str]) -> tuple[list[int], dict[int, list[str]], list[int], list[int]]:
    header_rounds: list[int] = []
    round_orders: dict[int, list[str]] = {}
    invalid_lines: list[int] = []
    for line_number, line in enumerate(lines, 1):
        match = ROUND_HEADER_RE.match(line)
        if match:
            round_id = int(match.group(1))
            header_rounds.append(round_id)
            try:
                order = ast.literal_eval(match.group(2))
                if not isinstance(order, list) or not all(isinstance(value, str) for value in order):
                    raise ValueError
                if round_id not in round_orders:
                    round_orders[round_id] = order
            except Exception:
                invalid_lines.append(line_number)
        elif ROUND_HEADER_PREFIX_RE.match(line):
            invalid_lines.append(line_number)
    counts = Counter(header_rounds)
    duplicates = sorted(round_id for round_id, count in counts.items() if count > 1)
    return header_rounds, round_orders, duplicates, invalid_lines


def parse_transcript(path: Path, model_family: str, run_id: int) -> ParseResult:
    raw_bytes = path.read_bytes()
    raw = raw_bytes.decode("utf-8", errors="replace")
    lines = raw.splitlines()
    header_rounds, round_orders, duplicates, invalid_headers = _header_inventory(lines)
    diagnostics = ParserDiagnostics(
        decode_replacement_count=raw.count("\ufffd"),
        duplicate_round_ids=duplicates,
        invalid_round_header_lines=invalid_headers,
    )
    utterances: list[Utterance] = []
    per_round_order: Counter[int] = Counter()
    active: Optional[dict] = None

    def flush(end_line: int) -> None:
        nonlocal active
        if active is None:
            return
        text = normalize_utterance_parts(active["parts"])
        has_linguistic_content = bool(text) and bool(text.strip("'\" \t\r\n"))
        if text and not has_linguistic_content:
            diagnostics.excluded_line_counts["empty_or_quote_only_said"] += 1
        if has_linguistic_content:
            round_id = active["round_id"]
            per_round_order[round_id] += 1
            utterances.append(
                Utterance(
                    model_family=model_family,
                    run_id=run_id,
                    round_id=round_id,
                    channel=active["channel"],
                    speaker=active["speaker"],
                    sequence_index=len(utterances),
                    order_in_round=per_round_order[round_id],
                    start_line=active["start_line"],
                    end_line=max(active["start_line"], end_line),
                    continuation_line_count=active["continuation_line_count"],
                    text=text,
                )
            )
        active = None

    for line_number, line in enumerate(lines, 1):
        stripped = line.strip()
        header_match = ROUND_HEADER_RE.match(line)
        if header_match:
            flush(line_number - 1)
            diagnostics.excluded_line_counts["round_order_header"] += 1
            continue
        if ROUND_HEADER_PREFIX_RE.match(line):
            flush(line_number - 1)
            diagnostics.excluded_line_counts["malformed_round_order_header"] += 1
            continue

        said_match = SAID_RE.match(line)
        if said_match:
            flush(line_number - 1)
            round_id = int(said_match.group(1))
            channel = said_match.group(2).strip()
            speaker = said_match.group(3).strip()
            payload = said_match.group(4).strip()
            expected_speakers = round_orders.get(round_id)
            invalid = False
            if channel not in KNOWN_AGENT_CHANNELS:
                diagnostics.unknown_channels[channel] += 1
                diagnostics.excluded_line_counts["unknown_channel_said"] += 1
                invalid = True
            if expected_speakers is None:
                diagnostics.said_rounds_without_header.append(round_id)
                diagnostics.excluded_line_counts["said_without_round_header"] += 1
                invalid = True
            elif speaker not in expected_speakers:
                diagnostics.unknown_speakers[speaker] += 1
                diagnostics.excluded_line_counts["unknown_speaker_said"] += 1
                invalid = True
            if not invalid:
                active = {
                    "round_id": round_id,
                    "channel": channel,
                    "speaker": speaker,
                    "start_line": line_number,
                    "parts": [payload] if payload else [],
                    "continuation_line_count": 0,
                }
            continue

        if "said:" in line and ROUND_BRACKET_RE.match(line):
            flush(line_number - 1)
            diagnostics.malformed_said_lines.append(line_number)
            diagnostics.excluded_line_counts["malformed_said_line"] += 1
            continue

        left = line.lstrip()
        if left.startswith("->"):
            flush(line_number - 1)
            diagnostics.excluded_line_counts["routing_arrow"] += 1
        elif left.startswith("Group established by "):
            flush(line_number - 1)
            diagnostics.excluded_line_counts["group_established_bookkeeping"] += 1
        elif left == "# ===== CONTINUATION START =====":
            flush(line_number - 1)
            diagnostics.excluded_line_counts["continuation_start_marker"] += 1
        elif left.startswith("# ====="):
            flush(line_number - 1)
            diagnostics.excluded_line_counts["structural_comment"] += 1
        elif left.startswith("====="):
            flush(line_number - 1)
            diagnostics.excluded_line_counts["structural_delimiter"] += 1
        elif REFEREE_SYSTEM_RE.match(left):
            flush(line_number - 1)
            diagnostics.excluded_line_counts["referee_system_metadata"] += 1
        elif ROUND_BRACKET_RE.match(line):
            flush(line_number - 1)
            diagnostics.excluded_line_counts["round_bracket_non_said"] += 1
        elif not stripped:
            if active is not None:
                active["parts"].append("")
        elif active is not None:
            active["parts"].append(stripped)
            active["continuation_line_count"] += 1
        else:
            diagnostics.excluded_line_counts["inactive_nonagent_line"] += 1
    flush(len(lines))

    by_round: dict[int, list[Utterance]] = defaultdict(list)
    for utterance in utterances:
        by_round[utterance.round_id].append(utterance)
    documents: dict[int, RoundDocument] = {}
    for round_id, values in sorted(by_round.items()):
        text = ROUND_DOCUMENT_SEPARATOR.join(value.text for value in values)
        if text:
            documents[round_id] = RoundDocument(
                model_family=model_family,
                run_id=run_id,
                round_id=round_id,
                utterance_indices=tuple(value.sequence_index for value in values),
                text=text,
            )

    for utterance in utterances:
        if utterance.round_id not in round_orders:
            diagnostics.invariant_failures.append(f"utterance_without_round_header:{utterance.sequence_index}")
        for pattern in FORBIDDEN_IN_TEXT:
            if pattern.search(utterance.text):
                diagnostics.invariant_failures.append(f"forbidden_marker:{utterance.sequence_index}:{pattern.pattern}")
    for round_id, document in documents.items():
        expected = by_round[round_id]
        if document.utterance_indices != tuple(value.sequence_index for value in expected):
            diagnostics.invariant_failures.append(f"document_order_mismatch:{round_id}")
        if document.text != ROUND_DOCUMENT_SEPARATOR.join(value.text for value in expected):
            diagnostics.invariant_failures.append(f"document_rebuild_mismatch:{round_id}")
        if any(value.round_id != round_id for value in expected):
            diagnostics.invariant_failures.append(f"cross_round_document:{round_id}")

    return ParseResult(
        path=path,
        model_family=model_family,
        run_id=run_id,
        file_sha256=hashlib.sha256(raw_bytes).hexdigest(),
        header_rounds=header_rounds,
        round_orders=round_orders,
        utterances=utterances,
        documents=documents,
        diagnostics=diagnostics,
    )
