"""Recent conversational-history policy used by HC-AD.

The bank is agent-specific and contains at most the six newest unique eligible
utterances from the previous three rounds. Own main outputs, own reactions, and
incoming utterances are eligible. Current-round, direct-trigger, system,
judge/referee, control, RAG-only, rejected-candidate, and cross-run text are not.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Iterable, Mapping, Sequence


AVOIDANCE_BANK_SIZE = 6
MEMORY_ROUNDS = 3
ELIGIBLE_RECORD_TYPES = frozenset(
    {"main_output", "reaction_output", "incoming_msg"}
)

_CONTROL_PREFIXES = (
    "(ref note)",
    "(group step1)",
    "(group invitation)",
    "(group upgrade)",
    "(fallback)",
    "system:",
    "referee:",
    "judge:",
)
_REFEREE_KEYS = frozenset(
    {
        "action_name",
        "is_interaction",
        "valence",
        "description",
        "group_invitation",
        "agree_to_group",
    }
)
_TRANSCRIPT_WRAPPER = re.compile(
    r"""^\s*
        (?:\[Round\s+\d+\]\s*)?
        (?:\([^)]+\)\s*)?
        (?P<name>[A-Za-z][A-Za-z0-9_-]{0,63})
        \s+said:\s*
        (?P<body>.+?)
        \s*$""",
    re.IGNORECASE | re.VERBOSE | re.DOTALL,
)
_FIVE_LETTER_AGENT_PREFIX = re.compile(
    r"^\s*[A-Z]{5}\s*:\s*",
    re.DOTALL,
)


@dataclass(frozen=True)
class AvoidanceBank:
    texts: tuple[str, ...]

    def __len__(self) -> int:
        return len(self.texts)


def _strip_balanced_quotes(text: str) -> str:
    pairs = {
        ('"', '"'),
        ("'", "'"),
        ("\u201c", "\u201d"),
        ("\u2018", "\u2019"),
    }
    if len(text) >= 2 and (text[0], text[-1]) in pairs:
        return text[1:-1].strip()
    return text


def normalize_response_text(
    text: str,
    *,
    agent_names: Sequence[str] = (),
) -> str:
    """Remove known speaker wrappers while preserving response content."""

    normalized = (text or "").strip()
    if not normalized:
        return ""

    transcript_match = _TRANSCRIPT_WRAPPER.match(normalized)
    if transcript_match:
        normalized = transcript_match.group("body").strip()
        normalized = _strip_balanced_quotes(normalized)

    for name in sorted(
        {name.strip() for name in agent_names if name and name.strip()},
        key=len,
        reverse=True,
    ):
        prefix = re.compile(
            rf"^\s*{re.escape(name)}\s*:\s*",
            re.DOTALL,
        )
        updated = prefix.sub("", normalized, count=1)
        if updated != normalized:
            normalized = updated.strip()
            break
    else:
        normalized = _FIVE_LETTER_AGENT_PREFIX.sub(
            "",
            normalized,
            count=1,
        ).strip()

    return _strip_balanced_quotes(normalized).strip()


def _looks_like_referee_or_control(text: str) -> bool:
    low = text.lstrip().lower()
    if any(low.startswith(prefix) for prefix in _CONTROL_PREFIXES):
        return True

    if text.lstrip().startswith("{") and text.rstrip().endswith("}"):
        try:
            payload = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            return False
        if isinstance(payload, dict) and len(_REFEREE_KEYS & payload.keys()) >= 2:
            return True
    return False


def load_jsonl_records(path: str | Path) -> list[dict]:
    log_path = Path(path)
    if not log_path.exists():
        return []

    records: list[dict] = []
    with log_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                records.append(record)
    return records


def build_avoidance_bank(
    records: Iterable[Mapping[str, object]],
    *,
    current_round: int,
    memory_rounds: int = MEMORY_ROUNDS,
    bank_size: int = AVOIDANCE_BANK_SIZE,
    current_trigger: str | None = None,
    agent_names: Sequence[str] = (),
) -> AvoidanceBank:
    """Return newest eligible unique prior-round text in chronological order."""

    if bank_size <= 0 or memory_rounds <= 0 or current_round <= 1:
        return AvoidanceBank(())

    min_round = max(1, int(current_round) - int(memory_rounds))
    max_round = int(current_round) - 1
    excluded = normalize_response_text(
        current_trigger or "",
        agent_names=agent_names,
    )

    newest_first: list[str] = []
    seen: set[str] = set()
    for record in reversed(list(records)):
        record_type = str(record.get("type") or "").strip().lower()
        if record_type not in ELIGIBLE_RECORD_TYPES:
            continue
        try:
            round_number = int(record.get("round", 0))
        except (TypeError, ValueError):
            continue
        if not (min_round <= round_number <= max_round):
            continue

        text = normalize_response_text(
            str(record.get("content") or ""),
            agent_names=agent_names,
        )
        if not text or _looks_like_referee_or_control(text):
            continue
        if excluded and text == excluded:
            continue
        if text in seen:
            continue

        seen.add(text)
        newest_first.append(text)
        if len(newest_first) >= bank_size:
            break

    return AvoidanceBank(tuple(reversed(newest_first)))


def build_avoidance_bank_from_log(
    path: str | Path,
    **kwargs: object,
) -> AvoidanceBank:
    return build_avoidance_bank(load_jsonl_records(path), **kwargs)
