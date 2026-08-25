from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import tiktoken
import zstandard as zstd

from config import (
    BLOCK_SIZE,
    DATA_DIR,
    EXPECTED_FAMILIES,
    EXPECTED_RUNS_PER_FAMILY,
    M,
    PHASES,
    TOKENIZER,
)


ROUND_ORDER_RE = re.compile(r"^=+\s*Round\s+(\d+)\s+order", re.IGNORECASE)
ROUND_BRACKET_RE = re.compile(r"^\[Round\s+(\d+)\]", re.IGNORECASE)
SAID_LINE_RE = re.compile(
    r"^\[Round\s+(\d+)\]\s*(?:\(([^)]*)\)\s*)?(.*?)\s+said:\s*(.*)$",
    re.IGNORECASE,
)
CONTINUATION_MARKER_RE = re.compile(
    r"^\s*#?\s*=+\s*CONTINUATION(?:\s+(?:START|END))?\s*=+\s*$",
    re.IGNORECASE,
)
INFRASTRUCTURE_LINE_RE = re.compile(
    r"^\s*(?:progress|retry(?:ing|\s+notice)?|error|exception|traceback|"
    r"system\s+message|referee|routing|router)\s*[:=]",
    re.IGNORECASE,
)
NON_AGENT_MARKERS = (
    "referee", "system", "routing", "router", "classifier", "classification",
    "summary", "metadata", "moderator", "judge", "evaluator",
)
PLACEHOLDER_RE = re.compile(
    r"^(?:none|null|empty|no\s*(?:response|output|content)|n/?a|error|failed|failure|"
    r"timeout|refusal|refused|\.\.\.)$",
    re.IGNORECASE,
)


def read_zstd_text(path: Path) -> str:
    with path.open("rb") as raw, zstd.ZstdDecompressor().stream_reader(raw) as reader:
        return reader.read().decode("utf-8", errors="strict")


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value)).strip()


def is_non_agent(tag: str, speaker: str) -> bool:
    probe = f"{tag} {speaker}".lower()
    return any(marker in probe for marker in NON_AGENT_MARKERS)


def strip_quote_start(value: str) -> str:
    value = value.strip()
    return value[1:] if value.startswith("'") else value


def strip_quote_end(value: str) -> str:
    value = value.strip()
    return value[:-1] if value.endswith("'") else value


def parse_established(text: str) -> tuple[list[str], set[int]]:
    messages: list[str] = []
    rounds: set[int] = set()
    current_round: int | None = None
    current: list[str] | None = None

    def finalize() -> None:
        nonlocal current
        if current is not None:
            value = normalize_text(" ".join(normalize_text(x) for x in current if normalize_text(x)))
            if value:
                messages.append(value)
        current = None

    for raw_line in text.splitlines():
        line = raw_line.rstrip("\n")
        if not line.strip():
            continue
        match = ROUND_ORDER_RE.match(line)
        if match:
            finalize()
            current_round = int(match.group(1))
            rounds.add(current_round)
            continue
        match = SAID_LINE_RE.match(line)
        if match:
            finalize()
            current_round = int(match.group(1))
            rounds.add(current_round)
            tag = (match.group(2) or "").strip()
            speaker = normalize_text(match.group(3) or "")
            content = strip_quote_end(strip_quote_start(match.group(4) or ""))
            current = None if is_non_agent(tag, speaker) else ([content] if content else [])
            continue
        match = ROUND_BRACKET_RE.match(line)
        if match:
            finalize()
            current_round = int(match.group(1))
            rounds.add(current_round)
            continue
        stripped = line.lstrip()
        if stripped.startswith("->"):
            continue
        if stripped.startswith("=====") or stripped.startswith("[Round "):
            finalize()
            continue
        if current_round is not None and current is not None:
            continuation = strip_quote_end(stripped)
            if continuation:
                current.append(continuation)
    finalize()
    return messages, rounds


def parse_frontier(text: str) -> tuple[list[str], set[int]]:
    messages: list[str] = []
    rounds: set[int] = set()
    current: dict[str, object] | None = None
    message_open = False

    def finalize() -> None:
        nonlocal current, message_open
        if current is not None:
            chunks = current["chunks"]
            assert isinstance(chunks, list)
            value = normalize_text(" ".join(normalize_text(x) for x in chunks if normalize_text(x)))
            value = normalize_text(strip_quote_end(value))
            tag, speaker = str(current["tag"]), str(current["speaker"])
            if value and not PLACEHOLDER_RE.fullmatch(value) and not is_non_agent(tag, speaker):
                messages.append(value)
        current = None
        message_open = False

    for raw_line in text.splitlines():
        line = raw_line.rstrip("\n")
        match = ROUND_ORDER_RE.match(line)
        if match:
            finalize()
            rounds.add(int(match.group(1)))
            continue
        match = SAID_LINE_RE.match(line)
        if match:
            finalize()
            rounds.add(int(match.group(1)))
            content = strip_quote_start(match.group(4) or "")
            current = {
                "tag": (match.group(2) or "").strip(),
                "speaker": normalize_text(match.group(3) or ""),
                "chunks": [content] if content else [],
            }
            message_open = True
            continue
        if ROUND_BRACKET_RE.match(line):
            finalize()
            continue
        stripped = line.lstrip()
        if not stripped:
            continue
        if stripped.startswith("->"):
            message_open = False
            continue
        if CONTINUATION_MARKER_RE.match(stripped) or stripped.startswith("====="):
            finalize()
            continue
        if INFRASTRUCTURE_LINE_RE.match(stripped) and not message_open:
            continue
        if current is not None and message_open:
            chunks = current["chunks"]
            assert isinstance(chunks, list)
            chunks.append(strip_quote_end(stripped))
    finalize()
    return messages, rounds


def llm_source_specifications() -> list[dict[str, str]]:
    """The manuscript baseline cohort: five families x three 1,000-round runs."""
    specifications: list[dict[str, str]] = []
    groups = [
        ("3_deepseek_1000_v{n}.txt.zst", "deepseek-v3_baseline_standard_temp0p9_v{n}", "DeepSeek-V3", "established"),
        ("3_gpt_1000_v{n}.txt.zst", "gpt-4-mini_baseline_standard_temp0p9_v{n}", "GPT-4-mini", "established"),
        ("3_phi-4_1000_v{n}.txt.zst", "phi-4_baseline_standard_temp0p9_v{n}", "Phi-4", "established"),
        ("gpt5.6_1000_v{n}.txt.zst", "gpt-5.6-terra_baseline_1000_v{n}", "GPT-5.6 Terra", "frontier"),
        ("sonnet_1000_v{n}.txt.zst", "claude-sonnet-5_baseline_1000_v{n}", "Claude Sonnet 5", "frontier"),
    ]
    for filename, unit_id, family, parser in groups:
        for number in range(1, EXPECTED_RUNS_PER_FAMILY + 1):
            specifications.append(
                {
                    "filename": filename.format(n=number),
                    "unit_id": unit_id.format(n=number),
                    "family": family,
                    "parser": parser,
                }
            )
    return sorted(specifications, key=lambda item: item["unit_id"])


def load_llm_units() -> list[dict[str, object]]:
    source_dir = DATA_DIR / "llm"
    if not source_dir.is_dir():
        raise FileNotFoundError("Missing local LLM source directory: data/llm")
    specifications = llm_source_specifications()
    expected_files = {item["filename"] for item in specifications}
    observed_files = {path.name for path in source_dir.glob("*.zst")}
    missing = expected_files - observed_files
    if missing:
        raise RuntimeError(f"Missing baseline transcript files: {sorted(missing)}")

    units: list[dict[str, object]] = []
    for item in specifications:
        text = read_zstd_text(source_dir / item["filename"])
        values, rounds = (
            parse_established(text)
            if item["parser"] == "established"
            else parse_frontier(text)
        )
        if rounds != set(range(1, 1001)):
            raise RuntimeError(f"Incomplete 1,000-round sequence: {item['unit_id']}")
        units.append(
            {
                "unit_id": item["unit_id"],
                "family": item["family"],
                "texts": values,
            }
        )

    frame = pd.DataFrame([{"unit_id": u["unit_id"], "family": u["family"]} for u in units])
    counts = frame.groupby("family").size().to_dict()
    expected = {family: EXPECTED_RUNS_PER_FAMILY for family in EXPECTED_FAMILIES}
    if counts != expected:
        raise RuntimeError(f"Baseline cohort mismatch: {counts}")
    return units


def phase_slices(n: int) -> dict[str, tuple[int, int]]:
    return {
        "early": (0, n // 3),
        "middle": (n // 3, (2 * n) // 3),
        "late": ((2 * n) // 3, n),
    }


def strict_token_blocks(texts: list[str], encoding: tiktoken.Encoding) -> list[str]:
    """Create exact 16-token blocks, retaining only losslessly round-trippable blocks."""
    stream = "\n".join(texts)
    token_ids = encoding.encode(stream, disallowed_special=()) if stream else []
    pieces = [encoding.decode_single_token_bytes(token_id) for token_id in token_ids]
    if b"".join(pieces) != stream.encode("utf-8"):
        raise RuntimeError("Token bytes do not reconstruct the phase stream")

    candidates: list[tuple[str, list[int]]] = []
    for block_zero in range(len(token_ids) // BLOCK_SIZE):
        start = block_zero * BLOCK_SIZE
        original = token_ids[start : start + BLOCK_SIZE]
        try:
            decoded = b"".join(pieces[start : start + BLOCK_SIZE]).decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            continue
        if decoded:
            candidates.append((decoded, original))
    if not candidates:
        return []

    decoded_values = [item[0] for item in candidates]
    reencoded_values = encoding.encode_ordinary_batch(
        decoded_values, num_threads=min(8, __import__("os").cpu_count() or 1)
    )
    return [
        text
        for (text, original), reencoded in zip(candidates, reencoded_values)
        if len(reencoded) == BLOCK_SIZE and list(reencoded) == list(original)
    ]


def construct_block_pools() -> tuple[dict[tuple[str, str], list[str]], pd.DataFrame]:
    encoding = tiktoken.get_encoding(TOKENIZER)
    pools: dict[tuple[str, str], list[str]] = {}
    index_rows: list[dict[str, object]] = []

    for unit in load_llm_units():
        values = unit["texts"]
        assert isinstance(values, list)
        phase_blocks: dict[str, list[str]] = {}
        for phase, (start, stop) in phase_slices(len(values)).items():
            phase_blocks[phase] = strict_token_blocks(values[start:stop], encoding)
        if min(len(phase_blocks[phase]) for phase in PHASES) < M:
            raise RuntimeError(f"Insufficient exact-token blocks in run: {unit['unit_id']}")
        unit_id, family = str(unit["unit_id"]), str(unit["family"])
        index_rows.append({"unit_id": unit_id, "family": family})
        for phase in PHASES:
            pools[(unit_id, phase)] = phase_blocks[phase]

    index = pd.DataFrame(index_rows)
    if len(index) != len(EXPECTED_FAMILIES) * EXPECTED_RUNS_PER_FAMILY:
        raise RuntimeError(f"Expected 15 baseline runs, found {len(index)}")
    return pools, index
