#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compute_lexical_diversity.py - PROMPT experiment version (corrected)

Purpose:
- Compute lexical growth for logs under PROMPT_RESULTS/.
- Support filename formats:
    deepseek_diff_V1.txt
    deepseek_newopen_V4.txt  (supports newopen)
    gpt_REGULAR_V2.txt       (supports regular -> mapped to t0.9)
    phi0.9_V3.txt            (supports legacy format)

Outputs:
    PROMPT_LEXICAL_OUT/
        ├── {model}_cumulative.png
        ├── {model}_diff.png
        ├── CUMULATIVE_ALL.csv
        └── DIFF_ALL.csv
"""

from __future__ import annotations

import re
import pathlib
import unicodedata
from typing import Dict, List, Tuple, Optional

import matplotlib.pyplot as plt
import pandas as pd

# ========== Basic config ==========
INPUT_DIR = "PROMPT_RESULTS"      # txt log directory
OUT_DIR   = "PROMPT_LEXICAL_OUT"  # output directory for images and CSV files

MODELS       = ["deepseek", "gpt", "phi"]
# Note: regular is mapped to t0.9, so this list uses t0.9.
PROMPT_TYPES = ["t0.9", "diff", "history", "new_open", "reverse"]

# Compute by round (W=1,H=1) or by window (for example 10,10).
WINDOW_SIZE = 10
HOP         = WINDOW_SIZE

# ========== Filename parsing (corrected) ==========
# 1) Match diff / history / new_open / regular / newopen.
#    model supports arbitrary letters; version supports arbitrary digits.
RE_PROMPT = re.compile(
    r"^(?P<model>[A-Za-z]+)_(?P<ptype>diff|history|new_open|newopen|regular|reverse)_V(?P<ver>\d+)\.txt$",
    re.IGNORECASE,
)
# 2) baseline (legacy temperature 0.9 format): deepseek0.9_V1.txt
RE_STD = re.compile(
    r"^(?P<model>[A-Za-z]+)0\.9_V(?P<ver>\d+)\.txt$",
    re.IGNORECASE,
)


def parse_filename(name: str) -> Optional[Tuple[str, str, str, str]]:
    """
    Input: filename including .txt.
    Returns: (model, prompt_type, version, scenario)
      - prompt_type is normalized:
         newopen -> new_open
         regular -> t0.9
    """
    m = RE_PROMPT.match(name)
    if m:
        model = m.group("model").lower()
        raw_ptype = m.group("ptype").lower()
        ver = f"V{m.group('ver')}"
        
        # Mapping logic.
        if raw_ptype == "newopen":
            ptype = "new_open"
            scenario = "prompt"
        elif raw_ptype == "regular":
            ptype = "t0.9"      # Treat regular as baseline t0.9.
            scenario = "standard"
        else:
            ptype = raw_ptype
            scenario = "prompt"
            
        return model, ptype, ver, scenario

    m2 = RE_STD.match(name)
    if m2:
        model = m2.group("model").lower()
        ptype = "t0.9"
        ver = f"V{m2.group('ver')}"
        return model, ptype, ver, "standard"

    return None


# ========== Tokenization ==========
TOKEN_RE = re.compile(r"[a-z]+")
STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "is", "are", "was", "were", "be",
    "been", "being", "for", "on", "with", "as", "at", "by", "from", "that", "this",
    "it", "its", "into", "about", "than",
}


def tokenise(txt: str) -> List[str]:
    txt = unicodedata.normalize("NFKD", txt)
    txt = txt.encode("ascii", "ignore").decode("ascii")
    txt = txt.lower()
    toks = TOKEN_RE.findall(txt)
    toks = [t for t in toks if 2 <= len(t) <= 20 and t not in STOPWORDS]
    return toks


# ========== Log parsing ==========
_PATTERNS = [
    re.compile(
        r"\[Round\s+(?P<round>\d+)\]\s*\((?P<role>[^)]+)\)\s*(?P<agent>[A-Za-z0-9_\-]+)\s+said:\s*'(?P<text>.*?)'\s*(?:->|$)",
        re.DOTALL | re.IGNORECASE,
    ),
    re.compile(
        r"\[Round\s+(?P<round>\d+)\]\s*\((?P<role>[^)]+)\)\s*(?P<agent>[A-Za-z0-9_\-]+)\s+said:\s*\"(?P<text>.*?)\"\s*(?:->|$)",
        re.DOTALL | re.IGNORECASE,
    ),
    re.compile(
        r"\[Round\s+(?P<round>\d+)\]\s*\((?P<role>[^)]+)\)\s*(?P<agent>[A-Za-z0-9_\-]+)\s*:\s*['\"]?(?P<text>.*?)['\"]?\s*(?:->|$)",
        re.DOTALL | re.IGNORECASE,
    ),
]
_ROUND_HEAD = re.compile(r"\[Round\s+(?P<round>\d+)\]", re.IGNORECASE)


def parse_log_fuzzy(path: pathlib.Path) -> Dict[int, Dict[str, List[str]]]:
    """Return {round_idx: {"ALL": [tokens...]}}."""
    text = path.read_text(encoding="utf-8", errors="ignore")
    per_round: Dict[int, Dict[str, List[str]]] = {}
    hits = 0

    for pat in _PATTERNS:
        for m in pat.finditer(text):
            rnd = int(m["round"])
            toks = tokenise(m["text"])
            if not toks:
                continue
            d = per_round.setdefault(rnd, {"ALL": []})
            d["ALL"].extend(toks)
            hits += 1
        if hits:
            break

    # Fallback: split coarsely by [Round k] blocks.
    if not hits:
        blocks = list(_ROUND_HEAD.finditer(text))
        for i, bm in enumerate(blocks):
            rnd = int(bm["round"])
            start = bm.end()
            end = blocks[i + 1].start() if i + 1 < len(blocks) else len(text)
            chunk = text[start:end]
            # Remove tails such as "-> summary".
            chunk = re.split(r"\n\s*->.*?$", chunk, flags=re.MULTILINE)[0]
            toks = tokenise(chunk)
            if toks:
                d = per_round.setdefault(rnd, {"ALL": []})
                d["ALL"].extend(toks)

    return {k: per_round[k] for k in sorted(per_round)}


# ========== Statistics ==========
def build_curve(tokens_per_idx: Dict[int, List[str]]) -> Tuple[List[int], List[int]]:
    seen = set()
    xs: List[int] = []
    ys: List[int] = []
    for idx in sorted(tokens_per_idx):
        xs.append(idx)
        seen.update(tokens_per_idx[idx])
        ys.append(len(seen))
    return xs, ys


def to_diff(series: List[int]) -> List[int]:
    if not series:
        return []
    return [series[0]] + [series[i] - series[i - 1] for i in range(1, len(series))]


def rounds_to_windows(tokens_per_round: Dict[int, List[str]], W: int, H: int) -> Dict[int, List[str]]:
    rnds = sorted(tokens_per_round)
    res: Dict[int, List[str]] = {}
    i = 0
    widx = 1
    while i < len(rnds):
        block = rnds[i: i + W]
        if not block:
            break
        toks: List[str] = []
        for r in block:
            toks.extend(tokens_per_round[r])
        if toks:
            res[widx] = toks
        widx += 1
        i += H
    return res


def avg_curves(curves: List[Tuple[List[int], List[int]]]):
    if not curves:
        return None
    n = min(len(s) for _, s in curves)
    xs = curves[0][0][:n]
    ys = [sum(c[1][i] for c in curves) / float(len(curves)) for i in range(n)]
    return xs, ys


# ========== Main flow ==========
def ensure_dir(p: pathlib.Path):
    p.mkdir(parents=True, exist_ok=True)


def get_curve(path: pathlib.Path, kind: str):
    per_round = parse_log_fuzzy(path)
    tokens_all = {r: per_round[r]["ALL"] for r in per_round}
    if WINDOW_SIZE > 1 or HOP > 1:
        tokens_seq = rounds_to_windows(tokens_all, WINDOW_SIZE, HOP)
    else:
        tokens_seq = tokens_all
    x, y = build_curve(tokens_seq)
    if kind == "diff":
        y = to_diff(y)
    return x, y


def main():
    in_dir = pathlib.Path(INPUT_DIR)
    if not in_dir.exists():
        raise SystemExit(f"Input directory does not exist: {in_dir.resolve()}")

    ensure_dir(pathlib.Path(OUT_DIR))

    all_txt = sorted(p for p in in_dir.rglob("*.txt"))
    if not all_txt:
        raise SystemExit(f"No txt files found: {in_dir.resolve()}")

    # (model, prompt_type) -> list[(path, version, scenario)]
    buckets: Dict[Tuple[str, str], List[Tuple[pathlib.Path, str, str]]] = {
        (m, pt): [] for m in MODELS for pt in PROMPT_TYPES
    }
    
    count = 0
    print("[INFO] Files found:")
    for p in all_txt:
        parsed = parse_filename(p.name)
        if not parsed:
            # print(f"  [skip] Could not parse: {p.name}")
            continue
        model, ptype, version, scenario = parsed
        if model not in MODELS:
            continue
        if ptype not in PROMPT_TYPES:
            continue

        buckets[(model, ptype)].append((p, version, scenario))
        count += 1
        print(f"  - {p.name:28s} -> model={model:8s} type={ptype:8s} ver={version:3s}")

    print(f"\n[INFO] Loaded {count} files.")

    # Rows collected for CSV output.
    CUM_ROWS: List[dict] = []
    DIFF_ROWS: List[dict] = []

    index_type = "Window" if (WINDOW_SIZE > 1 or HOP > 1) else "Round"

    # Use consistent colors: different prompt_type values get different colors within each model.
    for kind, rows in (("cumulative", CUM_ROWS), ("diff", DIFF_ROWS)):
        for model in MODELS:
            # Skip plotting when this model has no data.
            has_data = any(buckets.get((model, pt)) for pt in PROMPT_TYPES)
            if not has_data:
                continue

            plt.figure(figsize=(10, 5))
            colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0", "C1", "C2", "C3"])
            type2color = {pt: colors[i % len(colors)] for i, pt in enumerate(PROMPT_TYPES)}

            for ptype in PROMPT_TYPES:
                entries = buckets.get((model, ptype), [])
                if not entries:
                    continue
                curves: List[Tuple[List[int], List[int]]] = []

                # Plot each version first with dashed lines.
                # Sort by version number (V1, V2, ...).
                for p, version, scenario in sorted(entries, key=lambda x: int(x[1][1:])):
                    x, y = get_curve(p, kind=kind)
                    curves.append((x, y))
                    # Usually only the mean line is labeled for readability.
                    # Keep the existing behavior here.
                    # label = f"{ptype} {version}"
                    plt.plot(
                        x, y,
                        linestyle="--",
                        linewidth=1.0,
                        color=type2color[ptype],
                        alpha=0.4 # Keep dashed lines lighter.
                    )
                    for ix, val in zip(x, y):
                        rows.append({
                            "model": model,
                            "prompt_type": ptype,
                            "scenario": scenario,
                            "version": version,
                            "source": p.stem,
                            "W": WINDOW_SIZE,
                            "hop": HOP,
                            "index_type": index_type,
                            "index": ix,
                            "value": float(val),
                            "is_mean": 0,
                        })

                # Then plot the mean with a solid line.
                avg = avg_curves(curves)
                if avg:
                    ax, ay = avg
                    scenario0 = entries[0][2]
                    plt.plot(
                        ax, ay,
                        linestyle="-",
                        linewidth=2.5,
                        color=type2color[ptype],
                        label=f"{ptype} avg",
                    )
                    for ix, val in zip(ax, ay):
                        rows.append({
                            "model": model,
                            "prompt_type": ptype,
                            "scenario": scenario0,
                            "version": "",
                            "source": f"{model}_{ptype}_MEAN",
                            "W": WINDOW_SIZE,
                            "hop": HOP,
                            "index_type": index_type,
                            "index": ix,
                            "value": float(val),
                            "is_mean": 1,
                        })

            x_label = index_type
            plt.xlabel(x_label)
            if kind == "cumulative":
                y_label = "Cumulative unique tokens"
            else:
                y_label = f"New unique tokens / {x_label.lower()}"
            plt.ylabel(y_label)

            title = f"{model.upper()} – {y_label}"
            if WINDOW_SIZE > 1 or HOP > 1:
                title += f"  (W={WINDOW_SIZE}, H={HOP})"
            plt.title(title)

            plt.tight_layout()
            plt.legend(ncol=2, fontsize=8)
            out_png = pathlib.Path(OUT_DIR) / f"{model}_{kind}.png"
            plt.savefig(out_png, dpi=300)
            plt.close()
            print(f"✅ Saved: {out_png}")

    out_dir = pathlib.Path(OUT_DIR)
    if CUM_ROWS:
        pd.DataFrame(CUM_ROWS).to_csv(out_dir / "CUMULATIVE_ALL.csv", index=False)
        print(f"📝 Wrote {out_dir / 'CUMULATIVE_ALL.csv'}")
    if DIFF_ROWS:
        pd.DataFrame(DIFF_ROWS).to_csv(out_dir / "DIFF_ALL.csv", index=False)
        print(f"📝 Wrote {out_dir / 'DIFF_ALL.csv'}")


if __name__ == "__main__":
    main()
