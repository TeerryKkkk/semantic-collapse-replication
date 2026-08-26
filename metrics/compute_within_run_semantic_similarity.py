# -*- coding: utf-8 -*-
"""
Window-to-Previous Centroid Similarity (TF-IDF or Embedding)
-----------------------------------------------------------
- Split chat log by headers like: "===== Round N order: ... ====="
- Build per-round vectors via TF-IDF or Embedding
- Aggregate **non-overlapping** windows: [1..W], [W+1..2W], ...
- Compare each window's centroid to the **previous window**'s centroid

# === HOW TO USE ===
1) Supply an input log and output directory on the command line.
2) Set scientific CONFIG below:
   - REPRESENTATION = "tfidf" or "embedding"
   - BACKEND = "sbert" (uses all-MiniLM-L6-v2) or "lsa" (no-internet fallback)
   - WINDOW_SIZE, HOP (set HOP==WINDOW_SIZE for non-overlap: e.g., 5 means 1-5 vs 6-10)
   - DROP_INCOMPLETE_TAIL = True to drop trailing partial window
3) Run: python compute_within_run_semantic_similarity.py --input-path FILE --output-dir PATH
Outputs: CSV + PNG in the supplied output directory.
"""

import argparse
import re
from typing import List, Tuple, Dict
from pathlib import Path
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize as l2norm
from sklearn.decomposition import TruncatedSVD

from openai import OpenAI
# =====================
# ====== CONFIG =======
# =====================
WINDOW_SIZE = 10
HOP = WINDOW_SIZE                                 # use 5 for 1–5, 6–10, 11–15, ...
DROP_INCOMPLETE_TAIL = False              # drop last short window

# Representation switches
BACKEND = "openai"                        # if embedding: "sbert" or "lsa"
SBERT_MODEL = "" #BAAI/bge-large-en-v1.5 or sentence-transformers/all-MiniLM-L6-v2 or intfloat/e5-large-v2



REPRESENTATION = "embedding"
EMBED_BACKEND = "openai"        # "local" | "openai" | "lsa"

EMBED_MODEL_NAME = ""

# OpenAI embedding settings
OPENAI_EMBED_MODEL = "text-embedding-3-large"
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MAX_CHARS = 300_000
OPENAI_CHUNK_CHARS = 25_000
OPENAI_MAX_PARTS = 15

client = OpenAI(api_key=OPENAI_API_KEY)



# TF-IDF settings
TFIDF_PARAMS = dict(
    lowercase=True,
    ngram_range=(1, 2),
    stop_words="english",
    max_df=0.9,
    min_df=1,
)

# LSA embedding size (if BACKEND="lsa")
N_COMPONENTS_LSA = 128

# =====================
# ====== LOGIC ========
# =====================
ROUND_PATTERN = re.compile(r"===== Round\s+(\d+)\s+order:.*?=====", re.DOTALL)

# Line-level round parsing regexes aligned with compute_cross_run_semantic_similarity.py.
rg_order   = re.compile(r"^=+\s*Round\s+(\d+)\s+order", re.IGNORECASE)
rg_bracket = re.compile(r"^\[Round\s+(\d+)\]", re.IGNORECASE)

def parse_rounds(text: str) -> List[Tuple[int, str]]:
    """
    Parse rounds using the cross-run script's logic, keeping only natural-language utterances.

    Args:
        text: Raw contents of the whole log file.

    Returns:
        List[(round_id, round_text)], sorted by round.
    """
    # Split by line first.
    lines = text.splitlines()

    rg_order   = re.compile(r"^===== Round (\d+) order:")
    rg_bracket = re.compile(r"^\[Round (\d+)\]")

    round_text: Dict[int, List[str]] = {}
    cur_round = None

    for raw_line in lines:
        line = raw_line.rstrip("\n")

        # Skip fully blank lines.
        if not line.strip():
            continue

        # 1) "===== Round X order: [...] =====" only updates the current round.
        m_order = rg_order.match(line)
        if m_order:
            cur_round = int(m_order.group(1))
            round_text.setdefault(cur_round, [])
            continue

        # 2) "[Round X] (MAIN/...) AGENT said: '...'"
        m_bracket = rg_bracket.match(line)
        if m_bracket:
            cur_round = int(m_bracket.group(1))
            round_text.setdefault(cur_round, [])

            # Extract text only when "said:" is present.
            if "said:" in line:
                _, after = line.split("said:", 1)
                content = after.strip()

                # Remove leading/trailing single quotes.
                if content.startswith("'"):
                    content = content[1:]
                if content.endswith("'"):
                    content = content[:-1]

                if content:
                    round_text[cur_round].append(content)

            # Header line handled; do not pass it to later logic.
            continue

        # 3) Drop metadata/header/noise lines.
        stripped = line.lstrip()

        # a) Metadata lines such as action_name / valence / desc.
        if stripped.startswith("->"):
            continue

        # b) Header lines.
        if stripped.startswith("====="):
            continue

        # c) Fallback: drop any [Round ...] line not caught above.
        if stripped.startswith("[Round "):
            continue

        # 4) Treat remaining lines as natural language continuing the utterance.
        if cur_round is None:
            # Defensive guard; this should not occur.
            continue

        clean = stripped

        # Multi-line utterances may end with a trailing single quote.
        if clean.endswith("'"):
            clean = clean[:-1].rstrip()

        if clean:
            round_text.setdefault(cur_round, [])
            round_text[cur_round].append(clean)

    # 5) Join each round into one string.
    joined_round_text: Dict[int, str] = {
        r: " ".join(chunks) for r, chunks in round_text.items()
    }

    # 6) Convert to List[(round_id, text)] sorted by round for later window slicing.
    rounds_list: List[Tuple[int, str]] = sorted(
        joined_round_text.items(),
        key=lambda x: x[0]
    )
    return rounds_list


def build_window_texts(corpus: List[str], windows: List[Tuple[int, int]]) -> List[str]:
    """
    Build one window text by joining the selected round texts.
    corpus: per-round text list, where len(corpus) is the number of rounds.
    windows: [(s, e), ...], where s/e are 0-based indexes.
    Returns: one text string per window.
    """
    texts: List[str] = []
    for (s, e) in windows:
        pieces = []
        for t in corpus[s:e]:
            if t is None:
                continue
            t = t.strip()
            if t:
                pieces.append(t)
        # Avoid blank strings causing embedding errors.
        if pieces:
            texts.append("\n".join(pieces))
        else:
            texts.append(" ")
    return texts


def build_tfidf(corpus: List[str]):
    vec = TfidfVectorizer(**TFIDF_PARAMS)
    X = vec.fit_transform(corpus)            # sparse
    X = l2norm(X, norm="l2", axis=1)         # L2 rows
    return X



def make_windows(n_items: int, size: int, hop: int, drop_tail: bool=True):
    windows = []
    i = 0
    while i < n_items:
        s, e = i, i + size
        if e > n_items and drop_tail:
            break
        windows.append((s, min(e, n_items)))
        i += hop
    return windows

def window_centroids(X, windows):
    """
    X: per-round vectors (sparse or dense). Return dense, L2-normalized window centroids.
    """
    vecs = []
    for (s, e) in windows:
        Xi = X[s:e]
        count = max(1, e - s)
        if hasattr(Xi, "sum"):           # works for sparse & dense
            sumi = Xi.sum(axis=0)
            sumi = np.asarray(sumi).astype(float).ravel()
        else:
            sumi = np.asarray(Xi).astype(float).sum(axis=0).ravel()
        c = sumi / float(count)
        # L2 normalize
        c = c / (np.linalg.norm(c) + 1e-12)
        vecs.append(c)
    return np.vstack(vecs)               # (nW x d)

def compare_adjacent(win_vecs: np.ndarray):
    """
    Return an array where sims[i] = cosine(win[i], win[i-1]); sims[0] = NaN
    Since centroids are L2-normalized, cosine == dot.
    """

    """
    cosine(win[i], win[i-1]); sims[0] = NaN
    Since centroids are L2-normalized, cosine == dot.
    """
    nW = win_vecs.shape[0]
    sims = np.full(nW, np.nan, dtype=float)
    for i in range(1, nW):
        sims[i] = float(np.dot(win_vecs[i], win_vecs[i-1]))
    return sims

def compare_first_last(win_vecs: np.ndarray) -> float:
    """Cosine between the first and the last window centroid. NaN if <2 windows."""
    if win_vecs.shape[0] < 2:
        return float('nan')
    # centroids already L2-normalized -> cosine == dot
    return float((win_vecs[0] * win_vecs[-1]).sum())
def _embed_with_openai(text: str) -> np.ndarray:
    # Limit total length.
    text = text[:OPENAI_MAX_CHARS]

    # Split by character count to avoid overlong single inputs.
    parts = []
    for i in range(0, min(len(text), OPENAI_MAX_CHARS), OPENAI_CHUNK_CHARS):
        parts.append(text[i:i + OPENAI_CHUNK_CHARS])
        if len(parts) >= OPENAI_MAX_PARTS:
            break

    # One call, multiple inputs.
    resp = client.embeddings.create(
        model=OPENAI_EMBED_MODEL,
        input=parts,
    )
    vecs = [np.array(d.embedding, dtype="float32") for d in resp.data]
    v = np.mean(vecs, axis=0)  # Mean-pool multiple chunks.
    return v


def build_embeddings(corpus: List[str]):
    # ====== 1) OpenAI branch ======
    if EMBED_BACKEND.lower() == "openai":
        all_vecs = []
        for txt in corpus:
            v = _embed_with_openai(txt)
            all_vecs.append(v)
        E = np.vstack(all_vecs).astype("float32")
        # L2 normalization, matching the original code.
        E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)
        print(f"[OPENAI] using model: {OPENAI_EMBED_MODEL}, got shape={E.shape}")
        return E

    # ====== 2) Original local SBERT / LSA branch ======
    backend = BACKEND.lower()
    if backend == "sbert":
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(SBERT_MODEL)
            emb = model.encode(
                corpus,
                batch_size=64,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            print(f"[SBERT] using model: {SBERT_MODEL}, got shape={emb.shape}")
            return emb
        except Exception as e:
            print(f"[WARN] SBERT failed ({e}); falling back to LSA.")
            backend = "lsa"

    if backend == "lsa":
        tfidf = TfidfVectorizer(**TFIDF_PARAMS).fit(corpus)
        X = tfidf.transform(corpus)
        n_features = int(X.shape[1])
        n_comp = max(2, min(N_COMPONENTS_LSA, max(2, n_features - 1)))
        svd = TruncatedSVD(n_components=n_comp, random_state=42)
        E = svd.fit_transform(X)
        E = l2norm(E, norm="l2", axis=1)
        return E

    raise ValueError(f"Unknown BACKEND: {BACKEND}")
def compare_vs_first(win_vecs: np.ndarray):
    """
    Return an array where sims[i] = cosine(win[i], win[0]) for all i.
    centroids are L2-normalized, so cosine == dot.
    """
    nW = win_vecs.shape[0]
    sims = np.full(nW, np.nan, dtype=float)
    if nW == 0:
        return sims
    first_vec = win_vecs[0]
    for i in range(nW):
        sims[i] = float(np.dot(win_vecs[i], first_vec))
    return sims

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-name", default=None)
    return parser.parse_args()

def main():
    args = parse_args()
    output_name = args.output_name or args.input_path.stem

    print(f"[config] REPRESENTATION={REPRESENTATION}")
    if REPRESENTATION == "embedding":
        print(f"[config] EMBED_BACKEND={EMBED_BACKEND}")
        if EMBED_BACKEND == "local":
            print(f"[config] EMBED_MODEL_NAME={EMBED_MODEL_NAME}")
        elif EMBED_BACKEND == "openai":
            print(f"[config] OPENAI_EMBED_MODEL={OPENAI_EMBED_MODEL}")
    # Load
    p = args.input_path
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p.resolve()}")
    text = p.read_text(encoding="utf-8", errors="ignore")

    # Parse
    rounds = parse_rounds(text)
    round_nums = [r for (r, _) in rounds]
    corpus = [c for (_, c) in rounds]

    rep = REPRESENTATION.lower()

    # 1) Build one shared set of windows for both tfidf and embedding modes.
    windows = make_windows(len(corpus), WINDOW_SIZE, HOP, DROP_INCOMPLETE_TAIL)
    if len(windows) < 2:
        raise ValueError("Need at least 2 windows to compare against previous. Adjust WINDOW_SIZE/HOP.")

    # 2) Build window vectors by representation mode.
    if rep == "tfidf":
        # Original logic: build TF-IDF by round, then compute window centroids.
        X_round = build_tfidf(corpus)                    # sparse L2
        win_vecs = window_centroids(X_round, windows)    # dense L2
    elif rep == "embedding":
        # Option A: embed each window text directly, one OpenAI request per window.
        window_texts = build_window_texts(corpus, windows)
        win_vecs = build_embeddings(window_texts)        # dense L2; each row is one window
        print(f"[INFO] built window embeddings, shape={win_vecs.shape}")
    else:
        raise ValueError("REPRESENTATION must be 'tfidf' or 'embedding'")

    # 3) Keep similarity calculation unchanged.
    sims = compare_adjacent(win_vecs)                    # vs previous window
    sims_vs_first = compare_vs_first(win_vecs)           # vs first window
    first_last = compare_first_last(win_vecs)            # first vs last (global)
    try:
        print(f"[metric] cosine(first_window, last_window) = {first_last:.4f}")
    except Exception:
        print(f"[metric] cosine(first_window, last_window) = {first_last}")


    # Build result rows
    rows = []
    for idx, (s, e) in enumerate(windows):
        rows.append({
            "window_index": idx + 1,                     # 1-based
            "round_start": round_nums[s],
            "round_end": round_nums[e-1],
            "num_rounds": e - s,
            "sim_vs_prev": sims[idx],                    # NaN for first window
            "sim_vs_first": sims_vs_first[idx],          # each window vs first window
        })
    df = pd.DataFrame(rows)
    df["source"] = output_name           # Add source to detail rows.

    # Save outputs
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    tail = (BACKEND if rep=="embedding" else "tfidf")
    csv_path = out_dir / f"{output_name}__win_centroid_{rep}_{tail}_w{WINDOW_SIZE}_hop{HOP}.csv"   # Prefix with output_name.
    png_path = out_dir / f"{output_name}.png"

    df.to_csv(csv_path, index=False)


    # Extra summary (one row)
    try:
        adj_mean = float(np.nanmean(sims[1:])) if len(sims) > 1 else float('nan')
    except Exception:
        adj_mean = float('nan')
    summary_df = pd.DataFrame([{
        'W': WINDOW_SIZE,
        'hop': HOP,
        'n_windows': len(windows),
        'cosine_first_last': first_last,
        'adjacent_mean': adj_mean,
    }])
    summary_path = out_dir / f"{output_name}__summary_{rep}_{tail}_w{WINDOW_SIZE}_hop{HOP}.csv"     # Prefix with output_name.
    summary_df.to_csv(summary_path, index=False)

    # Plot
    plt.figure(figsize=(8, 5))

    # 1) Current window vs previous window.
    plt.plot(
        df["window_index"],
        df["sim_vs_prev"],
        marker="o",
        label="vs previous window"
    )

    # 2) Current window vs first window.
    plt.plot(
        df["window_index"],
        df["sim_vs_first"],
        marker="s",
        linestyle="--",
        label="vs first window"
    )

    plt.xlabel("Window index")
    plt.ylabel("Cosine similarity")
    title_tail = f"{rep.upper()} {tail.upper() if rep=='embedding' else ''}".strip()
    plt.title(
        f"Window Similarity (W={WINDOW_SIZE}, hop={HOP}) - {title_tail}\n"
        f"first vs last = {first_last:.3f}"
    )
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(png_path, dpi=160)
    plt.close()

    print(f"[OK] Windows: {windows}")
    print(f"[OK] Saved CSV: {csv_path.resolve()}")
    print(f"[OK] Saved PNG: {png_path.resolve()}")

if __name__ == "__main__":
    main()
