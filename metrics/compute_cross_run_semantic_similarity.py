# -*- coding: utf-8 -*-
"""
Cross-run analysis of multiple log files from the same model.
Outputs cross-file TF-IDF document cosine, Jaccard metrics,
per-file adjacent-window lexical growth, and plots.

"""

from __future__ import annotations
import argparse
import os
import re
import math
import itertools
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============ Config ============
# Embedding parameters used when VECTOR_MODE='embed'.
# Chinese option: 'BAAI/bge-small-zh-v1.5'; multilingual options: 'intfloat/multilingual-e5-small' or 'paraphrase-multilingual-MiniLM-L12-v2'.
VECTOR_MODE = "embed_openai"   # "embed_local" / "embed_openai" / "tfidf"
EMBED_MODEL_NAME = ""
OPENAI_EMBED_MODEL = "text-embedding-3-large"   # or "text-embedding-3-large"
EMBED_BATCH_SIZE = 64


WINDOW_SIZE = 10
ROUND_CUTOFF: int | None = None   # Analyze only the first 100 rounds; None means no cutoff

# Primary similarity metric name, set by mode.
PRIMARY_SIM_METRIC = 'tfidf_cosine' if VECTOR_MODE=='tfidf' else 'embed_cosine'
PRIMARY_METRIC_TITLE = 'TF-IDF document cosine (all pairs, weighted mean)' if VECTOR_MODE=='tfidf' else 'Embedding cosine (all pairs, weighted mean)'

# Plot style
PLOT_ALL_PAIRS_AS_DASHED = True
PLOT_WEIGHTED_MEAN_BOLD = True
MERGE_PANELS = False   # True additionally exports one panel combining all three metrics

# Vector mode: 'tfidf' or 'embed'


# TF-IDF parameters
TFIDF_LOWERCASE = True
TFIDF_TOKEN_PATTERN = r"(?u)\b\w+\b"
TFIDF_MIN_DF_GLOBAL = 2   # min_df when fitting the vectorizer on all window documents for a file pair

# ============ Parsing and utilities ============
rg_order   = re.compile(r"^=+\s*Round\s+(\d+)\s+order", re.IGNORECASE)
rg_bracket = re.compile(r"^\[Round\s+(\d+)\]", re.IGNORECASE)
tok_re     = re.compile(TFIDF_TOKEN_PATTERN)
import os
from openai import OpenAI
import numpy as np

# OpenAI API key is supplied through the environment.
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
client = OpenAI(api_key=OPENAI_API_KEY)

def build_openai_embeddings(texts, model="text-embedding-3-small", batch_size=128, dims=None):
    """
    texts: list[str]
    model: "text-embedding-3-small" / "text-embedding-3-large"
    dims: optional; 3-large can use a specified dimension such as 1024.
    """
    out = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i+batch_size]
        # Convert blank/None values to a space to avoid 400 responses.
        safe_chunk = [t if (isinstance(t, str) and t.strip()) else " " for t in chunk]
        resp = client.embeddings.create(
            model=model,
            input=safe_chunk,
            **({"dimensions": dims} if dims else {})
        )
        vecs = [item.embedding for item in resp.data]
        out.extend(vecs)
    # Convert to numpy so the later cosine calculation can use a direct dot product.
    arr = np.array(out, dtype="float32")
    # OpenAI embeddings are already normalized; the later code can still normalize defensively.
    return arr

def parse_round_texts(path: Path) -> Dict[int, str]:
    """Parse a log into {round_id: joined_text}, keeping only agent natural-language text."""
    round_texts: Dict[int, List[str]] = {}
    cur: int | None = None

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.rstrip("\n")

            # 1) "===== Round N order: [...] ====="
            m1 = rg_order.search(line)
            if m1:
                cur = int(m1.group(1))
                round_texts.setdefault(cur, [])
                continue

            # 2) "[Round N] (XXX) NAME said: '...'"
            m2 = rg_bracket.search(line)
            if m2:
                cur = int(m2.group(1))
                round_texts.setdefault(cur, [])

                # Extract the text after "said:".
                idx = line.find("said:")
                if idx != -1:
                    payload = line[idx + len("said:"):].strip()

                    # Original logic: remove paired leading/trailing quotes.
                    if len(payload) >= 2 and payload[0] == payload[-1] and payload[0] in ("'", '"'):
                        payload = payload[1:-1].strip()

                    # Fallback: handle unpaired boundary quotes from multi-line utterances.
                    if payload and payload[0] in ("'", '"'):
                        payload = payload[1:].lstrip()
                    if payload and payload[-1] in ("'", '"'):
                        payload = payload[:-1].rstrip()

                    if payload:
                        round_texts[cur].append(payload)

                continue  # Header line handled; continue with the next line.

            # Skip until the first round has been seen.
            if cur is None:
                continue

            stripped = line.strip()
            if not stripped:
                continue

            # 3) Drop metadata/noise lines.
            if stripped.startswith("->"):      # action_name / valence / desc, etc.
                continue
            if stripped.startswith("====="):   # Defensive drop for any "=====" line.
                continue
            if stripped.startswith("```"):     # Drop fenced blocks such as ```source / ```analysis.
                continue

            # 4) Treat all remaining lines as agent text.
            stripped = stripped.rstrip("'\"").rstrip()
            if stripped:
                round_texts[cur].append(stripped)

    return {r: "\n".join(txts) for r, txts in round_texts.items()}



def report_rounds(round_texts: Dict[int, str], name: str) -> Tuple[int, List[int], str]:
    rounds = sorted(round_texts.keys())
    max_r = max(rounds) if rounds else 0
    missing = [r for r in range(1, max_r + 1) if r not in round_texts]
    line = f"[CHECK] {name}: parsed_rounds={len(rounds)}, max_round={max_r}, missing_round_count={len(missing)}"
    if missing:
        head = ", ".join(map(str, missing[:30])); tail = " ..." if len(missing) > 30 else ""
        line += f"\n        missing round examples: {head}{tail}"
    return max_r, missing, line

def compute_windows(max_round: int, window: int) -> List[Tuple[int,int,int]]:
    wins = []; wid = 1; s = 1
    while s <= max_round:
        e = min(s + window - 1, max_round)
        wins.append((wid, s, e))
        wid += 1; s = e + 1
    return wins

def window_text(round_texts: Dict[int, str], s: int, e: int) -> str:
    return "\n".join(round_texts.get(r, "") for r in range(s, e+1)).strip()

def count_nonempty_rounds(round_texts: Dict[int, str], s: int, e: int) -> int:
    return sum(1 for r in range(s, e+1) if r in round_texts and round_texts[r].strip())

def toks(text: str) -> List[str]:
    return tok_re.findall(text.lower() if TFIDF_LOWERCASE else text)

def jaccard_sets(a_tokens: List[str], b_tokens: List[str]) -> float:
    A, B = set(a_tokens), set(b_tokens)
    U = len(A | B)
    return float(len(A & B) / U) if U > 0 else math.nan

def jaccard_bigrams(a_tokens: List[str], b_tokens: List[str]) -> float:
    Ab = set(zip(a_tokens, a_tokens[1:])) if len(a_tokens) > 1 else set()
    Bb = set(zip(b_tokens, b_tokens[1:])) if len(b_tokens) > 1 else set()
    U = len(Ab | Bb)
    return float(len(Ab & Bb) / U) if U > 0 else math.nan

# ============ Main flow ============
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-files", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = args.output_dir

    # 0. Output directory
    os.makedirs(output_dir, exist_ok=True)

    # 1. Collect log files
    files = [p for p in args.input_files if p.exists()]
    if not files:
        raise SystemExit("None of the supplied --input-files exist.")

    # 2. Parse all logs -> {path: {round_id: text}}
    parsed: Dict[Path, Dict[int, str]] = {}
    check_lines: List[str] = []
    max_round_global = 0
    for p in files:
        rt = parse_round_texts(p)
        if ROUND_CUTOFF is not None:
            rt = {r: t for r, t in rt.items() if r <= ROUND_CUTOFF}
        parsed[p] = rt
        max_r, missing, line = report_rounds(rt, p.name)
        check_lines.append(line)
        max_round_global = max(max_round_global, max_r)
    (output_dir / "checks.txt").write_text("\n".join(check_lines), encoding="utf-8")
    print("\n".join(check_lines))

    # 3. Align windows
    if ROUND_CUTOFF is not None:
        max_round_global = min(max_round_global, ROUND_CUTOFF)
    windows = compute_windows(max_round_global, WINDOW_SIZE)

    # 4. Prebuild file-window documents
    docs_by_file: Dict[Path, List[str]] = {p: [] for p in files}
    toks_by_file: Dict[Path, List[List[str]]] = {p: [] for p in files}
    nrounds_by_file: Dict[Path, List[int]] = {p: [] for p in files}
    for (wid, s, e) in windows:
        for p in files:
            doc = window_text(parsed[p], s, e)
            docs_by_file[p].append(doc)
            toks_by_file[p].append(toks(doc) if doc else [])
            nrounds_by_file[p].append(count_nonempty_rounds(parsed[p], s, e))

    # 5. Initialize by mode
    embed_model = None
    openai_client = None
    openai_embed_batch = None

    if VECTOR_MODE == "embed_local":
        from sentence_transformers import SentenceTransformer
        embed_model = SentenceTransformer(EMBED_MODEL_NAME)

    elif VECTOR_MODE == "embed_openai":

        # Suggested setup: set OPENAI_API_KEY=xxxx
        OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
        OPENAI_MODEL = OPENAI_EMBED_MODEL  # "text-embedding-3-small" / "text-embedding-3-large"

        # Safe per-window length limit and chunking strategy.
        MAX_WINDOW_CHARS = 300_000
        CHUNK_CHARS = 25_000
        MAX_PARTS = 15

        def _embed_one_long(text: str) -> np.ndarray:
            if not text:
                text = " "
            # Hard-truncate first to avoid very long inputs.
            text = text[:MAX_WINDOW_CHARS]

            parts = [text[i:i + CHUNK_CHARS] for i in range(0, len(text), CHUNK_CHARS)]
            parts = parts[:MAX_PARTS]

            vecs = []
            for part in parts:
                resp = openai_client.embeddings.create(
                    model=OPENAI_MODEL,
                    input=[part],
                )
                v = np.array(resp.data[0].embedding, dtype=np.float32)
                vecs.append(v)

            if len(vecs) == 1:
                return vecs[0]
            return np.mean(vecs, axis=0)

        def openai_embed_batch(texts: List[str]) -> List[List[float]]:
            out: List[List[float]] = []
            for i, t in enumerate(texts):
                if i % 20 == 0:
                    print(f"[openai-embed] {i}/{len(texts)} ...")
                vec = _embed_one_long(t)
                out.append(vec.tolist())
            return out


    # 5.5 For OpenAI mode, embed all documents once globally.
    if VECTOR_MODE == "embed_openai":
        docs_master: List[str] = []
        index_map: Dict[Tuple[str, int], int] = {}
        for p in files:
            for (wid, s, e), doc in zip(windows, docs_by_file[p]):
                idx = len(docs_master)
                _doc = doc if doc else " "
                docs_master.append(_doc)
                index_map[(p.name, wid)] = idx
        emb_store = np.asarray(openai_embed_batch(docs_master), dtype=np.float32)
    else:
        emb_store = None
        index_map = {}

    # 6. Compute window-level similarity by pair.
    pair_rows: List[dict] = []

    for p_i, p_j in itertools.combinations(files, 2):
        pair_name = f"{p_i.name}||{p_j.name}"

        docs_i = docs_by_file[p_i];  docs_j = docs_by_file[p_j]
        toks_i = toks_by_file[p_i];  toks_j = toks_by_file[p_j]
        nrs_i  = nrounds_by_file[p_i];  nrs_j  = nrounds_by_file[p_j]

        n = len(docs_i)
        assert n == len(docs_j) == len(toks_i) == len(toks_j) == len(nrs_i) == len(nrs_j)

        # 6.1 Get vectors by mode.
        if VECTOR_MODE == "tfidf":
            from sklearn.feature_extraction.text import TfidfVectorizer
            vect = TfidfVectorizer(
                lowercase=TFIDF_LOWERCASE,
                token_pattern=TFIDF_TOKEN_PATTERN,
                min_df=TFIDF_MIN_DF_GLOBAL,
            )
            corpus = [d if d else "" for d in (docs_i + docs_j)]
            X = vect.fit_transform(corpus)
            Ai = X[:n, :]
            Bj = X[n:, :]

        elif VECTOR_MODE == "embed_local":
            def _encode_local(texts: List[str]):
                batch = [t if (t and t.strip()) else "" for t in texts]
                return embed_model.encode(
                    batch,
                    batch_size=EMBED_BATCH_SIZE,
                    show_progress_bar=False,
                    normalize_embeddings=True,
                )
            Ai = np.asarray(_encode_local(docs_i), dtype=np.float32)
            Bj = np.asarray(_encode_local(docs_j), dtype=np.float32)

        elif VECTOR_MODE == "embed_openai":
            Ai = np.asarray(
                [emb_store[index_map[(p_i.name, wid)]] for (wid, _, _) in windows],
                dtype=np.float32,
            )
            Bj = np.asarray(
                [emb_store[index_map[(p_j.name, wid)]] for (wid, _, _) in windows],
                dtype=np.float32,
            )

            # L2-normalize each vector so the later dot product is cosine.
            Ai = Ai / np.clip(np.linalg.norm(Ai, axis=1, keepdims=True), 1e-8, None)
            Bj = Bj / np.clip(np.linalg.norm(Bj, axis=1, keepdims=True), 1e-8, None)


        # Fixed random permutation.
        rng = np.random.default_rng(42)
        perm = rng.permutation(n)

        # 6.2 Write each window.
        for (wid, s, e), di, dj, ti, tj, ni, nj, k in zip(
            windows, docs_i, docs_j, toks_i, toks_j, nrs_i, nrs_j, range(n)
        ):
            weight = int(min(ni, nj))
            primary_name = "tfidf_cosine" if VECTOR_MODE == "tfidf" else "embed_cosine"

            if not di or not dj:
                for metric in [
                    primary_name,
                    "jaccard_unigram",
                    "jaccard_bigram",
                    primary_name + "_rand",
                    "jaccard_unigram_rand",
                    "jaccard_bigram_rand",
                ]:
                    pair_rows.append(dict(
                        metric=metric, value=math.nan, weight=weight,
                        window_id=wid, round_start=s, round_end=e,
                        pair=pair_name, file_i=p_i.name, file_j=p_j.name
                    ))
                continue

            # Primary similarity plus baseline.
            if VECTOR_MODE == "tfidf":
                rowi = Ai[k]; rowj = Bj[k]
                sim_primary = float(rowi.multiply(rowj).sum())
                rowj_rand = Bj[perm[k]]
                sim_primary_rand = float(rowi.multiply(rowj_rand).sum())
            else:
                sim_primary = float(np.dot(Ai[k], Bj[k]))
                sim_primary_rand = float(np.dot(Ai[k], Bj[perm[k]]))

            # Jaccard plus baseline.
            jac_uni = jaccard_sets(ti, tj)
            jac_bi  = jaccard_bigrams(ti, tj)
            tj_rand = toks_j[perm[k]]
            jac_uni_rand = jaccard_sets(ti, tj_rand)
            jac_bi_rand  = jaccard_bigrams(ti, tj_rand)

            pair_rows += [
                dict(metric=primary_name, value=sim_primary, weight=weight,
                     window_id=wid, round_start=s, round_end=e,
                     pair=pair_name, file_i=p_i.name, file_j=p_j.name),
                dict(metric="jaccard_unigram", value=jac_uni, weight=weight,
                     window_id=wid, round_start=s, round_end=e,
                     pair=pair_name, file_i=p_i.name, file_j=p_j.name),
                dict(metric="jaccard_bigram", value=jac_bi, weight=weight,
                     window_id=wid, round_start=s, round_end=e,
                     pair=pair_name, file_i=p_i.name, file_j=p_j.name),

                dict(metric=primary_name + "_rand", value=sim_primary_rand, weight=weight,
                     window_id=wid, round_start=s, round_end=e,
                     pair=pair_name, file_i=p_i.name, file_j=p_j.name),
                dict(metric="jaccard_unigram_rand", value=jac_uni_rand, weight=weight,
                     window_id=wid, round_start=s, round_end=e,
                     pair=pair_name, file_i=p_i.name, file_j=p_j.name),
                dict(metric="jaccard_bigram_rand", value=jac_bi_rand, weight=weight,
                     window_id=wid, round_start=s, round_end=e,
                     pair=pair_name, file_i=p_i.name, file_j=p_j.name),
            ]

    # 7. Write pairwise_curves.csv.
    pairwise_csv = output_dir / "pairwise_curves.csv"
    pd.DataFrame(pair_rows).to_csv(pairwise_csv, index=False)
    print(f"[done] pairwise curves -> {pairwise_csv}")
    # --- NEW: save averaged (weighted) curve for the primary metric
    curves = pd.DataFrame(pair_rows)
    primary_name = "tfidf_cosine" if VECTOR_MODE == "tfidf" else "embed_cosine"

    avg_rows = []
    for wid, g in curves[curves["metric"] == primary_name].groupby("window_id"):
        v = g["value"].to_numpy(dtype=float)
        w = g["weight"].to_numpy(dtype=float)
        W = np.nansum(w)
        mean = float(np.nansum(v * w) / W) if W > 0 else math.nan
        avg_rows.append((int(wid), mean))

    if avg_rows:
        pd.DataFrame(avg_rows, columns=["window_id", "weighted_mean"])\
        .sort_values("window_id")\
        .to_csv(output_dir / "weighted_mean_primary.csv", index=False)
        print("[done] weighted mean (primary) ->", output_dir / "weighted_mean_primary.csv")

    # 8. Lexical growth.
    growth_rows: List[dict] = []
    for p in files:
        prev_uni: set[str] = set()
        prev_bi: set[Tuple[str, str]] = set()
        for (wid, s, e), ti in zip(windows, toks_by_file[p]):
            uni = set(ti)
            bi  = set(zip(ti, ti[1:])) if len(ti) > 1 else set()
            new_uni = uni - prev_uni
            new_bi  = bi  - prev_bi
            growth_rows.append(dict(
                file=p.name, window_id=wid, round_start=s, round_end=e,
                vocab_unigram=len(uni), vocab_bigram=len(bi),
                new_unigram=len(new_uni),
                new_unigram_ratio=(len(new_uni)/len(uni)) if len(uni)>0 else math.nan,
                new_bigram=len(new_bi),
                new_bigram_ratio=(len(new_bi)/len(bi)) if len(bi)>0 else math.nan,
            ))
            prev_uni, prev_bi = uni, bi

    growth_csv = output_dir / "perfile_vocab_growth.csv"
    pd.DataFrame(growth_rows).to_csv(growth_csv, index=False)
    print(f"[done] per-file vocab growth -> {growth_csv}")

    # 9. Plot.
    curves = pd.DataFrame(pair_rows)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    def plot_metric(metric: str, title: str):
        sub = curves[curves["metric"] == metric].copy()
        if sub.empty:
            return
        plt.figure(figsize=(12, 4.6))
        if PLOT_ALL_PAIRS_AS_DASHED:
            for pair, g in sub.groupby("pair"):
                plt.plot(g["window_id"], g["value"], linestyle="--", linewidth=2.0, alpha=0.6)
        if PLOT_WEIGHTED_MEAN_BOLD:
            rows = []
            for wid, g in sub.groupby("window_id"):
                v = g["value"].to_numpy(dtype=float)
                w = g["weight"].to_numpy(dtype=float)
                W = np.nansum(w)
                mean = float(np.nansum(v * w) / W) if W > 0 else math.nan
                rows.append((int(wid), mean))
            agg = pd.DataFrame(rows, columns=["window_id","weighted_mean"]).sort_values("window_id")
            plt.plot(agg["window_id"], agg["weighted_mean"], linewidth=3.0, label="Weighted mean")
        plt.title(title)
        plt.xlabel("Window ID")
        plt.ylabel("Similarity")
        plt.ylim(0.0, 1.0)
        plt.grid(True, axis="y", alpha=0.25)
        handles, labels = plt.gca().get_legend_handles_labels()
        if "Weighted mean" in labels:
            plt.legend([handles[-1]], ["Weighted mean"], loc="best")
        out = plots_dir / f"Pairwise_{metric}.png"
        plt.tight_layout()
        plt.savefig(out, dpi=150)
        plt.close()
        print(f"[plot] saved: {out}")

    primary_name = "tfidf_cosine" if VECTOR_MODE == "tfidf" else "embed_cosine"
    primary_title = "TF-IDF cosine (all pairs, weighted mean)" if VECTOR_MODE == "tfidf" else PRIMARY_METRIC_TITLE

    plot_metric(primary_name, primary_title)
    plot_metric("jaccard_unigram", "Unigram Jaccard (all pairs, weighted mean)")
    plot_metric("jaccard_bigram", "Bigram Jaccard (all pairs, weighted mean)")

    # Lexical growth plot.
    growth = pd.DataFrame(growth_rows)
    if not growth.empty:
        plt.figure(figsize=(12, 4.6))
        for name, g in growth.groupby("file"):
            plt.plot(g["window_id"], g["new_unigram_ratio"], linewidth=2.2, label=f"{name} new-unigram ratio")
        plt.title("Vocabulary growth per window (new unigram ratio vs previous window)")
        plt.xlabel("Window ID"); plt.ylabel("Ratio"); plt.ylim(0.0, 1.0)
        plt.grid(True, axis="y", alpha=0.25); plt.legend(loc="best")
        out = plots_dir / "Vocab_growth_new_unigram_ratio.png"
        plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()
        print(f"[plot] saved: {out}")

    # Combined panel.
    if MERGE_PANELS:
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        for ax, metric, ttl in zip(
            axes,
            [primary_name, "jaccard_unigram", "jaccard_bigram"],
            [primary_title, "Unigram Jaccard", "Bigram Jaccard"],
        ):
            sub = curves[curves["metric"] == metric]
            if PLOT_ALL_PAIRS_AS_DASHED:
                for pair, g in sub.groupby("pair"):
                    ax.plot(g["window_id"], g["value"], linestyle="--", linewidth=1.0, alpha=0.5)
            rows = []
            for wid, g in sub.groupby("window_id"):
                v = g["value"].to_numpy(dtype=float)
                w = g["weight"].to_numpy(dtype=float)
                W = np.nansum(w)
                mean = float(np.nansum(v * w) / W) if W > 0 else np.nan
                rows.append((int(wid), mean))
            agg = pd.DataFrame(rows, columns=["window_id","weighted_mean"]).sort_values("window_id")
            ax.plot(agg["window_id"], agg["weighted_mean"], linewidth=3.0, label="Weighted mean")
            ax.set_ylabel(ttl); ax.set_ylim(0.0, 1.0); ax.grid(True, axis="y", alpha=0.25)
        axes[-1].set_xlabel("Window ID")
        out = plots_dir / "Summary_panel.png"
        plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()
        print(f"[plot] saved: {out}")



if __name__ == "__main__":
    main()
