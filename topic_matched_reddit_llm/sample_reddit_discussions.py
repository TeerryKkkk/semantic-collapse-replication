"""Prepare the Reddit side of the topic-matched Human–LLM experiment.

Two selection modes are provided:
  * paper: reproduce the frozen six-thread manuscript sample in manifests/paper_threads.csv;
  * random: draw a new fixed-seed random sample from the same mechanically eligible pool.

No model API is called by this script.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
from pathlib import Path
import random
import shutil
import subprocess
from typing import Iterator, Optional

import tiktoken

from protocol import CONTINUATION_TOKENS, TOKENIZER_NAME

ROOT = Path(__file__).resolve().parent
DELETED_MARKERS = {"[deleted]", "[removed]"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=None, help="Path to the Reddit .jsonl.zst file.")
    parser.add_argument("--selection", choices=("paper", "random"), default="paper")
    parser.add_argument("--paper-manifest", type=Path, default=ROOT / "manifests" / "paper_threads.csv")
    parser.add_argument("--seed", type=int, default=20260817, help="Random-sampling seed for --selection random.")
    parser.add_argument("--n", type=int, default=6, help="Number of threads for --selection random.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "selection")
    parser.add_argument("--continuation-tokens", type=int, default=CONTINUATION_TOKENS)
    parser.add_argument("--seed-min-tokens", type=int, default=8)
    parser.add_argument("--seed-max-tokens", type=int, default=30)
    return parser.parse_args()


def resolve_dataset(path: Optional[Path]) -> Path:
    if path is not None:
        resolved = path.expanduser().resolve()
        if not resolved.exists():
            raise SystemExit(f"Dataset not found: {resolved}")
        return resolved
    candidates = sorted(ROOT.glob("*.jsonl.zst")) + sorted((ROOT / "data").glob("*.jsonl.zst"))
    candidates = list(dict.fromkeys(p.resolve() for p in candidates if p.is_file()))
    if len(candidates) != 1:
        raise SystemExit("Pass --dataset PATH, or place exactly one *.jsonl.zst file in the repository root or data/.")
    return candidates[0]


def iter_zstd_jsonl(path: Path) -> Iterator[dict]:
    try:
        import zstandard as zstd  # type: ignore
    except ImportError:
        zstd = None
    if zstd is not None:
        with path.open("rb") as raw:
            with zstd.ZstdDecompressor().stream_reader(raw) as reader:
                text = io.TextIOWrapper(reader, encoding="utf-8")
                for line_no, line in enumerate(text, 1):
                    if line.strip():
                        try:
                            yield json.loads(line)
                        except json.JSONDecodeError as exc:
                            raise RuntimeError(f"Invalid JSON at decompressed line {line_no}") from exc
        return
    zstd_bin = shutil.which("zstd")
    if not zstd_bin:
        raise RuntimeError("Install `zstandard` or make the `zstd` executable available.")
    process = subprocess.Popen(
        [zstd_bin, "-dc", str(path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )
    assert process.stdout is not None
    for line_no, line in enumerate(process.stdout, 1):
        if not line.strip():
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Invalid JSON at decompressed line {line_no}") from exc
    stderr = process.stderr.read() if process.stderr else ""
    rc = process.wait()
    if rc != 0:
        raise RuntimeError(f"zstd failed with exit code {rc}: {stderr}")


def _positive_timestamp(value) -> Optional[float]:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) and value > 0 else None


def clean_and_sort_comments(thread: dict) -> list[str]:
    """Apply the frozen Reddit cleaner and chronological ordering."""
    comments = thread.get("comments")
    if not isinstance(comments, list):
        raise RuntimeError("Thread comments field is not a list.")
    retained: list[tuple[float, int, str, str]] = []
    seen_ids: set[str] = set()
    post_id = str((thread.get("post") or {}).get("id") or "").strip()
    for original_index, comment in enumerate(comments):
        if not isinstance(comment, dict):
            continue
        body = comment.get("body")
        if not isinstance(body, str):
            continue
        body = body.strip()
        if not body or body.casefold() in DELETED_MARKERS:
            continue
        timestamp = _positive_timestamp(comment.get("created_utc"))
        if timestamp is None:
            continue
        author = comment.get("author")
        if isinstance(author, str) and author.strip().casefold() == "automoderator":
            continue
        comment_id = str(comment.get("id") or "").strip()
        if comment_id:
            if comment_id in seen_ids:
                raise RuntimeError(f"Duplicate comment id within thread {post_id}: {comment_id}")
            seen_ids.add(comment_id)
        link_id = str(comment.get("link_id") or "").strip()
        if post_id and link_id:
            normalized = link_id[3:] if link_id.startswith("t3_") else link_id
            if normalized != post_id:
                raise RuntimeError(f"Comment {comment_id!r} points to a different post.")
        retained.append((timestamp, original_index, comment_id, body))
    retained.sort(key=lambda row: (row[0], row[1], row[2]))
    return [row[3] for row in retained]


def mechanically_eligible(post: dict, *, title_tokens: int, human_tokens: int, args: argparse.Namespace) -> bool:
    if str(post.get("subreddit") or "").strip().casefold() != "askreddit":
        return False
    if not isinstance(post.get("selftext"), str) or post.get("selftext", "").strip():
        return False
    title = str(post.get("title") or "").strip()
    if not title or "?" not in title:
        return False
    if title.casefold() in DELETED_MARKERS or "removed by moderator" in title.casefold():
        return False
    if not (args.seed_min_tokens <= title_tokens <= args.seed_max_tokens):
        return False
    if bool(post.get("over_18")) or bool(post.get("stickied")) or bool(post.get("locked")):
        return False
    if post.get("removed_by_category") not in (None, ""):
        return False
    return human_tokens >= args.continuation_tokens


def fixed_seed_sample(rows: list[dict], *, n: int, seed: int) -> list[dict]:
    """Draw a reproducible random sample after sorting the eligible pool by thread id."""
    if n <= 0 or n > len(rows):
        raise ValueError(f"Requested n={n}, but eligible pool contains {len(rows)} threads.")
    pool = sorted(rows, key=lambda row: row["thread_id"])
    return random.Random(seed).sample(pool, n)


def read_paper_manifest(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or any(not row.get("thread_id") for row in rows):
        raise RuntimeError(f"Invalid paper manifest: {path}")
    return rows


def main() -> None:
    args = parse_args()
    dataset = resolve_dataset(args.dataset)
    encoder = tiktoken.get_encoding(TOKENIZER_NAME)
    paper_rows = read_paper_manifest(args.paper_manifest)
    paper_ids = {row["thread_id"] for row in paper_rows}

    eligible: list[dict] = []
    by_id: dict[str, dict] = {}
    records = 0
    for thread in iter_zstd_jsonl(dataset):
        records += 1
        post = thread.get("post") or {}
        thread_id = str(post.get("id") or "").strip()
        if not thread_id:
            continue
        title = str(post.get("title") or "").strip()
        comments = clean_and_sort_comments(thread)
        human_text = "\n".join(comments)
        human_ids = encoder.encode(human_text)
        title_ids = encoder.encode(title)
        row = {
            "thread_id": thread_id,
            "title": title,
            "title_tokens": len(title_ids),
            "valid_comments": len(comments),
            "human_total_tokens": len(human_ids),
            "human_token_ids": human_ids,
        }
        row["mechanically_eligible"] = mechanically_eligible(
            post, title_tokens=len(title_ids), human_tokens=len(human_ids), args=args
        )
        by_id[thread_id] = row
        if row["mechanically_eligible"]:
            eligible.append(row)

    if args.selection == "paper":
        missing = sorted(paper_ids - set(by_id))
        if missing:
            raise RuntimeError("Paper thread(s) not found in dataset: " + ", ".join(missing))
        selected = []
        for rank, paper in enumerate(paper_rows, 1):
            row = dict(by_id[paper["thread_id"]])
            if not row["mechanically_eligible"]:
                raise RuntimeError(f"Paper thread {row['thread_id']} does not satisfy the frozen mechanical eligibility rules.")
            expected_title = (paper.get("title") or "").strip()
            if expected_title and expected_title != row["title"]:
                raise RuntimeError(f"Title mismatch for paper thread {row['thread_id']}.")
            row["selected_rank"] = rank
            selected.append(row)
    else:
        try:
            selected = fixed_seed_sample(eligible, n=args.n, seed=args.seed)
        except ValueError as exc:
            raise RuntimeError(str(exc)) from exc
        for rank, row in enumerate(selected, 1):
            row["selected_rank"] = rank

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    human_root = output_dir / "selected_human"
    human_root.mkdir(exist_ok=True)

    manifest_rows = []
    for row in selected:
        thread_id = row["thread_id"]
        ids = row["human_token_ids"][: args.continuation_tokens]
        thread_dir = human_root / thread_id
        thread_dir.mkdir(parents=True, exist_ok=True)
        (thread_dir / "seed.txt").write_text(row["title"] + "\n", encoding="utf-8")
        (thread_dir / f"human_stream_{args.continuation_tokens}.token_ids.json").write_text(json.dumps(ids), encoding="utf-8")
        (thread_dir / f"human_stream_{args.continuation_tokens}.txt").write_text(encoder.decode(ids), encoding="utf-8")
        manifest_rows.append({
            "selected_rank": row["selected_rank"],
            "thread_id": thread_id,
            "title": row["title"],
            "valid_comments": row["valid_comments"],
            "human_total_tokens": row["human_total_tokens"],
            "continuation_tokens": args.continuation_tokens,
            "selection_mode": args.selection,
            "random_seed": args.seed if args.selection == "random" else "",
        })

    with (output_dir / "selection_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)
    (output_dir / "selection_metadata.json").write_text(json.dumps({
        "dataset": str(dataset),
        "records_read": records,
        "eligible_threads": len(eligible),
        "selection_mode": args.selection,
        "random_seed": args.seed if args.selection == "random" else None,
        "selected_threads": [row["thread_id"] for row in manifest_rows],
        "tokenizer": TOKENIZER_NAME,
        "continuation_tokens": args.continuation_tokens,
    }, indent=2), encoding="utf-8")

    print(f"Read {records} Reddit threads; mechanically eligible pool: {len(eligible)}")
    print(f"Prepared {len(selected)} thread(s) using selection={args.selection!r}.")
    for row in manifest_rows:
        print(f"  {row['selected_rank']}. {row['thread_id']} | {row['title']} | human={row['human_total_tokens']} tokens")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
