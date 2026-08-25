from __future__ import annotations

from config import (
    BLOCK_SIZE,
    M,
    PER_RUN_FILE,
    RAREFACTION_DRAWS,
    RESULTS_DIR,
    SUMMARY_FILE,
)
from src.embeddings import load_or_create_embeddings
from src.preprocessing import construct_block_pools
from src.statistics import analyze
from src.vendi import score_block_pools


def main() -> int:
    pools, unit_index = construct_block_pools()
    texts = {text for pool in pools.values() for text in pool}
    embeddings = load_or_create_embeddings(texts)
    per_run = score_block_pools(pools, unit_index, embeddings)
    summary = analyze(per_run)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    per_run.to_csv(PER_RUN_FILE, index=False, encoding="utf-8", lineterminator="\n")
    summary.to_csv(SUMMARY_FILE, index=False, encoding="utf-8", lineterminator="\n")

    print(
        f"baseline Vendi: block_size={BLOCK_SIZE}, m={M}, "
        f"rarefaction_draws={RAREFACTION_DRAWS}"
    )
    print(summary.to_string(index=False))
    print(f"wrote {PER_RUN_FILE}")
    print(f"wrote {SUMMARY_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
