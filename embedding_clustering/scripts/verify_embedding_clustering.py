"""Verify embedding-clustering summaries and optionally refit selected solutions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from embedding_clustering.verification import verify_and_write  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--embeddings",
        type=Path,
        default=Path("data/embeddings_l2.npy"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/embedding_manifest.csv"),
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("results/embedding_clustering/public"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("results/embedding_clustering/public/verification.json"),
    )
    parser.add_argument(
        "--refit-selected",
        action="store_true",
        help=(
            "Refit only the five already-selected K/seed solutions and require exact "
            "saved-label agreement. This does not repeat K selection."
        ),
    )
    parser.add_argument("--tolerance", type=float, default=1e-12)
    args = parser.parse_args()
    report = verify_and_write(
        args.embeddings,
        args.manifest,
        args.results,
        args.report,
        refit_selected=args.refit_selected,
        tolerance=args.tolerance,
    )
    print(json.dumps({"passed": report["passed"], "report": str(args.report)}, indent=2))


if __name__ == "__main__":
    main()
