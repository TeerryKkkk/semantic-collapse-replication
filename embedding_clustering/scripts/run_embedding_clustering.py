"""CLI entry point for the full within-model clustering workflow."""

from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from embedding_clustering.run_analysis import main  # noqa: E402


if __name__ == "__main__":
    main()
