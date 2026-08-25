from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
CACHE_DIR = ROOT / "cache"
RESULTS_DIR = ROOT / "results"

PER_RUN_FILE = RESULTS_DIR / "baseline_vendi_per_run.csv"
SUMMARY_FILE = RESULTS_DIR / "baseline_vendi_summary.csv"

BLOCK_SIZE = 16
M = 10
RAREFACTION_DRAWS = 200
BOOTSTRAP_DRAWS = 10_000
TOKENIZER = "cl100k_base"
EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_DIMENSION = 3072
PHASES = ("early", "middle", "late")
EXPECTED_FAMILIES = (
    "DeepSeek-V3",
    "GPT-4-mini",
    "Phi-4",
    "GPT-5.6 Terra",
    "Claude Sonnet 5",
)
EXPECTED_RUNS_PER_FAMILY = 3
SEED_BASE = 20260804
