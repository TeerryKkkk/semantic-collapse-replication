"""Fixed scientific settings for the two topic analyses."""

PRIMARY_K = 20
PRIMARY_SEED = 42
ROBUSTNESS_K = (10, 15, 20, 30, 40, 50)
ROBUSTNESS_SEEDS = tuple(range(10))

INTERVAL_SIZE = 10
EXPECTED_ROUNDS = 1000
EXPECTED_INTERVALS = 100
EARLY_INTERVALS = tuple(range(1, 11))
LATE_INTERVALS = tuple(range(91, 101))

MIN_DF = 5
MAX_DF = 0.95
TOKEN_PATTERN = r"(?u)\b[a-zA-Z][a-zA-Z]+\b"
NGRAM_RANGE = (1, 1)

LDA_MAX_ITER = 100
LDA_EVALUATE_EVERY = 5
LDA_PERP_TOL = 0.1
LDA_MEAN_CHANGE_TOL = 1e-3
LDA_MAX_DOC_UPDATE_ITER = 100

BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 20260821
BOOTSTRAP_SCHEME = "fixed_families_resample_runs_within_family_equal_family_weight"

MODEL_IDENTIFIER_STOPWORDS = {
    "gpt",
    "openai",
    "deepseek",
    "phi",
    "claude",
    "sonnet",
    "anthropic",
    "terra",
}

FAMILY_ORDER = (
    "GPT-4o-mini",
    "DeepSeek-V3",
    "Phi-4",
    "GPT-5.6 Terra",
    "Claude Sonnet",
)

EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_DIMENSION = 3072
EMBEDDING_API_URL = "https://api.openai.com/v1/embeddings"
EMBEDDING_BATCH_SIZE = 64
EMBEDDING_MAX_ATTEMPTS = 8

HIGH_CONFIDENCE_THRESHOLD = 0.50
PAIRWISE_BLOCK_SIZE = 1024
HISTOGRAM_BINS = 20_000
BETWEEN_SAMPLE_PER_TOPIC_PAIR = 20_000
BETWEEN_SAMPLE_SEED = 20260822
