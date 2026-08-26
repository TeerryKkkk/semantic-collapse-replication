# Metrics

This directory contains the core lexical and semantic metric implementations used across the interaction analyses.

For the formal metric definitions and their interpretation, see the manuscript and Supplementary Information.

## Scripts

### `compute_lexical_diversity.py`

Computes lexical-diversity and vocabulary-growth quantities from interaction transcripts.

### `compute_within_run_semantic_similarity.py`

Computes longitudinal within-run semantic similarity measures from fixed interaction intervals.

### `compute_cross_run_semantic_similarity.py`

Computes semantic similarity across independent trajectories using aligned interaction intervals.

### `compute_cross_run_semantic_support.py`

Computes cross-run semantic-support quantities from prepared message-level embedding caches.

### `compute_embedding_diversity_vendi.py`

Computes the time-resolved utterance-level normalized Vendi analysis.

Each 1,000-round trajectory is divided into non-overlapping 10-round intervals. Within each interval, 30 individual utterances are sampled without replacement 200 times. Normalized Vendi is computed for each draw from the utterance embedding similarity matrix and averaged across draws.

This script writes analysis tables only and does not generate figures.

## Running the time-resolved Vendi analysis

From the repository root:

```bash
python metrics/compute_embedding_diversity_vendi.py \
    --input-dir /path/to/transcripts \
    --output-dir /path/to/output
```

The script can reuse compatible local embedding caches. If missing embeddings are intentionally to be generated through the configured embedding service, use the explicit API option provided by the script.

Primary outputs are:

```text
input_manifest.csv
vendi_by_run_interval.csv
```

## Other metric scripts

The remaining metric scripts have their own input conventions and configuration settings. See the corresponding source file before running an individual metric pipeline.

The separate exact-token Vendi analysis is located in `vendi_analysis/`.
