# Exact-Token Baseline Vendi Analysis

This directory implements the exact-token Vendi analysis used to compare semantic support across early, middle, and late phases of the long-horizon baseline trajectories.

This analysis is separate from the time-resolved utterance-level Vendi analysis in `metrics/compute_embedding_diversity_vendi.py`.

For the exact block construction, rarefaction design, aggregation, bootstrap procedure, and interpretation, see the manuscript and Supplementary Information.

## Structure

```text
config.py
run_pipeline.py
src/
requirements.txt
```

Fixed scientific settings are defined in:

```text
config.py
```

## Installation

```bash
pip install -r requirements.txt
```

## Input data

The pipeline expects the compressed baseline transcripts under:

```text
data/llm/
```

using the filenames defined by the preprocessing module.

Large transcript files and embedding caches are not stored in the repository.

## Running

From this directory:

```bash
python run_pipeline.py
```

If a compatible local embedding cache is present, it is reused. Otherwise the embedding credential required by the pipeline must be available through the environment.

## Outputs

Results are written under:

```text
results/
```

including:

```text
baseline_vendi_per_run.csv
baseline_vendi_summary.csv
```

The first table contains run-level phase scores, and the second contains the aggregated estimates and uncertainty summaries.

For the reported early-to-late comparison and its methodological details, see the manuscript.
