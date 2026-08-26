# Empirical Unigram Reference for Lexical Accumulation

This directory implements the empirical-frequency IID unigram reference analysis used to evaluate lexical accumulation over extended interaction trajectories.

For each run, the observed lexical trajectory and its unigram reference are constructed from the same preprocessed lexical-token stream. The expected cumulative vocabulary and expected interval-level introduction of previously unseen types are evaluated analytically.

For the motivation, formal definitions, and reported comparisons, see the manuscript and Supplementary Information.

## Files

```text
analyze_unigram_baseline.py
cohort_manifest.csv
requirements.txt
```

`cohort_manifest.csv` defines the transcript files used by the released analysis configuration.

## Installation

From this directory:

```bash
pip install -r requirements.txt
```

## Running

```bash
python analyze_unigram_baseline.py \
    --input-dir /path/to/transcripts \
    --output-dir /path/to/output
```

A different compatible manifest can be supplied with:

```bash
--manifest /path/to/manifest.csv
```

## Outputs

The analysis writes:

```text
run_manifest_audit.csv
run_interval_results.csv
run_summary.csv
model_family_summary.csv
family_balanced_trajectory.csv
overall_summary.csv
```

The module does not require embeddings and contains no figure-generation code.

For the lexical preprocessing rules and analysis interpretation, see the manuscript.
