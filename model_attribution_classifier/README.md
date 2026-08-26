# Model-Attribution Classifier

This directory contains the independent-reference model-attribution classifier used to test whether interaction outputs retain information about the generating base-model family.

The primary workflow trains the classifier on an independent reference set and evaluates the frozen classifier on separate long-horizon interaction trajectories.

For the classifier design, training/test construction, weighting scheme, and interpretation, see the manuscript and Supplementary Information.

## Main entry point

From the repository root:

```bash
python -m model_attribution_classifier.run_independent_reference \
    --manifest /path/to/manifest.csv \
    --output-dir /path/to/output
```

## Manifest format

The manifest must contain:

```text
role
model_family
run_id
transcript_path
```

The `role` field identifies reference and test trajectories.

The script validates the expected independent-reference design before fitting.

## Preparing inputs without fitting

Manifest, transcript, and weighting preparation can be checked without embedding calls or model fitting:

```bash
python -m model_attribution_classifier.run_independent_reference \
    --manifest /path/to/manifest.csv \
    --output-dir /path/to/output \
    --prepare-only
```

## Embeddings

The classifier uses the embedding representation defined in the analysis code.

A resumable exact-text embedding cache can be supplied with:

```bash
--cache-path /path/to/cache.sqlite3
```

If required embeddings are not already cached, the relevant API credential must be provided through the environment.

## Outputs

The output directory contains the generated embedding cache, classifier artifact unless disabled, prediction tables, and summary metadata.

For the reported evaluation metrics and longitudinal analyses, see the manuscript.
