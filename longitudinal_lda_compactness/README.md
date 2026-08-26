# Longitudinal LDA and Topic Compactness

This directory contains the embedding-independent longitudinal topic analysis and the associated topic-conditioned semantic compactness analysis.

The primary LDA workflow constructs fixed interaction-interval documents, fits a shared topic model, and summarizes changes in topic breadth across interaction trajectories.

A separate compactness workflow assigns individual utterances using the fixed topic model and evaluates semantic distances within and between topic assignments.

For the full analysis design and methodological details, see the manuscript and Supplementary Information.

## Files

The main analysis entry points are:

```text
lda_topic_analysis.py
within_topic_compactness.py
```

Shared configuration and parsing utilities are contained in:

```text
config.py
parse_data.py
utils.py
```

## Installation

```bash
pip install -r requirements.txt
```

## Longitudinal LDA analysis

From this directory:

```bash
python lda_topic_analysis.py \
    --raw-dir /path/to/transcripts \
    --output-dir /path/to/output
```

To run only the primary topic model without the robustness grid:

```bash
python lda_topic_analysis.py \
    --raw-dir /path/to/transcripts \
    --output-dir /path/to/output \
    --skip-robustness
```

The fitted vectorizer and primary LDA model are written under:

```text
<output-dir>/models/
```

and the longitudinal topic-analysis tables are written under:

```text
<output-dir>/lda/
```

## Topic-conditioned semantic compactness

Using the fitted primary model:

```bash
python within_topic_compactness.py \
    --raw-dir /path/to/transcripts \
    --vectorizer /path/to/output/models/count_vectorizer.joblib \
    --lda-model /path/to/output/models/lda_K20_seed42.joblib \
    --output-dir /path/to/output
```

The compactness analysis can also use precomputed embeddings or a resumable embedding cache through the corresponding command-line options.

For the topic-model specification, robustness analyses, and interpretation of the compactness results, see the manuscript.
