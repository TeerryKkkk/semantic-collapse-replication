# Natural-Message Human–LLM Comparison

This directory implements the natural-message semantic comparison between human discussion trajectories and LLM interaction trajectories.

The analysis begins from analysis-ready message records: one complete human comment or model utterance per row, in the intended chronological or generation order.

Source-data construction, cohort definitions, and the corresponding methodological choices are described in the manuscript and Supplementary Information.

## Files

```text
complete_message_similarity.py
requirements.txt
```

## Installation

```bash
pip install -r requirements.txt
```

## Input format

The `compute` command expects a UTF-8 CSV with the following columns:

```text
source
unit_id
model_family
message_index
text
```

Each row must contain one complete message. Input messages should already be cleaned and ordered before this analysis is run.

## Step 1: compute message-level semantic trajectories

```bash
python complete_message_similarity.py compute \
    --messages /path/to/messages.csv \
    --output /path/to/unit_similarity.csv
```

This step embeds complete messages and computes semantic similarity relative to the first message in each trajectory.

If embeddings must be generated, the API credential is read from the environment variable specified by the script.

## Step 2: summarize the comparison

```bash
python complete_message_similarity.py summarize \
    --units /path/to/unit_similarity.csv \
    --output-dir /path/to/output
```

This step produces the longitudinal comparison, family-level summaries, token-length diagnostics, message-length sensitivity analysis, and bootstrap summaries.

## Outputs

The summary command writes analysis files including:

```text
complete_message_curve.csv
complete_message_family_curves.csv
complete_message_token_length_diagnostics.csv
message_length_sensitivity.csv
```

For the source dataset, cohort construction, analysis horizon, and statistical interpretation, see the manuscript.
