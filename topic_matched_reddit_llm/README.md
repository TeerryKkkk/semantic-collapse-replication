# Topic- and Token-Matched Human–LLM Comparison

This directory contains the topic- and token-matched Human–LLM comparison.

The pipeline selects the human discussion threads used in the analysis, generates matched multi-agent continuations, converts both sources to the same fixed-token analysis representation, and computes the semantic-diversity comparison.

For the data source, matching design, simulation protocol, and statistical analysis, see the manuscript and Supplementary Information.

## Installation

```bash
pip install -r requirements.txt
```

## Pipeline

The analysis is organized into four stages.

### 1. Select the human discussions

From this directory:

```bash
python sample_reddit_discussions.py \
    --dataset /path/to/reddit_dataset.jsonl.zst \
    --selection paper
```

The manifest identifying the selected discussion threads is included at:

```text
manifests/paper_threads.csv
```

Selection outputs are written under:

```text
outputs/selection/
```

### 2. Generate matched interaction trajectories

```bash
python run_topic_matched_simulations.py
```

Required model-service credentials are supplied through environment variables as defined in `model_providers.py`.

Generated trajectories are written under:

```text
outputs/runs/
```

### 3. Construct matched semantic-analysis windows

```bash
python prepare_semantic_windows.py
```

This produces the fixed-token chunks and matched analysis windows used for both human and model trajectories.

Outputs are written under:

```text
outputs/analysis_inputs/
```

### 4. Run the semantic-diversity analysis

```bash
python analyze_semantic_diversity.py
```

Analysis outputs are written under:

```text
outputs/semantic_analysis/
```

These include window-level semantic metrics, matched comparisons, early/late summaries, model-level summaries, thread-level summaries, and bootstrap results.

## Tests

Lightweight tests for protocol, routing, Reddit processing, and semantic metrics are included under:

```text
tests/
```

For the complete matching protocol and interpretation of the comparison, see the manuscript.
