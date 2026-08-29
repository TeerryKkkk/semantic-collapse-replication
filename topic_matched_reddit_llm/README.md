# Topic- and Token-Matched Human–LLM Comparison

This directory contains the reproducibility code for the topic- and token-matched Human–LLM comparison.

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

This preserves the exact 20,000-token Human and model trajectories. The trajectory analysis uses 100 consecutive non-overlapping 200-token windows for cumulative lexical diversity and first-window-anchored within-run semantic diversity. The existing local semantic analysis separately uses 100-token chunks grouped into ten non-overlapping 2,000-token intervals for normalized Vendi and within-interval pairwise cosine similarity and distance.

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

These include 200-token cumulative lexical and anchored within-run trajectory metrics, local Vendi and within-interval pairwise cosine similarity/distance metrics, matched comparisons, model- and thread-level summaries, and question-level paired bootstrap results. In `early_late_vendi.csv`, Early uses intervals 1–3 and Late uses intervals 8–10.

## Tests

Lightweight tests for protocol, routing, Reddit processing, and semantic metrics are included under:

```text
tests/
```

For the complete matching protocol and interpretation of the comparison, see the manuscript.
