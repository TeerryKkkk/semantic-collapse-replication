# Multi-LLM Systems Exhibit Robust Semantic Collapse

This repository contains simulation and analysis code accompanying the manuscript:

**Multi-LLM Systems Exhibit Robust Semantic Collapse**

The repository is organized by computational component, including the closed-loop simulation scaffold, lexical and semantic metrics, statistical analyses, human-reference comparisons, representational analyses, and intervention-specific code.

For the full experimental design, data construction, analysis definitions, and interpretation of the reported results, see the manuscript and Supplementary Information.

## Repository structure

| Path | Purpose |
| --- | --- |
| `simulation/` | Closed-loop multi-agent simulation scaffold |
| `metrics/` | Core lexical and semantic metric implementations, including the time-resolved normalized Vendi analysis |
| `statistics/` | Factorwise regression analysis for intervention comparisons |
| `lexical_unigram_reference/` | Empirical unigram reference analysis for lexical accumulation |
| `longitudinal_lda_compactness/` | Longitudinal LDA topic analysis and topic-conditioned semantic compactness analysis |
| `embedding_clustering/` | Embedding clustering, cluster selection, and run/phase association analysis |
| `model_attribution_classifier/` | Independent-reference model-attribution classifier |
| `human_reference_similarity/` | Natural-message Human–LLM semantic comparison from analysis-ready message records |
| `topic_matched_reddit_llm/` | Topic- and token-matched Human–LLM comparison |
| `vendi_analysis/` | Exact-token baseline Vendi analysis across early, middle, and late phases |
| `hc_ad_decoding/` | History-conditioned avoidance decoding intervention |

Each analysis directory contains its own entry points and, where needed, a module-specific `requirements.txt`.

## API credentials

No API credentials are stored in this repository.

Scripts that require external model or embedding services read credentials from environment variables. The required variables depend on the module and backend being used.

## Running the analyses

The analysis modules are designed to be run independently. See the README inside each top-level directory for the relevant entry point, expected inputs, and example invocation.

Where a module contains a `requirements.txt`, install those dependencies in the environment used for that analysis.

## Vendi analyses

Two distinct Vendi analyses are included in the repository.

`metrics/compute_embedding_diversity_vendi.py` implements the time-resolved utterance-level analysis. Within each 10-round interval, utterances are rarefied without replacement to a fixed sample of 30, normalized Vendi is calculated for each of 200 draws, and the draw-level values are averaged.

`vendi_analysis/` implements the separate exact-token baseline analysis used for the early-to-late comparison.

These analyses address different measurement questions and should be treated separately.

## Citation

If you use this code, please cite the accompanying manuscript:

**Multi-LLM Systems Exhibit Robust Semantic Collapse**
