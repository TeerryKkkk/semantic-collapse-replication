# History-Conditioned Avoidance Decoding

This directory contains the history-conditioned avoidance decoding intervention used in the manuscript.

The intervention is integrated into the standard conversational simulation scaffold while replacing the free-agent generation step with candidate generation and history-conditioned avoidance scoring.

For the intervention definition, avoidance-history construction, candidate selection, and experimental settings, see the manuscript and Supplementary Information.

## Main files

```text
run_hc_ad.py
hc_ad_decoder.py
avoidance_history.py
hc_ad_integration.py
standard_run.py
```

`standard_run.py` provides the bundled host simulation used by the intervention runner.

## Installation

This module has its own dependency set:

```bash
pip install -r requirements.txt
```

A compatible GPU environment is required for the local generation backend used by this implementation.

## Running one replicate

From this directory:

```bash
./launch_hc_ad.sh /path/to/new_run_directory REPLICATE_SEED
```

The run directory must be a new path.

## Running three replicates

```bash
./launch_hc_ad_three_runs.sh \
    /path/to/run_root \
    SEED1 \
    SEED2 \
    SEED3
```

Each replicate is written to a separate run directory.

## Credentials and runtime data

Credentials required by the host simulation are supplied through environment variables.

Runtime transcripts, retrieval data, and memory records are written inside the selected run directory and are not committed to the repository.

For the full HC-AD configuration and analysis of the intervention, see the manuscript.
