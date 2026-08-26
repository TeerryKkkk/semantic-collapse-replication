# Statistical Analysis

This directory contains the factorwise window-level regression analysis used for intervention comparisons.

The analysis operates on prepared within-run and cross-run metric tables and produces coefficient-level regression results together with multiplicity-adjusted summary tables.

For the statistical model specification, comparison families, and interpretation of the reported intervention effects, see the manuscript and Supplementary Information.

## Main script

```text
run_factorwise_window_regressions.py
```

The script defines its input tables, output directory, analysis windows, factor baselines, and model-group rules in the configuration section near the beginning of the file.

## Inputs

The analysis expects prepared within-run and cross-run metric tables in the schema used by the project analysis pipeline.

Input locations are specified in the script configuration.

## Running

From this directory:

```bash
python run_factorwise_window_regressions.py
```

## Outputs

The script produces tables including:

```text
per_factor_per_model_level_regression.csv
regression_significance_summary.csv
within_window_level_used.csv
cross_window_level_used.csv
within_factor_model_build_status.csv
cross_factor_model_build_status.csv
```

These outputs record the fitted comparisons and the data subsets used to construct them.

For the exact inferential conventions reported in the paper, see the manuscript.
