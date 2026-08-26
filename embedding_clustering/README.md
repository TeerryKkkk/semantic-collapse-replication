# Embedding Clustering

This directory implements the embedding clustering analysis and associated cluster-selection and run/phase association summaries described in the manuscript.

Clustering is performed directly in the fixed normalized embedding space using the spherical clustering implementation contained in `src/embedding_clustering/`.

For the scientific motivation, clustering specification, and interpretation of the resulting clusters, see the manuscript and Supplementary Information.

## Structure

```text
data/embedding_manifest.csv
scripts/
src/embedding_clustering/
requirements.txt
```

The repository includes the embedding manifest and clustering code. The large embedding matrix and generated clustering outputs are not included.

## Installation

```bash
pip install -r requirements.txt
```

## Running the clustering analysis

From this directory:

```bash
python scripts/run_embedding_clustering.py \
    --embeddings /path/to/embeddings_l2.npy \
    --manifest data/embedding_manifest.csv \
    --output /path/to/output
```

The input embedding matrix must correspond to the supplied manifest.

## Validation

Input and result verification utilities are available through:

```text
scripts/verify_embedding_clustering.py
```

When the corresponding embedding matrix and locally generated outputs are available, they can be checked with the verification script.

## Generated outputs

The clustering script writes cluster assignments, cluster-selection summaries, family-level summaries, and run/phase association results to the directory specified by `--output`.

For additional methodological details, see the manuscript.
