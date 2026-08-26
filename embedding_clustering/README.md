# Embedding Clustering

This directory implements the embedding clustering analysis and associated cluster-selection and run/phase association summaries described in the manuscript.

Clustering is performed directly in the fixed normalized embedding space using the spherical clustering implementation contained in `src/embedding_clustering/`.

For the scientific motivation, clustering specification, and interpretation of the resulting clusters, see the manuscript and Supplementary Information.

## Structure

```text
data/
scripts/
src/embedding_clustering/
results/embedding_clustering/public/
requirements.txt
```

The repository includes the embedding manifest and lightweight public result summaries. The large embedding matrix itself is not stored in the repository.

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
    --output results/embedding_clustering/public
```

The input embedding matrix must correspond to the supplied manifest.

## Validation

Input and result verification utilities are available through:

```text
scripts/verify_embedding_clustering.py
```

When the corresponding embedding matrix is available, the released summaries can be checked with the verification script.

## Public outputs

Lightweight analysis outputs included in the repository are stored under:

```text
results/embedding_clustering/public/
```

These include cluster assignments, cluster-selection summaries, family-level summaries, and run/phase association results.

For additional methodological details, see the manuscript.
