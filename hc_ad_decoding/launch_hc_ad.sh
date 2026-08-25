#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "Usage: $0 NEW_RUN_DIR REPLICATE_SEED" >&2
    exit 2
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

python "$script_dir/run_hc_ad.py" \
    --run-dir "$1" \
    --model-name meta-llama/Meta-Llama-3-8B-Instruct \
    --quantization nf4 \
    --rounds 200 \
    --seed "$2"
