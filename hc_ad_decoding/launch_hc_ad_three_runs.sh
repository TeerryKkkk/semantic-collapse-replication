#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
    echo "Usage: $0 RUN_ROOT SEED1 SEED2 SEED3" >&2
    exit 2
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
root="$1"
seeds=("$2" "$3" "$4")
mkdir -p "$root"

for i in 1 2 3; do
    seed="${seeds[$((i - 1))]}"
    "$script_dir/launch_hc_ad.sh" "$root/run_${i}" "$seed"
done
