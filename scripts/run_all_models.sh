#!/usr/bin/env bash
# Quick local loop: train every registered baseline on one config, sequentially.
#
#   ./scripts/run_all_models.sh [config.yaml] [extra.override=value ...]
#
# For parallel / cluster dispatch use scripts/sweep.py instead.
set -euo pipefail

CONFIG="${1:-experiments/experiment.yaml}"
shift || true
EXTRA=("$@")

MODELS=(mlp lstm dlinear)
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

for M in "${MODELS[@]}"; do
    echo "=================  training: ${M}  ================="
    python scripts/train.py -c "$CONFIG" "model.name=${M}" "run.name=quick_${M}" "${EXTRA[@]}"
done
