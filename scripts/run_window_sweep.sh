#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/controlled_rtx4090.toml}"

python -m sae_comp.cli run \
  --config "${CONFIG}" \
  --stages train-window-sweep,evaluate-window-sweep,probe-window-sweep,report-window-sweep
