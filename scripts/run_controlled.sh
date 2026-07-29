#!/usr/bin/env bash
set -euo pipefail

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -m sae_comp.cli run \
  --config configs/controlled_rtx4090.toml \
  --stages extract,train-controls,train-window-sweep,evaluate-controlled,probe-controlled,report-controlled
