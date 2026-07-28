#!/usr/bin/env bash
set -euo pipefail

export TOKENIZERS_PARALLELISM=false
python -m sae_comp.cli run \
  --config configs/smoke.toml \
  --stages extract,train,evaluate
