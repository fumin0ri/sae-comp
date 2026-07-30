#!/usr/bin/env bash
set -euo pipefail

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

TRAINING_SCALE="${TRAINING_SCALE:-1}"
RUN_DIR_OVERRIDE="${RUN_DIR_OVERRIDE:-}"
STANDARD_STEPS="${STANDARD_STEPS:-}"
BRANCH_STEPS="${BRANCH_STEPS:-}"
WARMUP_STEPS="${WARMUP_STEPS:-}"
PREDICTOR_WARMUP_STEPS="${PREDICTOR_WARMUP_STEPS:-}"
PREDICTION_RAMP_STEPS="${PREDICTION_RAMP_STEPS:-}"

python -m sae_comp.cuda_check

args=(
  python -m sae_comp.cli run
  --config configs/controlled_rtx4090.toml
  --stages extract,train-controls,train-window-sweep,saebench,report-saebench
  --training-scale "$TRAINING_SCALE"
)

[[ -n "$RUN_DIR_OVERRIDE" ]] && args+=(--run-dir "$RUN_DIR_OVERRIDE")
[[ -n "$STANDARD_STEPS" ]] && args+=(--standard-steps "$STANDARD_STEPS")
[[ -n "$BRANCH_STEPS" ]] && args+=(--branch-steps "$BRANCH_STEPS")
[[ -n "$WARMUP_STEPS" ]] && args+=(--warmup-steps "$WARMUP_STEPS")
[[ -n "$PREDICTOR_WARMUP_STEPS" ]] && args+=(
  --predictor-warmup-steps "$PREDICTOR_WARMUP_STEPS"
)
[[ -n "$PREDICTION_RAMP_STEPS" ]] && args+=(
  --prediction-ramp-steps "$PREDICTION_RAMP_STEPS"
)

"${args[@]}"
