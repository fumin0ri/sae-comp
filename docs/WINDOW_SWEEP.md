# Proposal window-width sweep

This experiment compares proposal windows `W = 8, 16, 32, 64, 128` without
retraining the Standard SAE or Temporal SAE controls.

All five proposal variants start from
`runs/controlled-pythia160m/checkpoints/shared_initialization.pt`. They use the
same 6,000 optimizer steps and only sample training sequences with at least 128
valid tokens. All common predictor parameters use identical initial values;
each smaller offset-embedding table is the corresponding prefix of the
W=128 table.

## Equal training volume

| W | Windows / step | Reconstruction tokens / step | Forecast offsets / window | Forecast pairs / step |
|---:|---:|---:|---:|---:|
| 8 | 64 | 512 | 7 | 448 |
| 16 | 32 | 512 | 14 | 448 |
| 32 | 16 | 512 | 28 | 448 |
| 64 | 8 | 512 | 56 | 448 |
| 128 | 4 | 512 | 112 | 448 |

Thus every condition processes exactly 3,072,000 reconstruction-token
positions and 2,688,000 forecast pairs. For `W > 8`, the forecast offsets used
at each step are sampled uniformly without replacement. The random generator
is seeded and the sampled offsets are sorted before use.

Forecast evaluation uses the same eligible validation sequences and the same
window start positions for every W. The confirmatory forecast summary compares
the common horizon, offsets 1 through 7. Means over every available offset are
also saved as supplementary diagnostics because their horizons differ by W.

## Run

After the controlled experiment has produced the activation cache, shared
initialization, and MMLU probe cache:

```bash
git pull origin main
python -m pip install --upgrade -e '.[probe]'
bash scripts/run_window_sweep.sh
```

The stages can also be resumed independently:

```bash
sae-comp train-window-sweep --config configs/controlled_rtx4090.toml
sae-comp evaluate-window-sweep --config configs/controlled_rtx4090.toml
sae-comp probe-window-sweep --config configs/controlled_rtx4090.toml
sae-comp report-window-sweep --config configs/controlled_rtx4090.toml
```

Probe evaluation retains its completed-job progress in
`evaluation/window_sweep_probes_progress.json` and resumes if interrupted.
Outputs include:

- `checkpoints/proposal_w008.pt` through `proposal_w128.pt`
- `window_sweep_training_history.json`
- `evaluation/window_sweep.json` and `.csv`
- `evaluation/window_sweep_probes.json` and `.csv`
- `WINDOW_SWEEP_REPORT.md`
