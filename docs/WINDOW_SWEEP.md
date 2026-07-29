# Proposal window-width sweep

The controlled experiment compares Standard Top-K SAE, Temporal SAE, and
proposal windows `W = 16, 32, 64`.

All three proposal variants start from
`runs/controlled-pythia160m/checkpoints/shared_initialization.pt`. They use the
same 6,000 optimizer steps and only sample training sequences with at least 64
valid tokens. All common predictor parameters use identical initial values;
each smaller offset-embedding table is the corresponding prefix of the
W=64 table.

The controlled configuration follows the Temporal SAE paper's principal
Pythia setup: Pythia-160m layer 8, 16,384 SAE features, and target k=20.
Standard and proposal conditions use token-wise Top-K; Temporal SAE uses
BatchTopK during training and its learned threshold during evaluation.
The shared initialization receives 12,000 updates and each controlled branch
receives 6,000 updates. The activation cache contains 40,960 training
sequences of length 128 (up to 5,242,880 valid positions) plus 1,024 locked
validation sequences. Method-specific objectives and optimizers are retained.

## Equal training volume

| W | Windows / step | Reconstruction tokens / step | Forecast offsets / window | Forecast pairs / step |
|---:|---:|---:|---:|---:|
| 16 | 32 | 512 | 14 | 448 |
| 32 | 16 | 512 | 28 | 448 |
| 64 | 8 | 512 | 56 | 448 |

Thus every condition processes exactly 3,072,000 reconstruction-token
positions and 2,688,000 forecast pairs. For every W, the forecast offsets used
at each step are sampled uniformly without replacement. The random generator
is seeded and the sampled offsets are sorted before use.

Forecast evaluation uses the same eligible validation sequences and the same
window start positions for every W. The confirmatory forecast summary compares
the common horizon, offsets 1 through 15. Means over every available offset are
also saved as supplementary diagnostics because their horizons differ by W.

## Run

Run the complete comparison, including controls, evaluation, probes, figures,
and the Markdown report:

```bash
git pull origin main
python -m pip install --upgrade -e '.[probe]'
bash scripts/run_controlled.sh
```

The stages can also be resumed independently:

```bash
sae-comp train-controls --config configs/controlled_rtx4090.toml
sae-comp train-window-sweep --config configs/controlled_rtx4090.toml
sae-comp evaluate-controlled --config configs/controlled_rtx4090.toml
sae-comp probe-controlled --config configs/controlled_rtx4090.toml
sae-comp report-controlled --config configs/controlled_rtx4090.toml
```

Probe evaluation retains its completed-job progress in
`evaluation/controlled_probes_progress.json` and resumes if interrupted.
Outputs include:

- `checkpoints/standard.pt`, `temporal.pt`, and `proposal_w016.pt` through
  `proposal_w064.pt`
- `window_sweep_training_history.json`
- `evaluation/controlled_metrics.json` and `.csv`
- `evaluation/controlled_probes.json` and `.csv`
- five PNG figures in `evaluation/plots/`
- `CONTROLLED_REPORT.md`
