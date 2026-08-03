# Controlled Rectified LpJEPA-SAE comparison

## Research question

Does a local language span contain a sparse residual-stream component shared across
token positions, while a second sparse component preserves position-specific detail?

The proposal follows `fumin0ri/my-sae` commit
`66d8a6f87929a9a415929043863acaa0f14d4207` and architecture
`high_low_rectified_lpjepa_sae_v2_axis_rdm`.

## Compared methods

- Standard token-wise Top-K SAE
- Temporal SAE using the retained Temporal-SAEs objective
- Predictor-free high/low Rectified LpJEPA-SAE at W=2, 4, 8, 16, and 32

All methods use the same frozen Pythia model, residual hook, dictionary width, Pile
document split, and SAEBench configuration. Proposal W conditions additionally share
the same exact random initialization and optimizer budget.

## Proposal architecture

Two distinct positions are sampled without replacement from one random span and
encoded by one shared SAE. The high group uses shifted ReLU and is not Top-K clipped.
The low group uses ReLU plus Top-K. High and low decoder partitions add to form the
full reconstruction.

High codes are trained using direct view invariance and distribution matching to an
i.i.d. Rectified Generalized Gaussian target. Distribution matching combines random
projection sliced 2-Wasserstein distance with axis-aligned 1D 2-Wasserstein distance.
The default axis sample is 512 high coordinates per step.

The exact loss is documented in [WINDOW_SWEEP.md](WINDOW_SWEEP.md). In particular,
the implementation has no predictor, teacher/EMA path, horizon conditioning, or
predicted-code reconstruction loss.

## Validation

The proposal-specific diagnostic reports:

- same-span versus shuffled-sequence high-code cosine by token distance;
- the same-minus-shuffled cosine margin;
- same-span high-code swap FVU versus shuffled high-code swap FVU;
- learned high active fraction relative to the RGG target.

FVU is aggregated as total squared error divided by total centered residual energy,
not as the mean of per-row ratios. Cross-method conclusions use SAEBench core,
sparse probing, SAE-Probes, and RAVEL. TPP and SCR are explicitly disabled.
