# Confirmatory experiment protocol

## Question

Does a T-SAE-inspired high/low fixed-endpoint JEPA objective separate a
forecastable high-level state from low-level endpoint detail while its final
full-EMA SAE preserves reconstruction better than a reconstruction-only SAE
or a Temporal SAE?

## Prespecified primary outcomes

1. MMLU semantic sparse-probe accuracy at 5 features per class.
2. MMLU context sparse-probe accuracy at 5 features per class.
3. Full-dictionary Lipschitz smoothness.
4. FVE, used as the reconstruction-quality guardrail.

Syntax probe accuracy, the other smoothness measures, fraction alive, L0, and
dense probes are secondary outcomes.

## Success criteria

The proposal is supported only when:

- semantic or context probe accuracy exceeds both baselines;
- FVE is no more than 0.02 below the best reconstruction baseline;
- the result is not explained by feature collapse (alive fraction and L0);
- its forecast true-context cosine exceeds shuffled-context cosine.

Temporal SAE is expected to be the strongest smoothness baseline. A smoother
representation is not by itself evidence of better forecastable state.

## Controls

- one immutable activation manifest for every condition;
- deterministic document-level train/validation separation;
- the same shared standard-SAE initialization;
- equal post-branch update count;
- locked MMLU test questions, used only after SAE training;
- no MMLU labels in any SAE objective;
- all metrics computed by one implementation.

## Method fidelity

The standard condition uses token-wise global Top-K. The proposal partitions
the same total dictionary and L0 budget into 20% high and 80% low blocks and
applies an independent Top-K budget to each. The Temporal condition uses
BatchTopK during training and an EMA threshold at inference. This preserves
each method's intended sparsifier, but means achieved L0 and fraction alive
must be reported rather than assumed equal.

The T-SAE objective follows the paper equation: a 20% high-level dictionary
reconstructs the input, the full dictionary reconstructs the remaining
residual, and a symmetric cosine contrastive loss aligns adjacent-token
high-level codes. It intentionally uses the paper's cosine objective rather
than the unnormalized dot-product logits present in one public training
script.

For endpoint `T=W-1`, the proposal maps every earlier high code and its
position, `P(z_high_k, position(k))`, to the same stop-gradient EMA high code
`z_high_T`. The high block reconstructs the endpoint alone and is the only
forecast-supervised block. The low block receives gradients through cumulative
full reconstruction only:

```text
Lrec = 0.2 * FVU(Dhigh(zhigh), hT)
     + 0.8 * FVU(Dhigh(zhigh) + Dlow(zlow), hT)
```

The predicted high code is decoded through the frozen EMA high decoder. The
entire online high/low SAE is EMA-updated and the final artifact retains both
groups. Variance regularization is excluded.

The window sweep fixes optimizer steps and residual positions read per step.
Because the architecture requires all `W-1` contexts, context-target pair
counts are reported rather than subsampled to artificial equality.

## Replication

The minimum confirmatory replication is three seeds. Stronger evidence uses
Pythia-160m layer 8 plus at least two additional model/layer settings. Each
replication must use a separate run directory and preserve its config and
activation fingerprint.
