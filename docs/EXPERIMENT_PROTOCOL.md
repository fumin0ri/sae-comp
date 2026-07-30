# Confirmatory experiment protocol

## Question

Does an all-context, fixed-endpoint Transition JEPA objective reshape a sparse
dictionary so that its final full-EMA SAE preserves reconstruction while
exposing more sequence-level semantic and contextual state than a
reconstruction-only SAE or a Temporal SAE?

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

The standard and proposal conditions use token-wise Top-K as in
`fumin0ri/my-sae`. The Temporal condition uses BatchTopK during training and an
EMA threshold at inference as in the T-SAE reference. This preserves each
method's intended sparsifier, but means the achieved L0 and fraction alive
must be reported rather than assumed equal.

The T-SAE objective follows the paper equation: a 20% high-level dictionary
reconstructs the input, the full dictionary reconstructs the remaining
residual, and a symmetric cosine contrastive loss aligns adjacent-token
high-level codes. It intentionally uses the paper's cosine objective rather
than the unnormalized dot-product logits present in one public training
script.

For a window endpoint `T=W-1`, the proposal predictor separately maps every
earlier code and its absolute context position,
`P(z_k, position(k))`, to the same stop-gradient full-EMA endpoint code `z_T`.
It does not pool contexts or receive intervening tokens. The online SAE
reconstructs only the endpoint. The sparse predicted code is decoded through
the frozen EMA decoder, and the final downstream artifact is the EMA encoder,
decoder, and normalization bias. Variance regularization is excluded.

The window sweep fixes optimizer steps and residual positions read per step.
Because the architecture requires all `W-1` contexts, context-target pair
counts are reported rather than subsampled to artificial equality.

## Replication

The minimum confirmatory replication is three seeds. Stronger evidence uses
Pythia-160m layer 8 plus at least two additional model/layer settings. Each
replication must use a separate run directory and preserve its config and
activation fingerprint.
