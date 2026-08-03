# Rectified LpJEPA maximum-span sweep

## Conditions

The proposal is evaluated at maximum span lengths `W = 2, 4, 8, 16, 32`.
Every W uses the same random initialization, optimizer-step count, pair batch,
long-sequence activation pool, and two-view reconstruction budget.

## Online view sampling

For each training pair:

1. sample `L ~ UniformInteger(2, W)`;
2. sample a span end from a W-independent safe endpoint range;
3. sample two distinct ordered positions uniformly without replacement from the span;
4. read the two residual vectors as exchangeable views `(h_a, h_b)`.

Neither view is an endpoint target. View A and view B therefore have identical
marginal position distributions. The common `burn_in_tokens=32` and W=32 boundary
support prevent a model from identifying a condition or view through padding or
sequence-boundary proximity.

## Architecture and objective

The same high/low encoder-decoder processes both views. High features use shifted
ReLU without Top-K. Low features use ReLU plus Top-K.

```text
(z_a^H, z_a^L) = E(h_a)
(z_b^H, z_b^L) = E(h_b)

L = (1-lambda_H) L_full-rec
  + lambda_H L_high-rec
  + lambda_inv L_invariance
  + lambda_rdm (L_random-RDM + lambda_axis L_axis-RDM)
```

- `L_full-rec`: FVU of high+low reconstruction for both views
- `L_high-rec`: FVU of high-only reconstruction for both views
- `L_invariance`: target-second-moment-normalized MSE between the paired high codes
- `L_random-RDM`: normalized random-projection sliced 2-Wasserstein distance
- `L_axis-RDM`: normalized coordinate-wise 1D 2-Wasserstein distance

The RDM target is an independent Rectified Generalized Gaussian product distribution.
The primary configuration uses Rectified Laplace (`p=1`) with target active fraction
0.025. `AXIS_RDM_FEATURES=512` samples up to 512 high coordinates without replacement
per step.

There is no predictor, horizon/position embedding, teacher encoder, EMA target,
stop-gradient target, contrastive negative, horizon weighting, variance loss, or
predicted-residual loss.

RDMReg ramps from the first optimizer step. Direct invariance is zero during the SAE
distribution warm-up and ramps after `SAE_WARMUP_STEPS`.

## Budget

With the controlled configuration, every W uses 512 exchangeable pairs, 1,024
residual values, and 1,024 reconstruction targets per optimizer step. W changes only
the distribution of admissible within-span token distances.

Standard Top-K, Temporal SAE, and all five Rectified LpJEPA conditions are evaluated
with the same SAEBench configuration. TPP and SCR remain excluded.
