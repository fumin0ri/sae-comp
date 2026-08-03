# SAEBench evaluation protocol

## Conditions

SAEBench compares Standard Top-K, Temporal SAE, and Rectified LpJEPA at maximum
span lengths W=2, 4, 8, 16, and 32. Proposal checkpoints contain the single trained
high/low SAE; there is no EMA export or predictor artifact.

## Evaluations

- `core`: reconstruction quality, CE-loss preservation, and observed L0
- `sparse_probing`: standard SAEBench sparse probing
- `sparse_probing_sae_probes`: SAE-Probes
- `ravel`: city attributes Country, Continent, and Language

TPP and SCR are explicitly excluded. Absorption is not enabled for Pythia-160M.

## Custom adapter

The adapter preserves each local activation rule exactly:

- Standard: token-wise global Top-K
- Temporal: learned inference threshold
- Rectified LpJEPA: shifted-ReLU high coordinates without Top-K and Top-K low
  coordinates with `low_k=20`

Local decoders are unit normalized and local decoding multiplies by `pre_scale`.
The adapter instead multiplies encoded feature activations by `pre_scale`, leaving
the unit decoder unchanged. This is a lossless reparameterization of reconstruction.

## Installation

```bash
pip install --index-url https://download.pytorch.org/whl/cu128 \
  torch==2.7.1 torchvision==0.22.1
pip install -c constraints/saebench-py311.txt -e '.[probe,saebench]'
```

The constraints keep the SAEBench v0.6.0, TransformerLens, SAE-Lens, torchvision,
and Python 3.11 dependency graph bounded and avoid pip `resolution-too-deep`.

## Resume and report

SAEBench result JSON files are reused unless `force_rerun=true`. The report stage can
produce a partial report while some evaluations are missing and lists the exact
missing condition/evaluation pairs. Final summary tables and graphs are written to
`saebench_results/` and `SAEBENCH_REPORT.md`.
