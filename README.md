# SAE comparison with SAEBench

Standard Top-K SAE、Temporal SAE、提案手法 Predictor-free Rectified LpJEPA-SAEを
同一のPythia residual stream上で学習し、SAEBench v0.6.0で比較する実験コードです。

提案手法は `fumin0ri/my-sae` commit
`66d8a6f87929a9a415929043863acaa0f14d4207` の
`high_low_rectified_lpjepa_sae_v2_axis_rdm` に対応しています。比較条件は次の7つです。

- Standard Top-K SAE
- Temporal SAE
- Rectified LpJEPA-SAE (`W=2`)
- Rectified LpJEPA-SAE (`W=4`)
- Rectified LpJEPA-SAE (`W=8`)
- Rectified LpJEPA-SAE (`W=16`)
- Rectified LpJEPA-SAE (`W=32`)

## 提案手法

長いdocument-disjoint residual列からspan長 `L~Uniform(2,W)` を選び、span内の
異なる2位置を復元なしのordered samplingで抽出します。2位置は交換可能なviewであり、
context、endpoint、future predictionの区別はありません。

単一のhigh/low SAEが両viewを処理します。

- high: shifted ReLU、Top-Kなし
- low: ReLU + Top-K (`low_k=20`)
- reconstruction: high-only FVU 10% + high+low full FVU 90%
- invariance: 2つのhigh code間の正規化MSE
- RDMReg: random-projection sliced 2-Wasserstein + axis-aligned 1D 2-Wasserstein
- target: Rectified Laplace、target active fraction 0.025
- axis-aligned coordinates: `AXIS_RDM_FEATURES=512`

predictor、horizon/position embedding、teacher encoder、EMA、stop-gradient target、
predicted-residual lossは使用しません。

## RTX 4090環境

```bash
git clone https://github.com/fumin0ri/sae-comp.git
cd sae-comp
conda create -n sae-comp python=3.11 -y
conda activate sae-comp

pip install --index-url https://download.pytorch.org/whl/cu128 \
  torch==2.7.1 torchvision==0.22.1
pip install -c constraints/saebench-py311.txt -e '.[probe,saebench]'
```

CUDA driverがPyTorch wheelより古い場合は、実験前に次を確認してください。

```bash
python -m sae_comp.cuda_check
```

pipが `resolution-too-deep` で停止する場合も、上記のconstraints付きinstallを使用して
ください。

## 実行

```bash
bash scripts/run_controlled.sh
```

実行stageは次の順です。

```text
extract
  -> train-controls
  -> train-window-sweep
  -> saebench
  -> report-saebench
```

`AXIS_RDM_FEATURES`は指定しない場合512です。

```bash
AXIS_RDM_FEATURES=512 bash scripts/run_controlled.sh
```

学習量は倍率または個別stepで変更できます。

```bash
TRAINING_SCALE=2 bash scripts/run_controlled.sh

STANDARD_STEPS=18000 \
BRANCH_STEPS=9000 \
SAE_WARMUP_STEPS=1500 \
REGULARIZATION_RAMP_STEPS=1500 \
RUN_DIR_OVERRIDE=runs/long-run \
bash scripts/run_controlled.sh
```

倍率はoptimizer stepとwarm-up/rampを変更し、batch sizeや1 stepあたりのpair数は
変更しません。異なるbudgetには自動的に別run directory suffixが付きます。

## Controlled defaults

- model: `EleutherAI/pythia-160m-deduped`
- layer: 8 post-residual
- dictionary size: 16,384
- Standard/Temporal target k: 20
- shared Standard pretraining: 12,000 steps
- controlled stage: 6,000 steps
- stored sequence: 128 tokens
- burn-in: 32 tokens
- proposal W: 2, 4, 8, 16, 32
- proposal pairs/step: 512 for every W
- proposal residual values and reconstructions/step: 1,024 for every W
- random RDM projections: 1,024 in chunks of 128
- axis RDM coordinates: 512

Proposalの全Wは同一のランダム初期値、optimizer step数、pair数、2-view再構成数、
activation poolを使います。Wだけがspan内で取り得るtoken距離分布を変えます。

## SAEBench

実行する評価:

- `core`
- `sparse_probing`
- `sparse_probing_sae_probes`
- `ravel`

TPPとSCRはユーザー指定により除外しています。結果は以下へ保存されます。

```text
RUN_DIR/
  checkpoints/
  window_sweep_training_history.json
  saebench_results/
    summary.json
    summary.csv
    plots/*.png
  SAEBENCH_REPORT.md
```

中断後は同じコマンドを再実行できます。既存の完了済みSAEBench結果は再利用されます。
全評価を再計算する場合は設定の `force_rerun = true` を使用してください。

詳細:

- [実験プロトコル](docs/EXPERIMENT_PROTOCOL.md)
- [W sweep](docs/WINDOW_SWEEP.md)
- [SAEBench実行仕様](docs/SAEBENCH.md)

## References

- Proposal: https://github.com/fumin0ri/my-sae
- Temporal SAE: https://github.com/AI4LIFE-GROUP/temporal-saes
- SAEBench: https://github.com/adamkarvonen/SAEBench
