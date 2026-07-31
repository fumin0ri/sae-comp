# SAE comparison with SAEBench

Standard Top-K SAE、Temporal SAE、提案手法 Transition JEPA-SAE を同一条件で
学習し、[SAEBench](https://github.com/adamkarvonen/SAEBench) v0.6.0 で比較する
実験コードです。提案手法は `W = 16, 32, 64` の3条件を評価します。

提案手法は `fumin0ri/my-sae` commit
`945a5aa54ab955064e8ed50cdcaefcc2a71fed16` の
`hierarchical_high_low_fixed_endpoint_ema_sae_v1` を実装しています。辞書幅と
Top-K budgetを20% high / 80% lowへ分け、それぞれ独立Top-Kを適用します。窓内の
各context位置 `k=0,...,W-2` のhigh codeから共通終端 `T=W-1` のEMA high codeを
予測し、low groupはfull endpoint再構成へdetailを加えます。最終評価にはhigh/low
分割を保持した完全EMA teacherを使用します。

比較する5条件は次のとおりです。

- Standard Top-K SAE
- Temporal SAE
- Hierarchical high/low Transition JEPA-SAE (`W=16`)
- Hierarchical high/low Transition JEPA-SAE (`W=32`)
- Hierarchical high/low Transition JEPA-SAE (`W=64`)

最新版upstreamにはunsplit JEPA baselineもありますが、このrepositoryでは指定された
5条件の比較を維持し、3つのProposal条件をすべて最新版のhierarchical提案法として
評価します。

SAEBench の `core`、`sparse_probing`、`sparse_probing_sae_probes`、`RAVEL`
を実行します。TPP と SCR は設定の allowlist から除外しており、指定すると
設定検証で停止します。

## セットアップ

Linux、Python 3.11、CUDA 対応 PyTorch を想定しています。

```bash
git clone https://github.com/fumin0ri/sae-comp.git
cd sae-comp
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install \
  torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install \
  -c constraints/saebench-cu128.txt \
  -e '.[saebench]'
python -m pip check
python -m sae_comp.cuda_check
```

SAEBench 0.6.0 が使用する TransformerLens 2.16.1 は Python 3.11 で
`torch>=2.6` を要求するため、PyTorch は `2.7.1` に固定しています。RTX 4090機の
CUDA 12.8 driverに対応するcu128 wheelを先に導入し、その後に制約ファイルを
適用してください。制約なしの `pip install -e '.[saebench]'` はSAEBenchの広い
依存範囲を長時間backtrackし、`resolution-too-deep` になることがあります。

既存環境を修復する場合も同じ順序で実行します。

```bash
python -m pip install --force-reinstall \
  torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install \
  -c constraints/saebench-cu128.txt \
  -e '.[saebench]'
python -m pip check
python -m sae_comp.cuda_check
```

必要なら Hugging Face の token を設定してください。

```bash
export HF_TOKEN=...
```

## 実行

RTX 4090 向けの比較全体は1コマンドで実行できます。

```bash
bash scripts/run_controlled.sh
```

実行順は以下です。

```text
extract
  -> train-controls
  -> train-window-sweep
  -> saebench
  -> report-saebench
```

途中から再開する場合は、必要な stage だけ実行できます。

```bash
sae-comp saebench --config configs/controlled_rtx4090.toml
sae-comp report-saebench --config configs/controlled_rtx4090.toml
```

SAEBench の既存結果は再利用されます。全評価を再計算する場合は
`configs/controlled_rtx4090.toml` の `force_rerun` を `true` に変更します。

## 学習量の変更

設定ファイルを編集せず、実験全体の学習stepとscheduleを倍率指定できます。

```bash
TRAINING_SCALE=2 bash scripts/run_controlled.sh
```

この例ではshared SAE pretraining、全branch、optimizer warm-up、proposal predictor
warm-up、prediction rampがすべて2倍になります。batch sizeと1 stepあたりの
residual position数は変わらないため、手法間とW間の比較条件は維持されます。
出力先は自動的に次へ分離されます。

```text
runs/saebench-pythia160m-deduped-hierarchical-v1-trainx2/
```

step数を個別に指定することもできます。

```bash
STANDARD_STEPS=30000 \
BRANCH_STEPS=15000 \
WARMUP_STEPS=1200 \
PREDICTOR_WARMUP_STEPS=2000 \
PREDICTION_RAMP_STEPS=2000 \
RUN_DIR_OVERRIDE=runs/sae-custom-budget \
bash scripts/run_controlled.sh
```

同じ指定はCLIからも利用できます。個別step指定は`--training-scale`より優先されます。

```bash
sae-comp run \
  --config configs/controlled_rtx4090.toml \
  --stages extract,train-controls,train-window-sweep,saebench,report-saebench \
  --training-scale 2
```

別stageを後から実行するときは、同じ倍率または同じ個別overrideを渡してください。
学習予算が変わり`--run-dir`を省略した場合は、予算固有のsuffixが自動付与されるため、
異なる学習量のcheckpointやSAEBench結果は混ざりません。

## 実験条件

- frozen LLM: `EleutherAI/pythia-160m-deduped`
- residual stream: `blocks.8.hook_resid_post`
- context length: 128
- dictionary size: 16,384
- target k: 20
- shared SAE pretraining: 12,000 steps
- branch training: 全条件 6,000 steps
- proposal residual positions/step: 全Wで512
- proposal contexts/window: `W-1`（全位置を使用）
- proposal dictionary: high 20% / low 80%
- proposal Top-K: high 20% / low 80%へ独立配分
- proposal reconstruction: high-only FVU 0.2 + full FVU 0.8

Temporal SAE と3つの提案条件は Standard SAE の共通 checkpoint から分岐します。
提案条件間では共有可能な初期値、optimizer step 数、読み込む residual position
数を揃えます。全contextを必ず使う新版の定義に従うため、context-target pair数は
Wごとに480、496、504/stepです。予測損失は全pairの平均なので勾配scaleは揃います。

## 出力

```text
runs/saebench-pythia160m-deduped-hierarchical-v1/
  checkpoints/
  saebench_results/
    core/
    sparse_probing/
    sparse_probing_sae_probes/
    ravel/
    plots/
      overview.png
      core.png
      probing.png
      ravel.png
    manifest.json
    summary.json
    summary.csv
  SAEBENCH_REPORT.md
```

`SAEBENCH_REPORT.md` には総合ダッシュボード、指標ごとの最高点条件、全数値表、
詳細グラフが自動的にまとめられます。

詳細は [SAEBench 実行仕様](docs/SAEBENCH.md) と
[window sweep](docs/WINDOW_SWEEP.md) を参照してください。

## 参照

- Proposal: [fumin0ri/my-sae](https://github.com/fumin0ri/my-sae)
- Temporal SAE:
  [AI4LIFE-GROUP/temporal-saes](https://github.com/AI4LIFE-GROUP/temporal-saes)
- Evaluation: [adamkarvonen/SAEBench](https://github.com/adamkarvonen/SAEBench)
