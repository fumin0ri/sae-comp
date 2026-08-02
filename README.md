# SAE comparison with SAEBench

Standard Top-K SAE、Temporal SAE、提案手法 Transition JEPA-SAE を同一条件で
学習し、[SAEBench](https://github.com/adamkarvonen/SAEBench) v0.6.0 で比較する
実験コードです。提案手法は最大span長 `W = 2, 4, 8, 16` の4条件を評価します。

提案手法は `fumin0ri/my-sae` commit
`39ca51a320afbab48486f38594768d37fc68c0dc` の
`high_low_random_pair_horizon_ema_sae_v3` を実装しています。辞書幅と
Top-K budgetを20% high / 80% lowへ分け、それぞれ独立Top-Kを適用します。長い連続
residual列からspan長 `L~Uniform(2,W)`、endpoint `t`、span内の非endpoint context
`k`をランダム抽出し、明示的なtoken距離 `h=t-k` とonline high codeからEMA endpoint
high codeを予測します。low groupはfull endpoint再構成へdetailを加えます。

比較する6条件は次のとおりです。

- Standard Top-K SAE
- Temporal SAE
- Random-pair high/low Transition JEPA-SAE (`W=2`)
- Random-pair high/low Transition JEPA-SAE (`W=4`)
- Random-pair high/low Transition JEPA-SAE (`W=8`)
- Random-pair high/low Transition JEPA-SAE (`W=16`)

Proposalの4条件は同じ長いdocument-disjoint sequence、初期値、optimizer step数、
pair batch、endpoint再構成数を使います。Wは保存済みwindow境界ではありません。

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

この例ではshared SAE pretraining、全branch、optimizer warm-up、proposal SAE-only
warm-up、prediction rampがすべて2倍になります。batch sizeと1 stepあたりの
sampled pair数は変わらないため、手法間とW間の比較条件は維持されます。
出力先は自動的に次へ分離されます。

```text
runs/saebench-pythia160m-deduped-random-pair-v3-trainx2/
```

step数を個別に指定することもできます。

```bash
STANDARD_STEPS=30000 \
BRANCH_STEPS=15000 \
WARMUP_STEPS=1200 \
SAE_WARMUP_STEPS=4000 \
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
- stored residual sequence: 128 token、16 token burn-in
- proposal sampled pairs/step: 全Wで512
- proposal endpoint reconstructions/step: 全Wで512
- proposal span: `L~Uniform(2,W)`、contextはspan内で一様抽出
- proposal horizon: `1..W-1`、prediction lossを`1/P(h)`で補正
- proposal dictionary: high 20% / low 80%
- proposal Top-K: high 20% / low 80%へ独立配分
- proposal reconstruction: high-only FVU 0.2 + full FVU 0.8

Temporal SAE と4つの提案条件は Standard SAE の共通 checkpoint から分岐します。
提案条件間では共有可能な初期値、optimizer step数、sampled pair数、endpoint再構成
数を揃えます。random span/context生成では短いhorizonが多くなるため、各horizonの
期待prediction-loss massが等しくなるinverse-probability weightingを適用します。

## 出力

```text
runs/saebench-pythia160m-deduped-random-pair-v3/
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
