# SAE comparison with SAEBench

Standard Top-K SAE、Temporal SAE、提案手法 Transition JEPA-SAE を同一条件で
学習し、[SAEBench](https://github.com/adamkarvonen/SAEBench) v0.6.0 で比較する
実験コードです。提案手法は `W = 16, 32, 64` の3条件を評価します。

比較する5条件は次のとおりです。

- Standard Top-K SAE
- Temporal SAE
- Transition JEPA-SAE (`W=16`)
- Transition JEPA-SAE (`W=32`)
- Transition JEPA-SAE (`W=64`)

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
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -e '.[saebench]'
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

## 実験条件

- frozen LLM: `EleutherAI/pythia-160m-deduped`
- residual stream: `blocks.8.hook_resid_post`
- context length: 128
- dictionary size: 16,384
- target k: 20
- shared SAE pretraining: 12,000 steps
- branch training: 全条件 6,000 steps
- proposal reconstruction tokens/step: 全Wで512
- proposal forecast pairs/step: 全Wで448

Temporal SAE と3つの提案条件は Standard SAE の共通 checkpoint から分岐します。
提案条件間では初期値、optimizer step 数、再構成 token 数、予測 pair 数を揃えます。

## 出力

```text
runs/saebench-pythia160m-deduped/
  checkpoints/
  saebench_results/
    core/
    sparse_probing/
    sparse_probing_sae_probes/
    ravel/
    plots/
    manifest.json
    summary.json
    summary.csv
  SAEBENCH_REPORT.md
```

詳細は [SAEBench 実行仕様](docs/SAEBENCH.md) と
[window sweep](docs/WINDOW_SWEEP.md) を参照してください。

## 参照

- Proposal: [fumin0ri/my-sae](https://github.com/fumin0ri/my-sae)
- Temporal SAE:
  [AI4LIFE-GROUP/temporal-saes](https://github.com/AI4LIFE-GROUP/temporal-saes)
- Evaluation: [adamkarvonen/SAEBench](https://github.com/adamkarvonen/SAEBench)
