# SAE comparison

`fumin0ri/my-sae` の Offset-conditioned Transition JEPA-SAE を、通常の
Top-K SAE と
[Temporal Sparse Autoencoders](https://github.com/AI4LIFE-GROUP/temporal-saes)
の T-SAE と同一条件で比較する実験コードです。実験自体は行わず、Linux + RTX
4090 で実行できるコードと設定を収録しています。

## 比較する手法

| 手法 | 学習目的 | 時系列信号 |
|---|---|---|
| Standard SAE | 各 residual の Top-K 再構成 | なし |
| Temporal SAE | 20:80 の high/low Matryoshka 再構成 + 対称 contrastive loss | 現在 token と直前 token |
| Transition JEPA-SAE | 再構成 + `z0` から offset ごとの将来 EMA code を予測 | 長さ10の将来 residual trajectory |

提案法は現在位置の sparse code `z0` と offset `k` のみから
`E[zk | z0, k]` に相当する予測可能成分を学習します。介在 token を入力しないため、
決定論的な状態遷移モデルとは解釈しません。

Temporal SAE は添付論文の設定に合わせ、Pythia-160m layer 8、16,384
features、BatchTopK `k=20`、high/low = 20%/80%、temporal loss weight =
1.0 を採用しています。contrastive loss は論文の式どおり、high-level code の
cosine similarity に対する双方向 InfoNCE です。

## 実験設計

3手法は以下を共有します。

- frozen LLM、layer、Pile document、train/validation split
- residual activation shard と正規化統計
- dictionary size、target L0、seed
- 共有の standard SAE 初期 checkpoint
- 分岐後の optimizer update 数
- locked validation と MMLU probe data

初めに standard SAE を12,000 steps学習し、同じ checkpointから3条件へ分岐します。
Standard controlも再構成のみで6,000 steps継続するため、Temporal/JEPAだけが追加の
更新回数を得ることはありません。これは論文の各手法を独立に学習する設定よりも、
「時系列目的を追加した効果」を直接測るための統制を強くした設計です。

既定のactivation cacheは40,960 x 128 = 5,242,880 train token positionsで、
提案実装の既定Pile規模に合わせています。BF16 shardはPythia-160mで約8 GiBです。

## 評価

論文を参考に、全手法へ同じ評価器を適用します。

- FVE
- input/reconstruction cosine similarity
- fraction alive、実測L0
- Lipschitz、Fourier、3-level wavelet、multiscale smoothness
- MMLUのsemantic（subject）、context（question ID）、syntax（spaCy POS）probe
- sparse probes: 1、5、10、20 features/class、およびdense
- Temporal SAEのみhigh/low splitを追加報告

Probeは疎行列上の早期停止logistic-loss SGDで学習します。各probeの完了時に
`probes_progress.json`へ保存するため、中断後に同じコマンドを実行すると続きから
再開します。

提案法についてはoffset 1..9のfuture-code cosine、normalized MSE、
true-context minus shuffled-contextも補助診断として出力します。これは提案法固有の
指標であり、3手法の共通比較とは分けて報告します。

## Linux RTX 4090でのセットアップ

Python 3.11とCUDA 12.1を想定しています。

```bash
git clone https://github.com/fumin0ri/sae-comp.git
cd sae-comp
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -e '.[probe]'
```

既存環境で`No module named 'click'`などspaCyの依存エラーが出た場合は、更新後の
probe extraで環境を修復してください。

```bash
git pull origin main
python -m pip install --upgrade -e '.[probe]'
```

Hugging Faceで認証が必要な場合だけtokenを設定します。

```bash
export HF_TOKEN=...
```

まず小規模なend-to-end確認を実行してください。

```bash
bash scripts/run_smoke.sh
```

本比較は次の1コマンドです。

```bash
bash scripts/run_controlled.sh
```

段階ごとにも実行できます。

```bash
sae-comp extract  --config configs/controlled_rtx4090.toml
sae-comp train    --config configs/controlled_rtx4090.toml
sae-comp evaluate --config configs/controlled_rtx4090.toml
sae-comp probes   --config configs/controlled_rtx4090.toml
sae-comp report   --config configs/controlled_rtx4090.toml
```

途中から再開する場合、完了済みstageを除いて `sae-comp run --stages ...` を使えます。
activation cacheは既存manifestを再利用します。再抽出時のみ `extract
--overwrite` を指定してください。

## 出力

```text
runs/controlled-pythia160m/
  checkpoints/
    shared_initialization.pt
    standard.pt
    temporal.pt
    proposal.pt
  evaluation/
    metrics.json
    metrics.csv
    probes.json
    probes.csv
  training_history.json
  REPORT.md
```

すべてのcheckpointに実験設定、activation fingerprint、参照実装URLを埋め込みます。
Pileはdocument hashで分割するため、同一document由来のtokenがtrainとvalidationに
跨りません。

## 参照した実装と論文

- Proposal: [fumin0ri/my-sae](https://github.com/fumin0ri/my-sae)
- Temporal SAE: [AI4LIFE-GROUP/temporal-saes](https://github.com/AI4LIFE-GROUP/temporal-saes)
- Oesterling et al., “Temporal Sparse Autoencoders: Leveraging the Sequential
  Nature of Language for Interpretability,” ICLR 2026.

実験上の仮説、採否基準、既知の差分は
[`docs/EXPERIMENT_PROTOCOL.md`](docs/EXPERIMENT_PROTOCOL.md) に固定しています。
