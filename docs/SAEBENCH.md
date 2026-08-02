# SAEBench evaluation protocol

## Version and model

API drift による結果差を避けるため、`sae-bench==0.6.0` に固定します。
学習と評価の frozen LLM はともに final checkpoint の
`EleutherAI/pythia-160m-deduped` です。TransformerLens 側の model name は
`pythia-160m-deduped`、hook は `blocks.8.hook_resid_post` です。

比較条件はStandard Top-K、Temporal SAE、random-pair horizon JEPAの最大span
`W=2,4,8,16`です。Proposalは長いdocument-disjoint Pile residual sequenceから
span、endpoint、contextを学習stepごとに抽出します。SAEBenchへ渡すartifactは
high/low分割と独立Top-Kを保持した最終full-EMA SAEです。

Python 3.11ではTransformerLens 2.16.1が`torch>=2.6`を要求します。依存解決には
repositoryの`constraints/saebench-cu128.txt`を使用し、PyTorch 2.7.1 /
torchvision 0.22.1のcu128 wheelとSAEBench依存を固定します。これによりpipが
互換性のないTransformerLens、SAE Lens、torchvisionの組み合わせをbacktrack
し続けることを防ぎます。

## Included evaluations

- `core`: reconstruction、CE loss preservation、sparsity、feature statistics
- `sparse_probing`: SAEBench 標準 sparse probing
- `sparse_probing_sae_probes`: SAE-Probes backend による probing
- `RAVEL`: cause、isolation、disentanglement

TPP と SCR はユーザー指定により除外しています。実装は SAEBench の
`run_all_evals_custom_saes` を使わず、上記4 module だけを直接 import します。
設定の `eval_types` に TPP または SCR を入れた場合も実行前にエラーになります。

Absorption は SAEBench が2B未満のモデルに推奨していないため、Pythia-160M の
本比較には含めません。AutoInterp は外部 API credential が必要で、Unlearning は
この Pythia residual-stream 比較の標準対象ではないため含めません。

RAVEL は `city` の `Country`、`Continent`、`Language` に固定します。
Pythia-160Mでは正答フィルタ後の `nobel_prize_winner/Gender` が単一ラベルに
縮退する場合があり、異なる属性ラベルの介入ペアを構成できないためです。この
city-only設定はSAEBench v0.6.0のRAVEL acceptance testとも一致します。

## Custom SAE adapter

ローカル SAE の再構成は次式です。

```text
z = sparse(encoder((x - pre_bias) / pre_scale))
x_hat = pre_bias + pre_scale * (z @ decoder)
```

SAEBench は unit-normalized decoder vectors を期待します。decoder は変更せず、
adapter が `pre_scale * z` を feature activation として返します。

```text
W_enc = encoder.weight.T / pre_scale
b_enc = encoder.bias
W_dec = decoder
b_dec = pre_bias
feature_acts = pre_scale * sparse((x - b_dec) @ W_enc + b_enc)
x_hat = feature_acts @ W_dec + b_dec
```

この変換は Standard、Temporal、Proposal の全 checkpoint で元の encode/decode
再構成を保存します。Proposalではonline studentではなく、encoder、decoder、
normalization biasを含む完全EMA teacherを変換し、20% high / 80% lowの独立
Top-Kをadapter内でも再現します。Standardはglobal token-wise Top-K、Temporal
SAEは学習済みthresholdを使用します。

## Resume and outputs

各評価は条件ごとに公式形式の JSON を即時保存します。`force_rerun=false` では
既存 JSON を再利用します。各 stage の後に6条件すべてのファイルが存在するか検証し、
欠損があれば完了扱いにしません。

`saebench_results/manifest.json` には version、model、hook、実行対象、除外対象、
checkpoint path、stage status を記録します。`report-saebench` は公式 JSON を読み、
`summary.json`、`summary.csv`、総合ダッシュボードと3種類の詳細グラフ、
`SAEBENCH_REPORT.md` を生成します。

評価途中でも `report-saebench` を実行できます。存在する公式 JSON だけを集計し、
評価ごとの完了条件数と不足条件をレポートに表示します。総合ダッシュボードは
4評価×6条件がすべて揃った時点で生成されます。
