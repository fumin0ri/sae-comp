# Proposal maximum-span sweep

提案手法は最大span長 `W = 2, 4, 8, 16` の4条件で比較します。全条件は
`runs/saebench-pythia160m-deduped-random-pair-v3/checkpoints/shared_initialization.pt`
から開始します。

## 学習データ生成

Pileのdocumentをtrain/validationへ決定論的に分離し、各documentを128 tokenの長い
連続residual sequenceとして保存します。最初の16 tokenはburn-inとして学習・評価
対象から除外します。旧固定window cacheとはformat IDが異なるため再抽出が必要です。

各proposal sampleでは次をonlineに実行します。

1. `L ~ Uniform(2, W)`を抽出する。
2. 全Wで共通のeligible rangeからendpoint `t`を抽出する。
3. span `[t-L+1,t]` の非endpoint位置からcontext `k`を一様抽出する。
4. `h=t-k`をpredictorへ明示的に渡す。

endpoint rangeは最大条件W=16のhorizon 15を使って固定するため、horizonやWから
sequence境界までの距離を推測できません。

## 等学習量の制御

| Max span W | Pair batch / step | Residual values / step | Endpoint reconstructions / step | Horizon support |
|---:|---:|---:|---:|---:|
| 2 | 512 | 1,024 | 512 | 1 |
| 4 | 512 | 1,024 | 512 | 1..3 |
| 8 | 512 | 1,024 | 512 | 1..7 |
| 16 | 512 | 1,024 | 512 | 1..15 |

各条件は6,000 optimizer stepsで、合計3,072,000 sampled pairsとendpoint
reconstructionsを処理します。span/context samplingが作る非一様な`P(h)`に対し、
latent prediction lossを`1 / ((W-1)P(h))`で重み付けします。これにより各horizonの
期待prediction-loss massが同じになります。再構成lossの重みは変更しません。

最大Wのpredictorを共通初期値として作成し、小さいWではhorizon embeddingの対応する
先頭行をコピーします。共有可能な全パラメータの初期値も一致します。

## 目的関数

```text
z_context = E_online(x_(t-h))
z_target  = stopgrad(E_EMA(x_t))
z_hat_t   = P(z_context_high, h)

L_rec = 0.2 * FVU(D_high(z_t_high), x_t)
      + 0.8 * FVU(D_high(z_t_high)+D_low(z_t_low), x_t)
L = L_rec + lambda_pred * balanced_latent_prediction_loss
```

予測codeをresidualへdecodeした誤差は評価diagnosticだけで、学習lossには含めません。
最初のSAE-only warm-upではprediction weightを0とし、その後rampしてjoint学習します。
online encoder、decoder、pre-bias全体をEMA更新し、完全EMA SAEをSAEBenchへ渡します。

## 評価

Standard Top-K SAE、Temporal SAE、4つのproposalを同じcustom SAE interfaceと
SAEBench設定で評価します。TPPとSCRは実行しません。

```bash
python -m pip install --upgrade \
  -c constraints/saebench-cu128.txt \
  -e '.[saebench]'
bash scripts/run_controlled.sh
```
