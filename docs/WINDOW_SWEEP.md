# Proposal window-width sweep

提案手法は `W = 16, 32, 64` の3条件で比較します。全条件は
`runs/saebench-pythia160m-deduped-hierarchical-v1/checkpoints/shared_initialization.pt`
から開始します。

## 等学習量の制御

| W | Windows / step | Residual positions / step | Contexts / window | Context-target pairs / step |
|---:|---:|---:|---:|---:|
| 16 | 32 | 512 | 15 | 480 |
| 32 | 16 | 512 | 31 | 496 |
| 64 | 8 | 512 | 63 | 504 |

各条件は6,000 optimizer steps、合計3,072,000 residual positionsを処理します。
全条件が64 token以上の同じtraining sequence poolを使います。新版は窓内の
全context位置から同じ固定終端を予測するため、pairを旧方式のようにsubsample
しません。したがってpair数はWによってわずかに異なりますが、損失は全pairの
平均です。

最大Wの predictor を共通初期値として作成し、小さいWでは position embedding の
先頭部分をコピーします。これにより共有可能なパラメータの初期値も一致します。

## 提案手法のhigh/low固定終端目的

`T=W-1` とすると、各 `k=0,...,T-1` について
`P(z_high_k, position(k))` が同じstop-gradient EMA target `z_high_T` を予測します。
辞書と総Top-Kを20% high / 80% lowへ分けて独立Top-Kを適用します。high-only
終端再構成へ0.2、high+lowのfull再構成へ0.8を与え、lowには予測lossを与えません。
予測high codeはfrozen EMA high decoderでresidualへ戻します。joint phase後は
high/low全体のonline encoder、decoder、pre-biasをEMA更新し、その完全EMA SAEを
SAEBenchへ渡します。variance regularizerは使用しません。

## 評価

Standard Top-K SAE、Temporal SAE、3つのWの階層型完全EMA SAEを同じcustom SAE
interfaceに変換し、
同一の SAEBench 呼び出しに渡します。SAEBench adapter は decoder の unit norm を
保ったまま activation normalization を feature scale に移すため、元 checkpoint と
再構成結果が一致します。Proposal adapterはhigh/lowの独立Top-Kも保持します。

実行する評価は `core`、`sparse_probing`、
`sparse_probing_sae_probes`、`RAVEL` です。TPP と SCR は実行しません。

```bash
python -m pip install --upgrade \
  -c constraints/saebench-cu128.txt \
  -e '.[saebench]'
bash scripts/run_controlled.sh
```
