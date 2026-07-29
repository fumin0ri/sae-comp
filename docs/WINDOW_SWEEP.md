# Proposal window-width sweep

提案手法は `W = 16, 32, 64` の3条件で比較します。全条件は
`runs/saebench-pythia160m-deduped/checkpoints/shared_initialization.pt`
から開始します。

## 等学習量の制御

| W | Windows / step | Reconstruction tokens / step | Forecast offsets / window | Forecast pairs / step |
|---:|---:|---:|---:|---:|
| 16 | 32 | 512 | 14 | 448 |
| 32 | 16 | 512 | 28 | 448 |
| 64 | 8 | 512 | 56 | 448 |

各条件は6,000 optimizer steps を実行するため、いずれも3,072,000
reconstruction-token positions と2,688,000 forecast pairs を処理します。
全条件が64 token 以上の同じ training sequence pool を使います。

最大Wの predictor を共通初期値として作成し、小さいWでは offset embedding の
先頭部分をコピーします。これにより共有可能なパラメータの初期値も一致します。

## 評価

Standard Top-K SAE、Temporal SAE、3つのWを同じカスタム SAE interface に変換し、
同一の SAEBench 呼び出しに渡します。SAEBench adapter は decoder の unit norm を
保ったまま activation normalization を feature scale に移すため、元 checkpoint と
再構成結果が一致します。

実行する評価は `core`、`sparse_probing`、
`sparse_probing_sae_probes`、`RAVEL` です。TPP と SCR は実行しません。

```bash
python -m pip install --upgrade -e '.[saebench]'
bash scripts/run_controlled.sh
```
