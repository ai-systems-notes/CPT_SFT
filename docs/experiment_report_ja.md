# 実験レポート: Qwen3-0.6B CPT + LoRA-SFT

## 評価範囲

完了した工程:

- Qwen3-0.6B-Baseのfull-parameter CPTをTransformers/PyTorchとUnslothで実行
- 2回のfull CPT比較
- Baseと2種類のCPT checkpointへ同一条件のPEFT LoRA-SFTを実行
- 6条件それぞれからQA 200行を生成し、計1,200回答を保存
- loss、throughput、GPU memory、回答長、keyword recallを保存

未実施の工程:

- QA参照回答200行の全件一次情報検証
- 外部LLM-as-a-Judge
- 事実精度に基づくモデル順位付け

## CPT実測

| Run | Framework | tokens/s | peak allocated | `nvidia-smi` peak | train time | held-out loss |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | Transformers/PyTorch | 4,754.61 | 8,474 MiB | 9,450 MiB | 321.18秒 | 2.5403 → 2.1455 |
| 1 | Unsloth | 4,761.86 | 7,673 MiB | 8,060 MiB | 320.55秒 | 2.5402 → 2.1455 |
| 2 | Transformers/PyTorch | 4,981.27 | 8,474 MiB | 9,450 MiB | 305.72秒 | 2.5403 → 2.1456 |
| 2 | Unsloth | 4,778.68 | 7,673 MiB | 8,060 MiB | 319.12秒 | 2.5402 → 2.1455 |

2回ともheld-out lossは低下しました。Unslothはpeak allocatedで約801 MiB、`nvidia-smi` peakで約1,390 MiB少ない値でした。速度差はrun間で逆転しており、このRTX 4070環境では明確な速度優位を確認できませんでした。

## LoRA-SFT実測

| Source model | loss | train time | tokens/s | peak allocated |
| --- | ---: | ---: | ---: | ---: |
| Base | 0.6172 | 10.36秒 | 2,871.27 | 2,519 MiB |
| Transformers-CPT | 0.6135 | 10.24秒 | 2,905.33 | 2,519 MiB |
| Unsloth-CPT | 0.6126 | 10.12秒 | 2,939.11 | 2,519 MiB |

各条件は同じ200件、3 epochs、seed 42、39 optimizer stepsです。SFTはPEFT LoRAであり、full fine-tuningではありません。

## QA形式・字句指標

| Condition | 100文字以内 | 平均文字数 | keyword recall |
| --- | ---: | ---: | ---: |
| Base | 10.0% | 172.8 | 36.5% |
| Transformers-CPT | 0.5% | 991.2 | 33.5% |
| Unsloth-CPT | 2.0% | 931.0 | 33.3% |
| Base + LoRA-SFT | 65.0% | 92.3 | 21.2% |
| Transformers-CPT + LoRA-SFT | 92.0% | 79.2 | 24.8% |
| Unsloth-CPT + LoRA-SFT | 94.5% | 78.5 | 25.3% |

`Question:`が回答途中の20文字目以降に再出現した行数:

| Condition | rows |
| --- | ---: |
| Base | 0 / 200 |
| Transformers-CPT | 145 / 200 |
| Unsloth-CPT | 130 / 200 |
| Base + LoRA-SFT | 0 / 200 |
| Transformers-CPT + LoRA-SFT | 0 / 200 |
| Unsloth-CPT + LoRA-SFT | 0 / 200 |

## 最新1,200回答からの定性的所感

以下は保存回答を確認したエージェントの所感であり、正解判定ではありません。

1. CPT単体は平均文字数が900文字を超え、`Question:`を追加してQA形式を継続する出力が多い。
2. LoRA-SFT後は3条件とも`Question:`継続が検出されず、短文指示への追従が大きく増えている。
3. Transformers-CPT-SFTとUnsloth-CPT-SFTは144/200行で回答文字列が完全一致し、2つのCPT実装から得たモデルのSFT後挙動は非常に近い。
4. Base-SFTも短文化するため、短文形式への改善はCPTだけでなくSFTの効果として分離して考える必要がある。
5. 短い回答でも、存在未確認のコマンドや設定名を含む例がある。簡潔さは正確さを保証しない。

## QAセットの制約

今回使用したQAは200行ですが、exact questionで83種類です。30の重複question groupがあり、147行が重複groupに属します。そのため、200独立問題のaccuracy benchmarkとして扱うことはできません。

現在のQAと1,200回答は、実行済みexperimentを再現・監査するためそのまま公開します。将来QAを修正する場合は別experimentとして回答を再生成し、現在の結果を上書きしません。

## 現時点の結論

- 0.6Bモデルのfull CPTをRTX 4070 12GBで実行できた。
- CPT後にheld-out lossが低下した。
- Unslothはこの条件でVRAMを削減したが、速度優位は一貫しなかった。
- LoRA-SFT後は短文形式への追従が大きく増えた。
- Coding Agent知識の事実精度向上は未評価である。

