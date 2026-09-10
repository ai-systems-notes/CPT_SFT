# Terra LLM-as-a-Judge: 12-model comparison

再構築した公式資料根拠の70問を、モデル名を伏せ、設問ごとに候補順を変えて再評価した。Gemini APIなどの外部APIは使用せず、3体の `gpt-5.6-terra` サブエージェントへ `split × category` を層化して分担した。旧スコアは参照せず、全問をゼロから採点した。

| Rank | Model | Average / 10 | Good-reference avg | Wins (ties included) | Unique wins |
|---:|---|---:|---:|---:|---:|
| 1 | `unsloth_cpt_sft_v3` | 3.400 | 3.400 | 21 | 0 |
| 2 | `transformers_cpt_sft_v3` | 3.343 | 3.343 | 21 | 0 |
| 3 | `base_sft_v3` | 3.329 | 3.329 | 23 | 9 |
| 4 | `transformers_cpt` | 3.300 | 3.300 | 19 | 0 |
| 5 | `unsloth_cpt` | 3.286 | 3.286 | 19 | 0 |
| 6 | `unsloth_cpt_sft_v2` | 3.157 | 3.157 | 24 | 1 |
| 7 | `unsloth_cpt_sft_v1` | 3.071 | 3.071 | 22 | 1 |
| 8 | `transformers_cpt_sft_v2` | 3.057 | 3.057 | 22 | 0 |
| 9 | `transformers_cpt_sft_v1` | 3.029 | 3.029 | 20 | 0 |
| 10 | `base` | 2.500 | 2.500 | 16 | 1 |
| 11 | `base_sft_v2` | 2.443 | 2.443 | 16 | 3 |
| 12 | `base_sft_v1` | 2.043 | 2.043 | 13 | 1 |

## Reference quality

- good: 70
- questionable: 0
- bad: 0

## Interpretation rules

- `sft_seen` 30問はSFT v3に直接収録した知識で、直接暗記の対照群として扱う。
- `heldout` 40問はSFT v3に直接収録していないCPTコーパス由来知識で、CPT知識を応答として引き出せるかを見る主評価とする。
- 全12モデルの回答生成には同じv3評価プロンプトを用いた。
- 絶対点は自動Judgeの主観を含むため、split別順位と同一系列のSFT前後差を重視する。
- 3分割は層化したが、Judge個体間の採点尺度差が完全になくなるわけではない。
- SFTモデルは各元モデルに適用したPEFT LoRAアダプターである。
