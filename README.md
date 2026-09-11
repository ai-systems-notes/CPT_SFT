# Qwen3-0.6B CPT + LoRA-SFT Experiment

[English](README_en.md) | 日本語

Qwen3-0.6B-BaseへAI coding agent関連の公式WebドキュメントをContinued Pretraining（CPT）し、Transformers/PyTorch実装とUnsloth実装の速度・VRAMをRTX 4070 12GBで比較する再現用リポジトリです。

CPT後には、Baseと2種類のCPT checkpointへ同じ汎用データでPEFT LoRA-SFTを適用し、6条件から各200行、合計1,200回答を生成しました。

本リポジトリの目的は、実用モデルの公開ではなく、個人GPUでCPT、LoRA-SFT、QA生成まで実行した手順・コード・設定・実測値を再現可能な形で残すことです。

> [!IMPORTANT]
> CPT学習、LoRA-SFT学習、12条件×70問＝840回答の生成、参照回答の一次情報検証、独立した2系統のLLM-as-a-Judgeまで完了しています。
> ドメイン別の知識獲得は有意差つきで報告できますが、単一seed・70問の規模では「ベストモデル」は主張しません。

## 実験状況

| 工程 | 状態 |
| --- | --- |
| 公式Webドキュメント収集・前処理 | 完了 |
| Full-parameter CPT（Transformers / Unsloth） | 完了 |
| LoRA-SFT（Base / Transformers-CPT / Unsloth-CPT） | 完了 |
| 6条件×200行、計1,200回答の生成 | 完了（旧評価セット） |
| 形式・字句指標の集計 | 完了 |
| 公式資料根拠の70問への評価セット再構築 | 完了 |
| 12条件×70問、計840回答の生成 | 完了 |
| QA参照回答の全件一次情報検証 | 完了（70/70。ただし2件は後述の未解決あり） |
| 外部LLM-as-a-Judge | 完了（`gpt-5.6-terra`×3 と `gemini-3.5-flash-lite` の2系統） |
| 対応あり比較による効果量の算出 | 完了 |

### 検証の到達点

- 70問すべてに出典URL、出典本文のSHA-256、根拠抜粋、自動検証結果を記録しています。自動検証は70/70が「必要キーワードが単一の公式文書内に揃う」で通過し、Judgeによる参照回答の品質評価も70/70がgoodでした。
- **2件は出典が学習コーパスに入っていません。** `seen_codex_013` と `seen_codex_014` の出典 `codex-manual.md` は、収集はできている実在の公式ページ（HTTP 200）ですが、1,062,687文字の集約ページのため前処理で `aggregate_document` として除外しました。参照回答自体はこの公式ページに根拠を持ちます。この2件は `sft_seen`（SFTへ直接収録した知識の暗記対照群）にあり、CPT想起の測定には使っていません。2件を除くと `base → base_sft_v3` は +1.767 から +1.893 に上がるため、残したまま報告するほうが保守的です。
- 判定は独立した2系統で行い、840スコアの相関はPearson r = 0.8688、有意判定は25件中24件が一致しました。
- 数値の再生成は `python3 cpt_training/scripts/analyze_blog_claims.py` で行えます。詳細は [`cpt_training/results/published/qa70_dual_judge/`](cpt_training/results/published/qa70_dual_judge/) を参照してください。

詳細な実験結果と定性的所感:

- [日本語レポート](docs/experiment_report_ja.md)
- [English report](docs/experiment_report_en.md)
- [公開用メトリクスと生成回答](cpt_training/results/published/)

## 主な実測結果

### Full CPT比較

Qwen3-0.6B-Base、sequence length 1,024、1 epoch、同一packed datasetを使用しました。

| Run | Framework | tokens/s | peak allocated | `nvidia-smi` peak | train time | held-out loss |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | Transformers/PyTorch | 4,754.61 | 8,474 MiB | 9,450 MiB | 321.18秒 | 2.5403 → 2.1455 |
| 1 | Unsloth | 4,761.86 | 7,673 MiB | 8,060 MiB | 320.55秒 | 2.5402 → 2.1455 |
| 2 | Transformers/PyTorch | 4,981.27 | 8,474 MiB | 9,450 MiB | 305.72秒 | 2.5403 → 2.1456 |
| 2 | Unsloth | 4,778.68 | 7,673 MiB | 8,060 MiB | 319.12秒 | 2.5402 → 2.1455 |

この環境ではUnslothのpeak allocated VRAMが2回とも約801 MiB少なくなりました。速度はRun 1でほぼ同等、Run 2ではTransformers/PyTorchが速く、Unslothが常に高速という結果にはなりませんでした。

### LoRA-SFT

SFTはfull fine-tuningではなく、PEFT LoRA（`r=16`, `alpha=32`, `dropout=0.05`）です。

| Source model | loss | train time | tokens/s | peak allocated |
| --- | ---: | ---: | ---: | ---: |
| Base | 0.6172 | 10.36秒 | 2,871.27 | 2,519 MiB |
| Transformers-CPT | 0.6135 | 10.24秒 | 2,905.33 | 2,519 MiB |
| Unsloth-CPT | 0.6126 | 10.12秒 | 2,939.11 | 2,519 MiB |

各条件は39 optimizer steps、29,754 non-padding tokensを処理しました。trainable parametersは10,092,544、adapter込み総parameterは606,142,464です。

### QA生成の補助指標

同じプロンプト、greedy decoding、BF16、`max_new_tokens=256`で生成しました。

| Condition | 100文字以内 | 平均文字数 | keyword recall |
| --- | ---: | ---: | ---: |
| Base | 10.0% | 172.8 | 36.5% |
| Transformers-CPT | 0.5% | 991.2 | 33.5% |
| Unsloth-CPT | 2.0% | 931.0 | 33.3% |
| Base + LoRA-SFT | 65.0% | 92.3 | 21.2% |
| Transformers-CPT + LoRA-SFT | 92.0% | 79.2 | 24.8% |
| Unsloth-CPT + LoRA-SFT | 94.5% | 78.5 | 25.3% |

これは形式・文字列一致の指標であり、事実正確性ではありません。

## 比較する6条件

```text
Qwen3-0.6B-Base
├── base
├── transformers_cpt       # full-parameter CPT
├── unsloth_cpt            # full-parameter CPT
├── base_sft               # Base + LoRA-SFT
├── transformers_cpt_sft   # Transformers-CPT + LoRA-SFT
└── unsloth_cpt_sft        # Unsloth-CPT + LoRA-SFT
```

## 学習条件

### CPT

- Base model: [`Qwen/Qwen3-0.6B-Base`](https://huggingface.co/Qwen/Qwen3-0.6B-Base)
- Revision: `da87bfb608c14b7cf20ba1ce41287e8de496c0cd`
- Sequence length: 1,024
- Update: full parameter
- Optimizer: AdamW 8-bit
- Parameter storage: FP32
- Compute: BF16
- Micro batch: 1
- Gradient accumulation: 16
- Learning rate: `1e-5`
- Epochs: 1
- Seed: `20260815`
- GPU: NVIDIA GeForce RTX 4070 12GB

Qwen3の正確なknowledge cutoffは公式に確認できていないため、本実験ではlate 2024と仮定しています。これは実験上の仮定です。

### LoRA-SFT

- Framework: Transformers + PEFT
- Dataset: 汎用instruction 200件
- LoRA: `r=16`, `alpha=32`, `dropout=0.05`
- Target modules: `q/k/v/o_proj`, `gate/up/down_proj`
- Learning rate: `2e-4`
- Epochs: 3
- Per-device batch: 4
- Gradient accumulation: 4
- Effective batch: 16
- Sequence length: 512
- Seed: 42
- Loss: promptをmaskし、answerとEOSだけを学習

## リポジトリ構成

```text
.
├── ai_coding_agent_cpt_data/
│   ├── QA_Dataset/          # 暫定QA 200行
│   ├── SFT_Dataset/         # 汎用SFT 200件
│   └── scripts/             # 収集・前処理・検証・Judge
├── cpt_training/
│   ├── configs/             # CPT設定
│   ├── scripts/             # CPT/SFT/QA実装
│   ├── src/                 # 共通処理
│   └── results/published/   # Git公開用の小容量結果
├── confirm_ui/              # 回答・Judge確認UI
├── docs/                    # 公開用レポート
├── notes/                   # 内部メモ（Git対象外）
└── scripts/                 # LoRA-SFT実行wrapper
```

## 再現手順

### 1. 必要環境

- Ubuntu 24.04
- Python 3.12
- CUDA対応NVIDIA GPU
- NVIDIA driverと`nvidia-smi`
- [`uv`](https://docs.astral.sh/uv/)
- model cache、venv、checkpoint用のディスク空き容量

### 2. 環境構築

データ収集、Transformers、Unslothを別venvへインストールします。

```bash
./ai_coding_agent_cpt_data/setup_env.sh
./cpt_training/setup_transformers_env.sh
./cpt_training/setup_unsloth_env.sh
```

### 3. 公式Webドキュメント収集

```bash
cd ai_coding_agent_cpt_data
../.venv-data/bin/python scripts/scrape_docs_spider.py
cd ..
```

raw本文は`ai_coding_agent_cpt_data/data/scraped_documents.jsonl`へ保存されます。容量と再配布範囲を考慮し、raw本文はGitへ含めません。

### 4. 前処理

```bash
./cpt_training/run_preprocess.sh
```

cleaning、QA参照元documentのCPT trainへの固定、held-out testとの分離、token packingを行い、`artifacts/datasets/cpt_v1/`へ保存します。QA行そのものは学習データへ入れません。

### 5. CPTスモークテスト

```bash
./cpt_training/run_compare_all.sh smoke
```

### 6. Full CPT比較

```bash
./cpt_training/run_compare_all.sh full --confirm-full
```

結果は`cpt_training/results/current/compare_full/`、checkpointは`checkpoints/{transformers,unsloth}/current/`へ保存されます。

### 7. SFTデータ検証とLoRA-SFT

```bash
python3 ai_coding_agent_cpt_data/scripts/verify_sft_leakage.py
./scripts/run_sft_base.sh --smoke-test
./scripts/run_sft_all.sh --confirm-full
```

3つのLoRA adapterは`checkpoints/sft/*/current/`へ保存されます。

### 8. 6条件のQA回答生成

```bash
./cpt_training/run_qa_evaluation.sh full --model-set 6
```

結果は`cpt_training/results/current/qa/`へ上書きされます。

### 9. 任意: 外部JudgeとUI

Judgeを使う場合のみ、Git管理外の`.env`へAPI keyを設定します。

```text
GEMINI_API_KEY=your_api_key
```

```bash
./cpt_training/run_gemini_judge.sh
./confirm_ui/run_ui.sh
```

UIは`results/current/qa/`に回答があればそれを使用し、なければGitに含まれる`results/published/sft_qa_ablation/`の1,200回答を表示します。外部Judgeは任意工程であり、公開結果にはJudgeスコアを含みません。

## 再現性の範囲

コード、設定、model revision、依存version、seed、入力・出力hashを記録しています。ただし、別環境でのビット単位一致は保証しません。

- 公式Webページが更新される
- raw収集本文をGitへ含めていない
- CUDA kernelとGPU hardwareが異なる
- downloader/cache状態が異なる
- 外部APIモデルが更新される

実験時raw入力のSHA-256は`3299966311cadfb01cc09f69fa953a5396205c45b5b98d87c6f066bd8cc1983b`です。厳密なデータ再現には同じraw JSONLが必要です。通常の再実行では現在の公式ページから新しいsnapshotを作ります。

## Gitへ含めないもの

- `.env`、API key
- Python venv、Hugging Face cache、Unsloth cache
- raw scraped document本文
- packed dataset
- CPT checkpoint、LoRA adapter、GGUF
- `results/current/`の上書き成果物
- `notes/`の内部メモ
- 未検証のローカル分析補助スクリプト

小容量のmetrics、metadata、モデル別1,200回答、統合QA回答は`cpt_training/results/published/`へ収録します。

