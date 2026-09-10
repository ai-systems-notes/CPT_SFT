# AI Coding Agent CPT Data

Claude Code、OpenAI Codex、ChatGPT関連の公式Webドキュメントを収集し、Qwen3-0.6B-BaseのCPT入力とclosed-book QA generation probeを準備するディレクトリです。

学習から外部Judge、UIまでの全体手順はリポジトリrootの[`README.md`](../README.md)を参照してください。

## ディレクトリ

```text
data/          Web収集結果。raw本文JSONL/CSVはGit管理しない
QA_Dataset/    暫定QA 200行（exact unique questionは83件）
SFT_Dataset/   CPT知識とは分離した汎用SFTデータ200件
scripts/       収集、前処理、検証、外部Judge
```

## 収集環境

```bash
./ai_coding_agent_cpt_data/setup_env.sh
```

## 公式Webドキュメント収集

scraperは次をseedとして、同一ドメイン内をBFSで収集します。

- `https://code.claude.com/docs/en/overview`
- `https://learn.chatgpt.com/docs`

```bash
cd ai_coding_agent_cpt_data
../.venv-data/bin/python scripts/scrape_docs_spider.py
cd ..
```

出力:

- `data/scraped_documents.jsonl`
- `data/scraped_manifest.csv`
- `data/scraping_report.md`

raw本文は容量と再配布範囲を考慮してGitへ含めません。Webページ更新により再収集内容は変化するため、厳密なデータ再現には同じJSONLとSHA-256が必要です。

## 学習データ前処理

```bash
./cpt_training/run_preprocess.sh
```

固定seedでcleaning、train/test分割、sequence length 1,024のpackingを行います。QAの参照元として一致した公式documentはCPT trainへ固定し、held-out perplexity用testから分離します。QAのquestion/answer行そのものはCPTへ入れません。

## QA generation set

```text
QA_Dataset/eval_qa_claude.jsonl   100行
QA_Dataset/eval_qa_codex.jsonl    100行
QA_Dataset/eval_qa_combined.jsonl 200行
```

6条件すべてで同じ200行を使って回答を生成しました。ただしexact unique questionは83件で、参照回答の全件一次情報検証も未実施です。事実accuracy benchmarkではなく、実行済みgeneration probeの再現用データとして公開します。

## Gemini API key

`run_gemini_judge.py`は環境変数または次のGit管理外ファイルから`GEMINI_API_KEY`を読みます。

```text
ai_coding_agent_cpt_data/.env
.env
```

テンプレート:

```bash
cp ai_coding_agent_cpt_data/.env.example ai_coding_agent_cpt_data/.env
```

外部Judgeは任意工程です。現在の公開結果にJudgeスコアは含みません。
