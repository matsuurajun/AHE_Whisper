# Repository Guidelines

## プロジェクト構成とモジュール配置
- `ahe_whisper/` にパイプライン本体（ASR/VAD/話者分離/アライメント/出力）が集約されています。主な入口は `ahe_whisper/pipeline.py` と `ahe_whisper/aligner.py` です。
- `gui.py` と `ahe_whisper/gui_logic.py` が niceGUI のUIとワーカー処理を担当します。
- `tools/` は診断・評価スクリプト群（正式なテストの代替）です。
- `docs/` と `README.md` は利用方法や設計メモです。
- `date/` は評価データや比較結果の置き場です。
- 出力は `AHE-Whisper-output/` 配下に実行ごとのフォルダで生成されます。

## ビルド・テスト・開発コマンド
- `uv sync` 依存関係をロックされた状態でセットアップします。
- `uv run python gui.py` Web UI を起動します（推奨）。
- `python gui.py` venv有効化後のUI起動です。
- `python prefetch_models.py` モデルを `models/` に事前取得します（オフライン用）。
- `AHE_BOUNDARY_SNAP=1 python gui.py` 境界スナップ診断を有効化します。
- `AHE_TRACE_ALIGNER=1 python gui.py` アライメントのトレースを有効化します。

## コーディングスタイルと命名規約
- Python 3.12.4前提。`ahe_whisper/config.py` のdataclass設計に合わせて変更してください。
- インデントは4スペース。JP/EN混在コメントは周辺スタイルに合わせて維持します。
- 命名は `snake_case`（関数/変数）、`PascalCase`（クラス）、`UPPER_SNAKE_CASE`（定数）。
- リポジトリ共通のフォーマッタ/リンタは未設定のため、近傍コードのスタイルに合わせます。

## テスト指針
- 公式テストスイートはありません。`tools/` の診断を使って検証します。
- 代表例:
  - `python tools/diagnose_embeddings_v90.py`
  - `python tools/analyze_boundary_snap.py AHE-Whisper-output/<run>/boundary_snap.jsonl`
- 診断の生成物は、必要がない限りコミットしないでください。

## コミット/PRガイド
- 既存履歴は `update` のような短文が多く、統一規約はありません。
- 追加する場合は、短く具体的な命令形を推奨します（例: "tune aligner switch penalties"）。
- PRには「変更概要」「理由」「実行した診断」「UIのスクリーンショットやサンプル出力」を含めてください。

## 設定とモデル
- 主要な挙動は `ahe_whisper/config.py` のdataclassで管理されています。
- モデルは `models/` にキャッシュされます（gitignore）。追加時は `prefetch_models.py` 更新と `CHANGELOG.md` 記載が必要です。

## エージェント向けメモ
- アーキテクチャや診断手順は `CLAUDE.md` を参照してください。
