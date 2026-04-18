# Changelog

## 2026-04-18

- `notebooks/01-exploratory-data-analysis.ipynb` に、EDAの目的・時系列特徴量・データ結合の意図が追いやすい日本語補足を追加。
- `notebooks/02-interactive-data-preparation.ipynb` に、特徴量生成・時間窓集計・ラベル作成・学習確認の日本語補足を追加。
- `notebooks/03-ingest-with-feature-store.ipynb` に、FeatureSet定義・バッチ取り込み・リアルタイム取り込みの日本語補足を追加。
- `notebooks/04-train-test-pipeline.ipynb` に、Feature Vector作成・オフライン取得・学習ワークフロー実行の日本語補足を追加。
- `notebooks/05-real-time-serving-pipeline.ipynb` に、Serving Class定義・アンサンブル構成・オンライン推論確認の日本語補足を追加。
- 主要なコードセルには、処理意図が追える日本語コメントを追加。
- `README.md` とノートブック内の参照リンクを、`notebooks/` 配下の配置に合わせて更新。
- 5本の notebook で、既存の英語マークダウンに対して同一セル内へ `日本語訳:` を追記。
- 5本の notebook で、既存の英語コードコメントの直後に `# 日本語訳:` を追記。
- `notebooks/01-exploratory-data-analysis.ipynb` の `%pip install` セルで、1文字ずつ分解されていた `source` を行単位に修正。
- `src/date_adjust.py` に、時系列再配置ロジックの意図が追いやすい日本語ドキュメント文字列と補足コメントを追加。
- `src/get_vector.py` に、Feature Vector 作成とオフライン特徴量取得の流れを説明する日本語コメントを追加。
- `src/serving.py` に、モデル読み込みと推論レスポンス生成の処理を説明する日本語コメントを追加。
- `src/train_sklearn.py` に、データ結合・分割・ハイパーパラメータ探索・評価指標計算の日本語コメントを追加。
- `src/train_workflow.py` に、MLRun パイプライン各ステップとモデル監視設定の日本語コメントを追加。
