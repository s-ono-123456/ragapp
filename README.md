# ragapp

## 概要
このアプリは、StreamlitとOpenAIのGPT-4.1-nanoモデルを活用したシンプルなWebアプリケーションです。加えて、`sample/`ディレクトリ内のMarkdown設計書を分割（チャンキング）し、AI活用や検索などに利用できる形に変換するスクリプトも含まれています。

## ファイル構成
```
ragapp/
├── app.py                # StreamlitによるWebアプリ本体
├── llm_client.py         # OpenAI APIを使った応答取得ラッパー
├── chunk_md.py           # sample/内Markdownファイルのチャンキングスクリプト
├── requirements.txt      # 必要なPythonパッケージ
├── README.md             # このファイル
├── ragapp.code-workspace # VSCode用ワークスペース設定
├── sample/               # サンプル設計書Markdown格納ディレクトリ
│   ├── 受注管理画面.md
│   ├── 在庫管理画面.md
│   └── 発送管理画面.md
└── __pycache__/          # Pythonキャッシュ
```

## 必要条件
- Python 3.7以上
- OpenAI APIキー（環境変数 `OPENAI_API_KEY` に設定）

## セットアップ手順
1. 必要なパッケージのインストール
   ```bash
   pip install -r requirements.txt
   ```
2. OpenAI APIキーを環境変数に設定
   ```bash
   set OPENAI_API_KEY=sk-...   # Windowsの場合
   export OPENAI_API_KEY=sk-... # Mac/Linuxの場合
   ```
3. アプリの起動
   ```bash
   streamlit run app.py
   ```

## 使い方
- テキストボックスに質問を入力し、エンターキーを押すとAIの応答が表示されます。

## Markdownチャンキングについて
- `chunk_md.py` を実行すると、`sample/`ディレクトリ内の全Markdownファイルを見出しや文字数で分割し、AIや検索用途で扱いやすいチャンクデータを生成できます。
- 実行例：
  ```bash
  python chunk_md.py
  ```
  ※標準出力にチャンク例が表示されます。

## 依存パッケージ
- streamlit
- openai
- langchain
- langchain-openai

## 補足
- `llm_client.py` はOpenAIのAPIラッパーです。モデル名やAPIキーは適宜変更してください。
- `sample/`配下のMarkdownは業務システムの画面設計例です。ご自身の設計書に差し替えて利用可能です.
