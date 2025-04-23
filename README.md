# ragapp

## 概要
このアプリは、StreamlitとOpenAIのGPT-4.1-nanoモデルを活用したシンプルなWebアプリケーションです。加えて、`sample/`ディレクトリ内のMarkdown設計書を分割（チャンキング）し、AI活用や検索などに利用できる形に変換するスクリプトも含まれています。

## ファイル構成
```
ragapp/
├── app.py                # StreamlitによるWebアプリ本体
├── llm_client.py         # OpenAI APIを使った応答取得ラッパー
├── chunk_md.py           # sample/内Markdownファイルのチャンキングスクリプト
├── embed_and_index.py    # Markdownチャンクの埋め込み生成＆FAISSインデックス作成スクリプト
├── requirements.txt      # 必要なPythonパッケージ
├── README.md             # このファイル
├── ragapp.code-workspace # VSCode用ワークスペース設定
├── sample/               # サンプル設計書Markdown格納ディレクトリ
│   ├── 受注管理画面.md
│   ├── 在庫管理画面.md
│   └── 発送管理画面.md
├── index_files/          # FAISSインデックス・チャンク情報の保存先
│   ├── faiss.index
│   └── chunks.pkl
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
   streamlit run app.py --server.fileWatcherType none
   ```

## 使い方
- テキストボックスに質問を入力し、エンターキーを押すとAIの応答が表示されます。

## Markdownチャンキング・埋め込み・インデックス化について

- `chunk_md.py` を実行すると、`sample/`ディレクトリ内の全Markdownファイルを見出しや文字数で分割し、AIや検索用途で扱いやすいチャンクデータを生成できます。
  - 実行例：
    ```bash
    python chunk_md.py
    ```
    ※標準出力にチャンク例が表示されます。

- `embed_and_index.py` を実行すると、`chunk_md.py`で生成されるチャンクに対して日本語BERTモデル（GLuCoSE-base-ja-v2）で埋め込みベクトルを生成し、FAISSでインデックス化します。生成物は`index_files/`ディレクトリ（`faiss.index`, `chunks.pkl`）に保存されます。
  - 実行例：
    ```bash
    python embed_and_index.py
    ```
    ※`index_files/`配下にFAISSインデックスとチャンク情報が保存されます。

## 依存パッケージ
- streamlit
- openai
- langchain
- langchain-openai
- transformers
- torch
- faiss-cpu
- sentencepiece

## 補足
- `llm_client.py` はOpenAIのAPIラッパーです。モデル名やAPIキーは適宜変更してください。
- `embed_and_index.py` はMarkdownチャンクの埋め込み生成とFAISSインデックス作成を行います。日本語BERTモデル（GLuCoSE-base-ja-v2）を利用します。
- `index_files/` 配下にはFAISSインデックス（faiss.index）とチャンク情報（chunks.pkl）が保存されます。
- `sample/`配下のMarkdownは業務システムの画面設計例です。ご自身の設計書に差し替えて利用可能です。
- txtai[graph]をインストールするには環境変数にPYTHONUTF8を追加し、値を1にセットする必要あり。