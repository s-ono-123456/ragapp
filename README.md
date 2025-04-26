# ragapp

## 概要
このアプリは、StreamlitとOpenAIのGPT-4.1-nanoモデルを活用したシンプルなWebアプリケーションです。加えて、`sample/`ディレクトリ内のMarkdown設計書を分割（チャンキング）し、AI活用や検索などに利用できる形に変換するスクリプトも含まれています。サブディレクトリごとに設計書を管理し、個別にインデックス化することもできます。

## ファイル構成
```
ragapp/
├── app.py                # StreamlitによるWebアプリ本体
├── llm_client.py         # OpenAI APIを使った応答取得ラッパー
├── chunk_md.py           # sample/内Markdownファイルのチャンキングスクリプト
├── embed_and_index.py    # Markdownチャンクの埋め込み生成＆txtaiインデックス作成スクリプト
├── requirements.txt      # 必要なPythonパッケージ
├── README.md             # このファイル
├── ragapp.code-workspace # VSCode用ワークスペース設定
├── sample/               # サンプル設計書Markdown格納ディレクトリ
│   ├── batch_design/     # バッチ設計書のサブディレクトリ
│   │   ├── 受注データ取込バッチ.md
│   │   ├── 受注確定バッチ.md
│   │   └── ...
│   ├── screen_design/    # 画面設計書のサブディレクトリ
│   │   ├── 受注入力画面.md
│   │   ├── 受注管理画面.md
│   │   └── ...
├── index_files/          # txtaiインデックス・チャンク情報の保存先
│   ├── all/              # 全サブディレクトリを含むインデックス
│   │   ├── chunks.pkl
│   │   └── txtai.index/
│   ├── batch_design/     # バッチ設計書のみのインデックス
│   │   ├── chunks.pkl
│   │   └── txtai.index/
│   ├── screen_design/    # 画面設計書のみのインデックス 
│   │   ├── chunks.pkl
│   │   └── txtai.index/
├── pages/                # Streamlitのマルチページアプリ用ディレクトリ
│   └── md_viewer.py      # Markdown設計書表示アプリ
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
- メインアプリ（app.py）:
  - サイドバーの「検索対象」からサブディレクトリを選択できます（「全て」または特定のサブディレクトリ）
  - テキストボックスに質問を入力し、エンターキーを押すとAIの応答が表示されます
  - 選択したサブディレクトリの設計書のみを対象に検索・回答が行われます

- 設計書表示アプリ（pages/md_viewer.py）:
  - サイドバーの「設計書の種類」からサブディレクトリを選択できます
  - 選択したサブディレクトリ内のMarkdownファイル一覧が表示されます
  - ファイルを選択すると内容が表示されます

## Markdownチャンキング・埋め込み・インデックス化について

- `chunk_md.py` を実行すると、`sample/`ディレクトリ内の全Markdownファイルを見出しや文字数で分割し、AIや検索用途で扱いやすいチャンクデータを生成できます。サブディレクトリ内のファイルも処理されます。
  - 実行例：
    ```bash
    python chunk_md.py
    ```
    ※標準出力にチャンク例とサブディレクトリごとのチャンク数が表示されます。

- `embed_and_index.py` を実行すると、`chunk_md.py`で生成されるチャンクに対して日本語BERTモデル（GLuCoSE-base-ja-v2）で埋め込みベクトルを生成し、txtaiでインデックス化します。デフォルトでは、すべてのサブディレクトリを含む「all」インデックスと、各サブディレクトリごとのインデックスが生成されます。
  - 実行例（すべてのインデックスを生成）：
    ```bash
    python embed_and_index.py
    ```
  - 特定のサブディレクトリのみインデックス化：
    ```bash
    python embed_and_index.py --subdir batch_design
    ```
  - すべてのサブディレクトリを1つのインデックスとして処理：
    ```bash
    python embed_and_index.py --all-in-one
    ```
    ※`index_files/`配下にtxtaiインデックスとチャンク情報が保存されます。

## 依存パッケージ
- streamlit
- openai
- langchain
- langchain-openai
- transformers
- torch
- txtai[graph]
- sentencepiece

## 補足
- `llm_client.py` はOpenAIのAPIラッパーです。モデル名やAPIキーは適宜変更してください。
- `embed_and_index.py` はMarkdownチャンクの埋め込み生成とtxtaiインデックス作成を行います。日本語BERTモデル（GLuCoSE-base-ja-v2）を利用します。
- `index_files/` 配下には各サブディレクトリごとのtxtaiインデックスとチャンク情報が保存されます。
- `sample/`配下のMarkdownは業務システムの架空の設計書例です。ご自身の設計書に差し替えて利用可能です。サブディレクトリを作成して整理することができます。
- txtai[graph]をインストールするには環境変数にPYTHONUTF8を追加し、値を1にセットする必要あり。