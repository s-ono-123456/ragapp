import streamlit as st
import openai
import os

# OpenMP競合を回避するための設定
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from llm_client import get_gpt_response, get_available_indices

# OpenAI APIキーの設定
openai.api_key = os.getenv("OPENAI_API_KEY")
# Streamlitの設定
st.set_page_config(page_title="サンプル RAG アプリ")
st.title("サンプル RAG アプリ")
st.write("これはRAGのサンプルアプリケーションです。")

# 利用可能なインデックスを取得
available_indices = get_available_indices()

# サイドバーにグラフ機能と検索インデックスの選択機能を追加
st.sidebar.title("検索オプション")
use_graph = st.sidebar.checkbox("グラフ検索を有効にする", value=True, help="ONにするとグラフ検索機能を使用して関連性の高い情報を検索します")

# インデックス選択（サブディレクトリごとに検索対象を絞り込む）
selected_index = st.sidebar.selectbox(
    "検索対象",
    available_indices,
    index=0 if "all" in available_indices else 0,
    format_func=lambda x: "全て" if x == "all" else x
)

# ユーザーからの入力を受け取る
user_input = st.text_input("質問を入力してください:")

if user_input:
    # グラフ機能と選択されたインデックスを関数に渡す
    result = get_gpt_response(user_input, use_graph=use_graph, index_name=selected_index)
    tab1, tab2, tab3 = st.tabs(["応答結果", "リランキング後の参考情報", "リランキング前の参考情報"])
    with tab1:
        st.write(result["response"])
    with tab2:
        for i, ref in enumerate(result["references"], 1):
            titles = []
            metadata = ref.get("metadata", {})
            if metadata:
                if "h2" in metadata:
                    if isinstance(metadata["h2"], list):
                        titles.extend(metadata["h2"])
                    else:
                        titles.append(metadata["h2"])
                if "h3" in metadata:
                    if isinstance(metadata["h3"], list):
                        titles.extend(metadata["h3"])
                    else:
                        titles.append(metadata["h3"])
            
            # ファイル情報を表示用のタイトルに含める
            file_info = ref.get('file', '')
            title_str = " / ".join(titles) if titles else f"参考情報 {i}"
            if file_info:
                title_str = f"{title_str} ({file_info})"
            
            with st.expander(title_str):
                score_info = f"リランキングスコア: {ref.get('rerank_score', 0):.4f}"
                st.info(score_info)
                st.markdown(ref['page_content'])
    with tab3:
        for i, ref in enumerate(result["initial_references"], 1):
            titles = []
            metadata = ref.get("metadata", {})
            if metadata:
                if "h2" in metadata:
                    if isinstance(metadata["h2"], list):
                        titles.extend(metadata["h2"])
                    else:
                        titles.append(metadata["h2"])
                if "h3" in metadata:
                    if isinstance(metadata["h3"], list):
                        titles.extend(metadata["h3"])
                    else:
                        titles.append(metadata["h3"])
            
            # ファイル情報を表示用のタイトルに含める
            file_info = ref.get('file', '')
            title_str = " / ".join(titles) if titles else f"初期検索結果 {i}"
            if file_info:
                title_str = f"{title_str} ({file_info})"
            
            with st.expander(title_str):
                score_info = f"初期検索スコア: {ref.get('score', 0):.4f}"
                st.info(score_info)
                st.markdown(ref['page_content'])