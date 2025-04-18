import streamlit as st
import openai
import os
from llm_client import get_gpt_response

# OpenAI APIキーの設定
openai.api_key = os.getenv("OPENAI_API_KEY")
# Streamlitの設定
st.title("サンプル Streamlit アプリ")
st.write("これはStreamlitを使ったサンプルアプリケーションです。")

# ユーザーからの入力を受け取る
user_input = st.text_input("質問を入力してください:")

if user_input:
    result = get_gpt_response(user_input)
    tab1, tab2 = st.tabs(["応答結果", "参考情報"])
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
            title_str = " / ".join(titles) if titles else f"参考情報 {i}"
            with st.expander(title_str):
                st.markdown(ref['page_content'])