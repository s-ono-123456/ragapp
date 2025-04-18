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
            st.markdown(f"**[{i}]** {ref['page_content']}")
            if "metadata" in ref:
                st.markdown(f"`{ref['metadata']}`")