import streamlit as st
import openai

# OpenAI APIキーの設定
openai.api_key = ""
st.title("サンプル Streamlit アプリ")
st.write("これはStreamlitを使ったサンプルアプリケーションです。")

def get_gpt_response(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return response['choices'][0]['message']['content']

# ユーザーからの入力を受け取る
user_input = st.text_input("あなたのメッセージを入力してください:")

if st.button("送信"):
    if user_input:
        response = get_gpt_response(user_input)
        st.write("GPT-4o-miniの応答:", response)