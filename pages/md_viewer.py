import streamlit as st
import glob
import os

st.set_page_config(page_title="設計書表示アプリ")
st.title("設計書表示アプリ")

# sampleディレクトリのmdファイル一覧取得
md_files = glob.glob("sample/*.md")
md_file_names = [os.path.basename(f) for f in md_files]

st.sidebar.title("sampleディレクトリのmdファイル")
selected_md = st.sidebar.selectbox("表示するmdファイルを選択してください", md_file_names)

if selected_md:
    with open(os.path.join("sample", selected_md), "r", encoding="utf-8") as f:
        md_content = f.read()
    st.markdown(md_content)