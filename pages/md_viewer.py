import streamlit as st
import glob
import os
from chunk_md import get_subdirs

st.set_page_config(page_title="設計書表示アプリ")
st.title("設計書表示アプリ")

# sampleディレクトリのサブディレクトリ一覧を取得
subdirs = get_subdirs()
subdirs = [""] + subdirs  # 空の選択肢を追加（ルートディレクトリを示す）

# サブディレクトリの選択
st.sidebar.title("設計書の種類")
selected_subdir = st.sidebar.selectbox(
    "設計書の種類を選択してください", 
    subdirs,
    format_func=lambda x: "全て" if x == "" else x
)

# 選択されたサブディレクトリに基づいてファイル一覧を取得
if selected_subdir == "":
    # ルートディレクトリのファイル
    md_files = glob.glob("sample/*.md")
else:
    # 選択されたサブディレクトリのファイル
    md_files = glob.glob(f"sample/{selected_subdir}/*.md")

md_file_names = [os.path.basename(f) for f in md_files]

st.sidebar.title("ファイル一覧")
if md_file_names:
    selected_md = st.sidebar.selectbox("表示するファイルを選択してください", md_file_names)
    
    if selected_md:
        if selected_subdir == "":
            file_path = os.path.join("sample", selected_md)
        else:
            file_path = os.path.join("sample", selected_subdir, selected_md)
        
        with open(file_path, "r", encoding="utf-8") as f:
            md_content = f.read()
        st.markdown(md_content)
else:
    st.sidebar.write("ファイルが見つかりません")