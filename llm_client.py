from langchain_openai import ChatOpenAI
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import faiss
import pickle
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
MODEL_NAME = 'pkshatech/GLuCoSE-base-ja-v2'
INDEX_PATH = 'index_files/faiss.index'
CHUNKS_PATH = 'index_files/chunks.pkl'

CHAT_OPENAI_MODEL = "gpt-4.1-nano"

# 初回のみロード
_tokenizer = None
_model = None
_index = None
_chunks = None

def _load_resources():
    global _tokenizer, _model, _index, _chunks
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if _model is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        _model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    if _index is None:
        _index = faiss.read_index(INDEX_PATH)
    if _chunks is None:
        with open(CHUNKS_PATH, 'rb') as f:
            _chunks = pickle.load(f)

def _get_query_embedding(text):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    inputs = _tokenizer([text], padding=True, truncation=True, return_tensors='pt', max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = _model(**inputs)
        emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
    return emb

def get_gpt_response(prompt, top_k=3):
    """
    prompt: ユーザーからの質問文
    top_k: 関連チャンクの取得数
    """
    _load_resources()

    # Step back prompting: ユーザーの質問をより広い視野で捉えた文章に変換
    step_back_prompt = (
        "以下の質問を、より広い視野で捉えた質問文に書き換えてください。\n"
        "元の質問: " + prompt + "\n"
        "書き換えた質問:"
    )
    llm = ChatOpenAI(
        model=CHAT_OPENAI_MODEL,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    step_back_response = llm.invoke(step_back_prompt)
    expanded_prompt = step_back_response.content.strip()
    print(f"Expanded Prompt: {expanded_prompt}")

    # クエリ埋め込み
    query_emb = _get_query_embedding(expanded_prompt)
    # FAISS検索
    D, I = _index.search(query_emb, top_k)
    # 関連チャンク抽出
    related_chunks = []
    for idx in I[0]:
        if idx < len(_chunks):
            chunk = _chunks[idx]
            # 参考情報としてpage_contentとmetadata（あれば）を返す
            ref = {
                "page_content": chunk['content'].page_content
            }
            if hasattr(chunk['content'], "metadata"):
                ref["metadata"] = chunk['content'].metadata
            related_chunks.append(ref)
    # 関連情報を付加したプロンプト作成
    context = "\n\n".join([c["page_content"] for c in related_chunks])
    augmented_prompt = f"【参考情報】\n{context}\n\n【質問】\n{prompt}"
    # LLM呼び出し
    response = llm.invoke(augmented_prompt)
    return {
        "response": response.content,
        "references": related_chunks
    }
