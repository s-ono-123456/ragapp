from langchain_openai import ChatOpenAI
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from txtai.embeddings import Embeddings
import pickle
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
MODEL_NAME = 'pkshatech/GLuCoSE-base-ja-v2'
INDEX_PATH = 'index_files/txtai.index'
CHUNKS_PATH = 'index_files/chunks.pkl'

CHAT_OPENAI_MODEL = "gpt-4.1-nano"

# 初回のみロード
_tokenizer = None
_model = None
_embeddings = None
_chunks = None

def _load_resources():
    global _tokenizer, _model, _embeddings, _chunks
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if _model is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        _model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    if _embeddings is None:
        _embeddings = Embeddings()
        _embeddings.load(INDEX_PATH)
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

def get_gpt_response(prompt, top_k=3, use_graph=True):
    """
    prompt: ユーザーからの質問文
    top_k: 関連チャンクの取得数
    use_graph: グラフ検索を使用するかどうか
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

    # 検索実行
    results = _embeddings.search(
        expanded_prompt, 
        limit = top_k,
        graph = use_graph
    )
    
    # 関連チャンク抽出
    related_chunks = []
    
    # グラフ検索の有無で処理を分岐
    if use_graph:
        # NetworkXオブジェクトの場合は、ノードを取得してスコア順にソート
        # NetworkXオブジェクトはデフォルトでノードIDをキーとした辞書のような動作をする
        # nodes()メソッドではなくnode属性を使用する
        print(f"Results: {results.scan(data=True)}")
        nodes = results.scan(data=True)
        # nodes = list(results)
        for result in nodes:
            # txtaiはデフォルトで(id, score, text)の形式のタプルを返す
            idx = result[0]  # タプルの最初の要素がID
            if idx < len(_chunks):
                chunk = _chunks[idx]
                # 参考情報としてpage_contentとmetadata（あれば）を返す
                ref = {
                    "page_content": chunk['content'].page_content,
                    "score": result[1]['score']  # スコア情報も含める
                }
                if hasattr(chunk['content'], "metadata"):
                    ref["metadata"] = chunk['content'].metadata
                related_chunks.append(ref)
    else:
        # 通常の検索結果（タプル形式 (id, score, text)）の場合
        print(f"Results: {results}")
        for result in results:
            # txtaiはデフォルトで(id, score, text)の形式のタプルを返す
            idx = result[0]  # タプルの最初の要素がID
            if idx < len(_chunks):
                chunk = _chunks[idx]
                # 参考情報としてpage_contentとmetadata（あれば）を返す
                ref = {
                    "page_content": chunk['content'].page_content,
                    "score": result[1]  # スコア情報も含める
                }
                if hasattr(chunk['content'], "metadata"):
                    ref["metadata"] = chunk['content'].metadata
                related_chunks.append(ref)
    
    # 関連情報を付加したプロンプト作成
    print(f"Related Chunks: {related_chunks}")
    context = "\n\n".join([f"[スコア: {c['score']:.4f}]\n{c['page_content']}" for c in related_chunks])
    augmented_prompt = f"【参考情報】\n{context}\n\n【質問】\n{prompt}"
    # LLM呼び出し
    response = llm.invoke(augmented_prompt)
    return {
        "response": response.content,
        "references": related_chunks
    }
