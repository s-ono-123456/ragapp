from langchain_openai import ChatOpenAI
import os
import glob
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from txtai.embeddings import Embeddings
import pickle
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import CrossEncoder
from chunk_md import get_subdirs

MODEL_NAME = 'pkshatech/GLuCoSE-base-ja-v2'
RERANKER_MODEL = 'hotchpotch/japanese-reranker-cross-encoder-large-v1'
INDEX_DIR = 'index_files'
CHAT_OPENAI_MODEL = "gpt-4.1-nano"

# 初回のみロード
_tokenizer = None
_model = None
_embeddings_cache = {}  # キャッシュ: {index_name: embeddings}
_chunks_cache = {}      # キャッシュ: {index_name: chunks}
_reranker = None

def get_available_indices():
    """利用可能なインデックス一覧を取得する"""
    indices = []
    directories = [d for d in os.listdir(INDEX_DIR) 
                  if os.path.isdir(os.path.join(INDEX_DIR, d))]
    
    for directory in directories:
        index_path = os.path.join(INDEX_DIR, directory, "txtai.index")
        chunks_path = os.path.join(INDEX_DIR, directory, "chunks.pkl")
        
        if os.path.exists(index_path) and os.path.exists(chunks_path):
            indices.append(directory)
    
    return indices

def _load_base_resources():
    """基本リソース（トークナイザー、モデル、リランカー）のロード"""
    global _tokenizer, _model, _reranker
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if _model is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        _model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    if _reranker is None:
        _reranker = CrossEncoder(RERANKER_MODEL)

def _load_index_resources(index_name="all"):
    """指定されたインデックスをロード"""
    global _embeddings_cache, _chunks_cache
    
    # すでにロード済みの場合はキャッシュを返す
    if index_name in _embeddings_cache and index_name in _chunks_cache:
        return _embeddings_cache[index_name], _chunks_cache[index_name]
    
    # インデックスパスの設定
    index_path = os.path.join(INDEX_DIR, index_name, "txtai.index")
    chunks_path = os.path.join(INDEX_DIR, index_name, "chunks.pkl")
    
    if not os.path.exists(index_path) or not os.path.exists(chunks_path):
        raise FileNotFoundError(f"インデックス '{index_name}' が見つかりません")
    
    # インデックスのロード
    embeddings = Embeddings()
    embeddings.load(index_path)
    
    # チャンクのロード
    with open(chunks_path, 'rb') as f:
        chunks = pickle.load(f)
    
    # キャッシュに保存
    _embeddings_cache[index_name] = embeddings
    _chunks_cache[index_name] = chunks
    
    return embeddings, chunks

def _get_query_embedding(text):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    inputs = _tokenizer([text], padding=True, truncation=True, return_tensors='pt', max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = _model(**inputs)
        emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
    return emb

def get_gpt_response(prompt, top_k=5, use_graph=True, index_name="all"):
    """
    prompt: ユーザーからの質問文
    top_k: 関連チャンクの取得数
    use_graph: グラフ検索を使用するかどうか
    index_name: 使用するインデックス名
    """
    # 基本リソースをロード
    _load_base_resources()
    
    # 指定されたインデックスをロード
    try:
        embeddings, chunks = _load_index_resources(index_name)
    except FileNotFoundError as e:
        return {"response": f"エラー: {str(e)}", "references": [], "initial_references": []}

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
    results = embeddings.search(
        expanded_prompt, 
        limit = 20,  # リランキングのためにより多くの候補を取得
        graph = use_graph
    )
    
    # 関連チャンク抽出
    related_chunks = []
    
    # グラフ検索の有無で処理を分岐
    if use_graph:
        # print(f"Results: {results.scan(data=True)}")
        nodes = results.scan(data=True)
        # nodes = list(results)
        for result in nodes:
            # txtaiはNetworkXでは(id, {'id': id , 'text':text 'score':score})の形式のタプルを返す
            idx = result[0]  # タプルの最初の要素がID
            if idx < len(chunks):
                chunk = chunks[idx]
                # 参考情報としてpage_contentとmetadata（あれば）を返す
                ref = {
                    "page_content": chunk['content'].page_content,
                    "score": result[1]['score'],  # スコア情報も含める
                    "file": chunk['file'],  # ファイル情報も含める
                    "subdir": chunk['subdir']  # サブディレクトリ情報も含める
                }
                if hasattr(chunk['content'], "metadata"):
                    ref["metadata"] = chunk['content'].metadata
                related_chunks.append(ref)
    else:
        # 通常の検索結果（タプル形式 (id, score, text)）の場合
        # print(f"Results: {results}")
        for result in results:
            # txtaiはデフォルトで(id, score, text)の形式のタプルを返す
            idx = result[0]  # タプルの最初の要素がID
            if idx < len(chunks):
                chunk = chunks[idx]
                # 参考情報としてpage_contentとmetadata（あれば）を返す
                ref = {
                    "page_content": chunk['content'].page_content,
                    "score": result[1],  # スコア情報も含める
                    "file": chunk['file'],  # ファイル情報も含める
                    "subdir": chunk['subdir']  # サブディレクトリ情報も含める
                }
                if hasattr(chunk['content'], "metadata"):
                    ref["metadata"] = chunk['content'].metadata
                related_chunks.append(ref)
    
    # リランキング前の初期検索結果を保存
    initial_chunks = related_chunks.copy()
    
    # Cross-Encoderによるリランキング処理
    if related_chunks:
        print(f"初期検索結果数: {len(related_chunks)}")
        
        # クエリとチャンクのペアを作成
        rerank_pairs = [[prompt, chunk["page_content"]] for chunk in related_chunks]
        
        # リランキングスコアを計算
        rerank_scores = _reranker.predict(rerank_pairs)
        
        # スコアをチャンクに追加
        for i, score in enumerate(rerank_scores):
            related_chunks[i]["rerank_score"] = float(score)
        
        # スコアで降順ソート
        related_chunks = sorted(related_chunks, key=lambda x: x["rerank_score"], reverse=True)
        
        # 上位のチャンクのみを使用
        related_chunks = related_chunks[:top_k]
        print(f"リランキング後の結果数: {len(related_chunks)}")
    
    # 関連情報を付加したプロンプト作成
    # print(f"Related Chunks: {related_chunks}")
    context = "\n\n".join([f"[スコア: {c.get('rerank_score', 0):.4f}, ファイル: {c.get('file', '')}]\n{c['page_content']}" for c in related_chunks])
    augmented_prompt = f"【参考情報】\n{context}\n\n【質問】\n{prompt}"
    # LLM呼び出し
    response = llm.invoke(augmented_prompt)
    return {
        "response": response.content,
        "references": related_chunks,
        "initial_references": initial_chunks  # リランキング前の結果も返す
    }
