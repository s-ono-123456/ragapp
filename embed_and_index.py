# 埋め込み生成とtxtaiインデックス化用スクリプト
import pickle
import os

# OpenMP競合を回避するための設定
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from txtai.embeddings import Embeddings
from transformers import AutoTokenizer, AutoModel
import torch
from chunk_md import chunk_markdown_files

MODEL_NAME = 'pkshatech/GLuCoSE-base-ja-v2'

# テキストリストをBERT埋め込みに変換
def get_embeddings(texts, tokenizer, model, device):
    embeddings = []
    batch_size = 8
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, return_tensors='pt', max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            # [CLS]トークンのベクトルを使う
            batch_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.append(batch_emb)
    return np.vstack(embeddings)

if __name__ == '__main__':
    import numpy as np
    
    # 出力ディレクトリの確認と作成
    if not os.path.exists('index_files'):
        os.makedirs('index_files')
    
    # チャンクの作成
    chunks = chunk_markdown_files()
    texts = [chunk['content'].page_content for chunk in chunks]
    print(f"{len(texts)}個のチャンクを埋め込み生成します…")
    
    # txtaiのEmbeddingsオブジェクトを作成（グラフインデックスを有効化）
    embeddings = Embeddings({
        "method": "transformers", 
        "path": MODEL_NAME,
        "graph": True,  # グラフインデックスを有効化
        "scoring": "bm25",  # BM25スコアリングを使用
        "centroid": True  # 中心性を計算して重要なノードを特定
    })
    
    # データの準備（txtai用）
    data = [(i, text, None) for i, text in enumerate(texts)]
    
    # インデックスの構築
    embeddings.index(data)
    
    # インデックスとチャンク情報の保存
    embeddings.save("index_files/txtai.index")
    
    with open('index_files/chunks.pkl', 'wb') as f:
        pickle.dump(chunks, f)
    
    print("グラフインデックスを有効化したtxtaiインデックス(index_files/txtai.index)とチャンク情報(index_files/chunks.pkl)を保存しました。")
