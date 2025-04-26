# 埋め込み生成とtxtaiインデックス化用スクリプト
import pickle
import os
import numpy as np

# OpenMP競合を回避するための設定
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from txtai.embeddings import Embeddings
from transformers import AutoTokenizer, AutoModel
import torch
from chunk_md import chunk_markdown_files, get_subdirs

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

def create_index(chunks, output_dir="index_files", index_name="default"):
    """
    指定されたチャンクリストからtxtaiインデックスを作成する
    
    Args:
        chunks: チャンクリスト
        output_dir: 出力ディレクトリ
        index_name: インデックスの名前
    """
    # テキストの抽出
    texts = [chunk['content'].page_content for chunk in chunks]
    print(f"{len(texts)}個のチャンクを埋め込み生成します…（{index_name}）")
    
    if not texts:
        print(f"警告: {index_name}には処理するチャンクがありません")
        return False
    
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
    
    # 出力ディレクトリの確認と作成
    index_dir = os.path.join(output_dir, index_name)
    if not os.path.exists(index_dir):
        os.makedirs(index_dir)
    
    # インデックスとチャンク情報の保存
    embeddings.save(os.path.join(index_dir, "txtai.index"))
    
    with open(os.path.join(index_dir, "chunks.pkl"), 'wb') as f:
        pickle.dump(chunks, f)
    
    print(f"グラフインデックスを有効化したtxtaiインデックス({index_dir}/txtai.index)とチャンク情報({index_dir}/chunks.pkl)を保存しました。")
    return True

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='サブディレクトリごとにインデックスを作成します')
    parser.add_argument('--subdir', help='処理するサブディレクトリ名（指定しないと全て処理）')
    parser.add_argument('--all-in-one', action='store_true', help='全てのサブディレクトリを1つのインデックスとして処理する')
    args = parser.parse_args()
    
    # 出力ディレクトリの確認と作成
    if not os.path.exists('index_files'):
        os.makedirs('index_files')
    
    if args.all_in_one:
        # すべてのチャンクを一つのインデックスとして処理
        chunks = chunk_markdown_files()
        create_index(chunks, index_name="all")
    elif args.subdir:
        # 特定のサブディレクトリのみを処理
        chunks = chunk_markdown_files(args.subdir)
        create_index(chunks, index_name=args.subdir)
    else:
        # サブディレクトリごとに個別のインデックスを作成
        # 全体用のインデックス
        all_chunks = chunk_markdown_files()
        create_index(all_chunks, index_name="all")
        
        # サブディレクトリごとのインデックス
        subdirs = get_subdirs()
        for subdir in subdirs:
            chunks = chunk_markdown_files(subdir)
            create_index(chunks, index_name=subdir)
