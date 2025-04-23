# 埋め込み生成とFAISSインデックス化用スクリプト
import pickle
import faiss
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
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    chunks = chunk_markdown_files()
    texts = [chunk['content'].page_content for chunk in chunks]
    print(f"{len(texts)}個のチャンクを埋め込み生成します…")
    embeddings = get_embeddings(texts, tokenizer, model, device)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    faiss.write_index(index, 'faiss.index')
    with open('chunks.pkl', 'wb') as f:
        pickle.dump(chunks, f)
    print("FAISSインデックス(faiss.index)とチャンク情報(chunks.pkl)を保存しました。")
