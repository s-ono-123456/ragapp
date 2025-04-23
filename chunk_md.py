# sampleフォルダ内の全Markdownファイルをチャンキングする
import glob
import os

# OpenMP競合を回避するための設定
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter

# sampleディレクトリのパス
SAMPLE_DIR = os.path.join(os.path.dirname(__file__), 'sample')

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100

def chunk_markdown_files():
    """
    sampleフォルダ内の全Markdownファイルをチャンキングし、
    各チャンクの情報を辞書リストで返す。
    """
    md_files = glob.glob(os.path.join(SAMPLE_DIR, '*.md'))
    all_chunks = []
    for md_file in md_files:
        with open(md_file, encoding='utf-8') as f:
            md_text = f.read()
        splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[('#', 'h1'), ('##', 'h2'), ('###', 'h3')])
        header_chunks = splitter.split_text(md_text)
        char_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        for chunk in header_chunks:
            sub_chunks = char_splitter.split_documents([chunk])
            for sub_chunk in sub_chunks:
                all_chunks.append({
                    'file': os.path.basename(md_file),
                    'content': sub_chunk
                })
    return all_chunks

# チャンク例の出力（デバッグ用）
if __name__ == '__main__':
    chunks = chunk_markdown_files()
    for chunk in chunks[:5]:
        print(f"[file: {chunk['file']}]:\n{chunk['content'].page_content[:200]}\n---\n")