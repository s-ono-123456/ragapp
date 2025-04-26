# sampleフォルダ内の全Markdownファイルをチャンキングする
import glob
import os

# OpenMP競合を回避するための設定
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter

# sampleディレクトリのパス
SAMPLE_DIR = os.path.join(os.path.dirname(__file__), 'sample')

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

def get_subdirs():
    """
    sampleフォルダ内のサブディレクトリ一覧を取得する
    """
    subdirs = [d for d in os.listdir(SAMPLE_DIR) 
               if os.path.isdir(os.path.join(SAMPLE_DIR, d))]
    return subdirs

def chunk_markdown_files(subdir=None):
    """
    sampleフォルダ内の全Markdownファイルをチャンキングし、
    各チャンクの情報を辞書リストで返す。
    
    Args:
        subdir (str, optional): サブディレクトリ名。指定すると、そのサブディレクトリ内のファイルのみを処理する。
    """
    all_chunks = []
    
    if subdir:
        # 特定のサブディレクトリのみ処理
        subdir_path = os.path.join(SAMPLE_DIR, subdir)
        md_files = glob.glob(os.path.join(subdir_path, '*.md'))
    else:
        # 全てのサブディレクトリを処理
        md_files = []
        for root, _, files in os.walk(SAMPLE_DIR):
            for file in files:
                if file.endswith('.md'):
                    md_files.append(os.path.join(root, file))
    
    for md_file in md_files:
        # 相対パスを抽出 (サブディレクトリ名を含む)
        rel_path = os.path.relpath(md_file, SAMPLE_DIR)
        
        with open(md_file, encoding='utf-8') as f:
            md_text = f.read()
        splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[('#', 'h1'), ('##', 'h2'), ('###', 'h3')])
        header_chunks = splitter.split_text(md_text)
        char_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        for chunk in header_chunks:
            sub_chunks = char_splitter.split_documents([chunk])
            for sub_chunk in sub_chunks:
                all_chunks.append({
                    'file': rel_path,  # サブディレクトリ名を含む相対パス
                    'subdir': os.path.dirname(rel_path) if os.path.dirname(rel_path) else None,  # サブディレクトリ名
                    'content': sub_chunk
                })
    return all_chunks

# チャンク例の出力（デバッグ用）
if __name__ == '__main__':
    # 利用可能なサブディレクトリを表示
    subdirs = get_subdirs()
    print(f"利用可能なサブディレクトリ: {subdirs}")
    
    # 全てのチャンクを取得
    all_chunks = chunk_markdown_files()
    print(f"全チャンク数: {len(all_chunks)}")
    
    # サブディレクトリごとにチャンクを取得
    for subdir in subdirs:
        chunks = chunk_markdown_files(subdir)
        print(f"{subdir}のチャンク数: {len(chunks)}")
        if chunks:
            print(f"サンプル: {chunks[0]['file']}")