import os
import re
from datetime import datetime
from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter
from loguru import logger

# 自定义文本加载器，处理编码问题
def load_text_file_with_encoding(file_path, encoding='utf-8'):
    """加载文本文件，处理编码问题"""
    try:
        with open(file_path, 'r', encoding=encoding) as f:
            content = f.read()
        return content
    except UnicodeDecodeError:
        # 如果UTF-8失败，尝试其他编码
        encodings = ['gbk', 'gb2312', 'latin-1', 'cp1252']
        for enc in encodings:
            try:
                with open(file_path, 'r', encoding=enc) as f:
                    content = f.read()
                print(f"使用编码 {enc} 成功读取文件")
                return content
            except UnicodeDecodeError:
                continue
        raise RuntimeError(f"无法用任何编码读取文件: {file_path}")

# 文本清洗函数
def clean_text(text):
    """清洗文本数据"""
    # 去除特殊字符
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s]', '', text)
    # 合并连续空格
    text = re.sub(r'\s+', ' ', text).strip()
    # 去除过长空白行
    text = '\n'.join([line for line in text.split('\n') if len(line) > 10])
    return text

def doc_clean_split(file_path):
    """
        切分并清理文档
    :param file_path: 文件路径
    :return: 清理后的文档列表
    """
    # 加载文件
    content = load_text_file_with_encoding(file_path)
    # 创建Document对象
    document = Document(page_content=content, metadata={"source": file_path})

    # 1000个字符为一个块
    text_splitter = CharacterTextSplitter(chunk_size=int(os.getenv('CHUNK_SIZE')), chunk_overlap=int(os.getenv('CHUNK_OVERLAP')))
    # documents = loader.load_and_split(text_splitter)
    documents = text_splitter.split_documents([document])
    # 清洗文档
    logger.info("正在清洗文档...")
    cleaned_docs = [Document(page_content=clean_text(doc.page_content), metadata=doc.metadata) for doc in documents]
    logger.info(f"文档清洗完成，共 {len(cleaned_docs)} 个文档片段")
    return cleaned_docs

def get_unique_filename(filename: str) -> str:
    """
    生成唯一的文件名，避免重名覆盖
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name, ext = os.path.splitext(filename)
    return f"{name}_{timestamp}{ext}"