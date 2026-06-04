from sentence_transformers import SentenceTransformer
from typing import List
from loguru import logger

class LocalEmbeddings:
    """本地嵌入模型封装"""

    def __init__(self, model_path: str):
        self.model = SentenceTransformer(model_path)
        logger.info(f"模型加载成功，维度：{self.model.get_sentence_embedding_dimension()}")
        # 测试
        test_embedding = self.embed_query("测试文本")
        logger.info(f"向量维度：{len(test_embedding)}")

    def embed_query(self, text: str) -> List[float]:
        """嵌入单个文本"""
        return self.model.encode(text, normalize_embeddings=True).tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts, normalize_embeddings=True).tolist()
