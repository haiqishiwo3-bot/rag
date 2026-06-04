import os
import asyncio
import sys
import hashlib
from langchain_core.documents import Document
from pymilvus import MilvusClient, DataType
from loguru import logger

# 添加 src 目录到 Python 路径
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)


# 向量库操作
class VectorDb:
    HOST = os.getenv('MILVUS_HOST')
    PORT = os.getenv('MILVUS_PORT')
    DB_NAME = os.getenv('MILVUS_DB_NAME')
    COLLECTION_NAME = os.getenv('MILVUS_COLLECTION_NAME')

    def __init__(self):
        self.client = MilvusClient(uri=f"http://{self.HOST}:{self.PORT}/{self.DB_NAME}")

    def create_vector_collection(self, collection_name, dimension=1024):
        """创建向量集合（如果不存在）"""
        try:
            # 检查集合是否存在
            if self.client.has_collection(collection_name):
                logger.warning(f"集合 '{collection_name}' 已存在")
                return True

            # 定义集合 schema
            schema = self.client.create_schema(
                auto_id=True,
                enable_dynamic_field=True
            )

            schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
            schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=dimension)
            schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=65535)
            schema.add_field(field_name="filename", datatype=DataType.VARCHAR, max_length=65535)
            schema.add_field(field_name="metadata", datatype=DataType.JSON)
            # 添加内容哈希字段用于去重
            schema.add_field(field_name="content_hash", datatype=DataType.VARCHAR, max_length=64)

            # 创建集合
            index_params = self.client.prepare_index_params()
            index_params.add_index(
                field_name="vector",
                index_type="AUTOINDEX",
                metric_type="COSINE"
            )

            # 为 content_hash 添加索引（使用正确的索引类型）
            index_params.add_index(
                field_name="content_hash",
                index_type="Trie",  # Milvus 支持的索引类型：Trie, INVERTED, BINARY
            )

            self.client.create_collection(
                collection_name=collection_name,
                schema=schema,
                index_params=index_params
            )
            logger.info(f"成功创建集合 '{collection_name}'")
            return True

        except Exception as e:
            logger.error(f"创建集合失败：{e}")
            return False

    def search(self, collection_name, query_vector, top_k):
        return self.client.search(
            collection_name=collection_name,
            data=[query_vector],
            limit=top_k,
            output_fields=["content", "metadata", "filename", "content_hash"]
        )

    def query(self, collection_name, content_hashes,content_file):
        return self.client.query(
            collection_name=collection_name,
            filter=f"{content_file} in {content_hashes}",
            output_fields=[content_file]
        )

    def flush(self,collection_name:str):
        self.client.flush(collection_name)

    def insert(self,collection_name:str,data:list):
        if data is None or len(data)==0:
            return
        self.client.insert(collection_name,data)
