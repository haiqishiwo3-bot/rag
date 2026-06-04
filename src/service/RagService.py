import asyncio
import hashlib
import json
import os
import sys
from fastapi import UploadFile, HTTPException
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import PromptTemplate
from loguru import logger
from starlette.responses import JSONResponse
import uuid
from typing import List
from concurrent.futures import ThreadPoolExecutor

from model.LocalEmbeddings import LocalEmbeddings

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from model.LlmConfig import get_llm_model, get_llm_token
from db.VectorDb import VectorDb
from model.LlmProvider import BaseLLMProvider
from util.FileUtil import get_unique_filename, doc_clean_split

executor = ThreadPoolExecutor(max_workers=4)

class RagService:
    # Rag提示词模板
    PROMPT_CONTENT: str = """你是一位严谨的企业知识库智能助手，严格遵循信息检索原则。请根据以下提供的上下文内容，精准回答用户问题。请遵守以下准则：
            1. **来源限定**：仅使用上下文中的信息作答，禁止引入外部知识或个人推断。
            2. **忠实回应**：若上下文中缺乏足够信息，直接回答：“未能检索到相关内容。”不得编造或推测。
            3. **引用与高亮**：对于答案中的关键事实或数据，并将引用的原文片段用**双星号**高亮显示，并且在引用之后将其引用来源展示，也使用**双星号**高亮显示。
            4. **回答结构**：先直接回答问题，再附上引用说明。保持回答简洁、相关，避免冗余。
            
            上下文：
            {context}

            问题：
            {question}
            
            示例回答格式：
            - 有相关信息：根据上下文，XXX 是...**引用原文**...*来源文件*
            - 无相关信息：未能检索到相关内容。
            
            请开始你的回答："""

    # 最大文件大小（10MB）
    MAX_FILE_SIZE = 10 * 1024 * 1024
    MODEL_PATH = os.getenv('EMBEDDING_MODEL_PATH')
    UPLOAD_DIR = os.getenv('UPLOAD_DIR')
    TOP_K = int(os.getenv('TOP_K'))

    def __init__(self):
        self.vector = VectorDb()
        # 多轮对话历史存储
        self.session_histories = {}
        self.embedding_model = LocalEmbeddings(self.MODEL_PATH)

    def get_completion_token(self):
        """
                 当前使用的token数量
             :return: 流式输出的大模型结果
             """
        return get_llm_token()

    async def rag_qa(self, query: str, session_id: str = None):
        """
            RAG问答流程
        :param query: 用户问题
        :return: 流式输出的大模型结果
        """
        # 获取对应的大模型，模型名称可配置
        model_name = os.getenv('LLM_MODEL')

        llm = get_llm_model(model_name)

        # 向量库的集合可配置
        collection_name = os.getenv('RAG_COLLECTION')
        chat_history = []
        # 如果不传入session_id则表示为第一次进入聊天，直接生成一个session_id作为本次的会话id
        if session_id is None or session_id not in self.session_histories:
            session_id = str(uuid.uuid4())
            self.session_histories[session_id] = []
        else:
            for content_data in self.session_histories[session_id]:
                if "user" == content_data["type"]:
                    chat_history.append(HumanMessage(content=content_data["content"]))
                elif "assistant" == content_data["type"]:
                    chat_history.append(AIMessage(content=content_data["content"]))

        # 使用 async for 遍历生成器并yield结果
        async for chunk in self.rag_qa_vector_llm(query, collection_name, session_id, chat_history, llm):
            print(chunk)
            yield chunk

        # RAG问答函数

    async def rag_qa_vector_llm(self, query: str, collection_name: str, session_id: str,
                                chat_history: List[BaseMessage],
                                llm: BaseLLMProvider):
        """
             RAG和向量库以及大模型交互的核心逻辑
        :param session_id: 本轮对话id
        :param query: 用户问题
        :param collection_name:  向量库名称
        :param chat_history:  历史聊天记录
        :param llm:  大模型实例
        :return: 流式输出的大模型结果
        """
        logger.debug(f"\n问题: {query}")

        # 1. 创建集合(如果不存在)
        if not self.vector.create_vector_collection(collection_name):
            logger.error("创建集合失败")
            yield f"data: {json.dumps({'token': '创建集合失败', 'finished': True}, ensure_ascii=False)}\n\n"
            return
        # 2. 检索相关文档  top_K为配置项
        retrieved_docs = await self.retrieve_documents(collection_name, query, self.TOP_K)

        if not retrieved_docs:
            yield f"data: {json.dumps({'token': '未能检索到相关内容', 'finished': True}, ensure_ascii=False)}\n\n"
            return

        logger.info(f"\n检索到 {len(retrieved_docs)} 个相关文档片段")

        # 2. 如果提供了LLM，进行增强生成
        # 构建上下文
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])

        # 构建提示模板
        prompt_template = PromptTemplate(
            template=self.PROMPT_CONTENT,
            input_variables=["context", "question"]
        )

        prompt = prompt_template.format(context=context, question=query)
        self.session_histories[session_id].append({"content": prompt, "type": "user"})
        self.session_histories[session_id].append({"content": "", "type": "assistant"})
        chat_history.append(HumanMessage(content=prompt))
        try:
            full_response = ""
            # 流式调用LLM
            async for token in llm.generate(chat_history):
                if token:  # 确保token不为空
                    sse_data = {
                        "token": token,
                        "session_id": session_id,
                        "finished": False,
                    }
                    full_response += token
                    yield f"data: {json.dumps(sse_data, ensure_ascii=False)}\n\n"
            # 结果存下来，后续历史对话使用
            self.session_histories[session_id][-1]["content"] = full_response
            logger.info("完成")
            # 发送完成信号
            yield f"data: {json.dumps({'token': '', "session_id": session_id, 'finished': True}, ensure_ascii=False)}\n\n"

        except Exception as e:
            logger.error(f"LLM生成回答失败: {e}")
            error_data = {
                "token": f"生成回答时出错: {str(e)}",
                "session_id": session_id,
                "finished": True
            }
            yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n"

    async def rag_file_upload(self, file: UploadFile):
        """
          上传单个文件
          """
        try:
            filename = file.filename
            # 生成唯一文件名
            unique_filename = get_unique_filename(file.filename)

            # 确保上传目录存在
            os.makedirs(self.UPLOAD_DIR, exist_ok=True)

            file_path = os.path.join(self.UPLOAD_DIR, unique_filename)

            # 保存文件
            content = await file.read()
            # 检查文件大小
            if len(content) > self.MAX_FILE_SIZE:
                raise HTTPException(
                    status_code=400,
                    detail=f"文件太大。最大允许: {self.MAX_FILE_SIZE / 1024 / 1024}MB"
                )
            with open(file_path, "wb") as f:
                f.write(content)
            # 文件存入向量库的方法
            await self.rag_file_upload_to_vector_db(file_path, filename, os.getenv('RAG_COLLECTION'))
            return JSONResponse(
                status_code=200,
                content={
                    "message": "文件上传成功",
                    "filename": file.filename,
                    "saved_as": unique_filename,
                    "size": len(content),
                    "path": file_path
                }
            )
        except Exception as e:
            logger.error(f"错误为:{e}")
            raise HTTPException(status_code=500, detail=f"上传失败: {str(e)}")

    async def rag_file_upload_to_vector_db(self, file_path, filename, collection_name):
        """
            rag的文件上传功能
        :param file_path: 文件路径
        :param filename:  文件名称
        :param collection_name:  对应的集合名称
        :return:
        """
        # 1. 创建集合(如果不存在)
        if not self.vector.create_vector_collection(collection_name):
            logger.error("创建集合失败")
            return
        cleaned_docs = doc_clean_split(file_path)
        # 保存入库
        await self.insert_documents_to_vector(collection_name, filename, cleaned_docs)
        return True

    async def retrieve_documents(self, collection_name, query, top_k):
        """异步检索相关文档"""
        try:
            loop = asyncio.get_event_loop()

            # 异步执行向量嵌入
            query_vector = await loop.run_in_executor(
                executor,
                self.embedding_model.embed_query,
                query
            )

            # 异步执行 Milvus 搜索
            search_results = await loop.run_in_executor(
                executor,
                lambda: self.vector.search(collection_name, query_vector, top_k)
            )

            # 提取相关文档
            retrieved_docs = []
            for hits in search_results:
                for hit in hits:
                    doc = Document(
                        page_content=f"内容为:{hit['entity']['content']}，引用来源为:{hit['entity']['filename']}",
                        metadata=hit['entity']['metadata']
                    )
                    retrieved_docs.append(doc)

            return retrieved_docs

        except Exception as e:
            logger.error(f"检索文档失败：{e}")
            return []

    async def insert_documents_to_vector(self, collection_name, filename, documents):
        """异步将文档向量化并插入到 Milvus（带去重功能）"""
        try:
            loop = asyncio.get_event_loop()

            # 准备数据
            contents = [doc.page_content for doc in documents]
            metadatas = [doc.metadata for doc in documents]

            # 1. 计算内容哈希
            content_hashes = [hashlib.md5(content.encode()).hexdigest() for content in contents]

            # 2. 检查是否已存在（通过 content_hash 查询）
            try:
                existing_data = await loop.run_in_executor(
                    executor,
                    lambda: self.vector.query(collection_name, content_hashes, "content_hash")
                )
                existing_hashes = {item['content_hash'] for item in existing_data}
                logger.info(f"检测到 {len(existing_hashes)} 个已存在的内容")
            except Exception as e:
                logger.warning(f"查询已存在内容失败：{e}，将插入所有内容")
                existing_hashes = set()

            # 3. 过滤掉已存在的内容
            new_contents = []
            new_metadatas = []
            new_hashes = []

            for content, content_hash, metadata in zip(contents, content_hashes, metadatas):
                if content_hash not in existing_hashes:
                    new_contents.append(content)
                    new_metadatas.append({**metadata, 'content_hash': content_hash})
                    new_hashes.append(content_hash)

            if not new_contents:
                logger.info("所有内容已存在，跳过插入")
                return True

            logger.info(f"检测到 {len(new_contents)} 个新内容，准备生成向量...")

            # 4. 异步生成嵌入向量（只为新内容）
            new_vectors = await loop.run_in_executor(
                executor,
                self.embedding_model.embed_documents,
                new_contents
            )

            # 5. 准备插入数据
            data = []
            for content, vector, metadata, content_hash in zip(new_contents, new_vectors, new_metadatas, new_hashes):
                data.append({
                    "content": content,
                    "vector": vector,
                    "filename": filename,
                    "metadata": metadata,
                    "content_hash": content_hash
                })

            # 6. 异步插入数据
            logger.info(f"正在向 Milvus 插入 {len(data)} 个新文档...")
            insert_result = await loop.run_in_executor(
                executor,
                self.vector.insert,
                collection_name,
                data
            )
            logger.info(f"成功插入 {len(insert_result['ids'])} 个新文档")

            # 7. 异步刷新集合
            await loop.run_in_executor(
                executor,
                self.vector.flush,
                collection_name
            )
            return True

        except Exception as e:
            logger.error(f"插入文档失败：{e}")
            return False
