import os
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage
from typing import AsyncGenerator, Union, List

from langchain_openai import ChatOpenAI
import tiktoken

from model.LlmProvider import BaseLLMProvider
from loguru import logger


class OpenAiLlm(BaseLLMProvider):

    def __init__(self):
        load_dotenv()
        self.llm = ChatOpenAI(
            model_name=os.getenv('LLM_MODEL_NAME'),
            openai_api_key=os.getenv('LLM_API_KEY'),
            openai_api_base=os.getenv('LLM_API_BASE'),
            streaming=True  # 启用流式输出
        )
        self.encoder = tiktoken.get_encoding("cl100k_base")  # OpenAI使用的编码
        self.completion_tokens = 0
        logger.info("openAI的llm创建成功")

    async def generate(self, messages: Union[str, List[BaseMessage]]) -> AsyncGenerator[str, None]:
        """流式生成"""
        # 如果是字符串，包装成单条 HumanMessage
        if isinstance(messages, str):
            messages = [HumanMessage(content=messages)]
        try:
            async for chunk in self.llm.astream(messages):
                if hasattr(chunk, 'content'):
                    token = chunk.content
                else:
                    token = str(chunk)
                if token and token.strip():
                    token_count = len(self.encoder.encode(token))
                    self.completion_tokens += token_count
                    yield token
        except Exception as e:
            logger.error(f"流式生成错误: {e}")
            yield f"生成过程出错: {str(e)}"

    def get_token_num(self) -> int:
        return self.completion_tokens
