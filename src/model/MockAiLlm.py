import asyncio
from langchain_core.messages import BaseMessage
from typing import AsyncGenerator, Union, List

from loguru import logger

from model.LlmProvider import BaseLLMProvider


class MockAiLlm(BaseLLMProvider):

    def __init__(self):
        super().__init__()
        logger.info("mock的llm创建成功")

    async def generate(self, messages: Union[str, List[BaseMessage]]) -> AsyncGenerator[str, None]:
        mock_response = f"本次返回的是mock数据"

        # mock一个字一个字的输出
        for char in mock_response:
            yield char
            await asyncio.sleep(0.1)

    def get_token_num(self) -> int:
        return 0
