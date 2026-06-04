from langchain_core.messages import BaseMessage
from typing import AsyncGenerator, Union, List

from abc import abstractmethod
from  loguru import logger


class BaseLLMProvider:

    def __init__(self):
        logger.info("工厂方法")

    @abstractmethod
    async def generate(self,  messages:Union[str, List[BaseMessage]]) -> AsyncGenerator[str, None]:
        pass

    @abstractmethod
    def get_token_num(self)->int:
        pass