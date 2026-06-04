from model.LlmProvider import BaseLLMProvider
from model.MockAiLlm import MockAiLlm
from model.OpenAiLlm import OpenAiLlm

llmConfig = {
    "openAi": OpenAiLlm(),
    "default": MockAiLlm()
}


# 根据字符串获取对应的大模型实现方法
def get_llm_model(modelName: str) -> BaseLLMProvider:
    if modelName is None or modelName == "" or modelName not in llmConfig:
        return llmConfig["default"]
    else:
        return llmConfig[modelName]


def get_llm_token():
    res = {}
    for config in llmConfig:
        res[config] = llmConfig[config].get_token_num()
    return res
