from langchain_openai import ChatOpenAI

from env_utils import OPENAI_API_KEY, OPENAI_BASE_URL, LOCAL_BASE_URL, DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL
from langchain_demo.my_llm import llm

# openai的大模型
# llm = ChatOpenAI(
#     model='gpt-4o-mini',
#     temperature=0.8,
#     api_key=OPENAI_API_KEY,
#     base_url=OPENAI_BASE_URL
# )

# llm = ChatOpenAI(
#     model='claude-opus-4-6',
#     temperature=0.8,
#     api_key=OPENAI_API_KEY,
#     base_url=OPENAI_BASE_URL
# )

# llm = ChatOpenAI(
#     model='deepseek-reasoner',
#     temperature=0.8,
#     api_key=DEEPSEEK_API_KEY,
#     base_url=DEEPSEEK_BASE_URL
# )

# 本地私有化部署的大模型
# llm = ChatOpenAI(
#     model='qwen3-8b',
#     temperature=0.8,
#     api_key='xx',
#     base_url=LOCAL_BASE_URL,
#     # qwen3特有的参数：enable_thinking    表示开启深度思考
#     extra_body={'chat_template_kwargs': {'enable_thinking': True}},
# )


message = [
    ('system', '你是一个智能助手。'),
    ('human', '请介绍一下什么是深度学习?')
]

resp = llm.invoke(message)

print(resp)