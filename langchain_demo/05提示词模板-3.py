from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate, ChatPromptTemplate, MessagesPlaceholder

from langchain_demo.my_llm import llm


# {topic} : 变量占位符
#   消息占位符
# prompt_template = ChatPromptTemplate.from_messages([
#     ("system", "你是一个幽默的电视台主持人！"),
#     ("user", "帮我生成一个简短的，关于{topic}的报幕词。")
# ])
# print(prompt_template.invoke({"topic": "相声"}))
#
# chain = prompt_template | llm
#
# res = chain.invoke({"topic": "相声"})
# print(res)

prompt_template = ChatPromptTemplate.from_messages([
    ("system", "你是一个幽默的电视台主持人！"),
    MessagesPlaceholder("input")
])

prompt_template.invoke({"input": [HumanMessage(content="你好，主持人!")]})


chain = prompt_template | llm

print(chain.invoke({"input": [HumanMessage(content="你好，主持人!")]}))