from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import  MessagesPlaceholder, \
    FewShotChatMessagePromptTemplate

from langchain_demo.my_llm import llm

from langchain_core.prompts import ChatPromptTemplate


# ICL:
#  2 🦜 9 的结果是多少？

examples = [
    {"input": "2 🦜 2", "output": "4"},
    {"input": "2 🦜 3", "output": "6"},
]

# 单个用户输入和AI回复的模板
base_prompt = ChatPromptTemplate.from_messages(
    [
        ('human', '{input}'),
        ('ai', '{output}'),
    ]
)

# 包含示例的提示词模板
few_shot_prompt = FewShotChatMessagePromptTemplate(
    examples=examples,
    example_prompt=base_prompt,
)

# 聊天模板
final_template = ChatPromptTemplate.from_messages([
    ("system", "你是智能机器人AI助手！"),
    few_shot_prompt,
    MessagesPlaceholder("msgs")  #
])


# chain = final_template | llm
# 加入输出解析器
chain = final_template | llm | StrOutputParser()
print(chain.invoke({"msgs": [HumanMessage(content="2 🦜 9 的结果是多少？")]}))
# resp = chain.invoke({"msgs": [HumanMessage(content="中国最后一个皇帝是谁？")]})
# print(resp.content)
# print(resp)
