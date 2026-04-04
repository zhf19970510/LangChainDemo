import json
from typing import Optional

from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser, SimpleJsonOutputParser
from langchain_core.prompts import MessagesPlaceholder, \
    FewShotChatMessagePromptTemplate, PromptTemplate
from pydantic import BaseModel, Field

from langchain_demo.my_llm import llm

from langchain_core.prompts import ChatPromptTemplate


# 创建聊天提示模板，要求模型以特定格式回答问题
prompt = ChatPromptTemplate.from_template(
    "尽你所能回答用户的问题。"  # 基本指令
    '你必须始终输出一个包含"answer"和"followup_question"键的JSON对象。其中"answer"代表：对用户问题的回答；"followup_question"代表：用户可能提出的后续问题'
    "{question}"  # 用户问题占位符
)


chain = prompt | llm | SimpleJsonOutputParser()
resp = chain.invoke({"question": "细胞的动力源是什么？"})
print(resp)