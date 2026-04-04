import json
from typing import Optional

from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser, SimpleJsonOutputParser
from langchain_core.prompts import MessagesPlaceholder, \
    FewShotChatMessagePromptTemplate, PromptTemplate
from pydantic import BaseModel, Field

from langchain_demo.my_llm import llm

from langchain_core.prompts import ChatPromptTemplate



class ResponseFormatter(BaseModel):
    """始终使用此工具来结构化你的用户响应"""  # 文档字符串说明这个类用于格式化响应

    answer: str = Field(description="对用户问题的回答")  # 回答内容字段
    followup_question: str = Field(description="用户可能提出的后续问题")  # 后续问题字段



runnable = llm.bind_tools([ResponseFormatter])

resp = runnable.invoke("细胞的动力源是什么？")
print(resp.tool_calls[-1]['args'])
resp.pretty_print()