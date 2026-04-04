import json
from typing import Optional

from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import MessagesPlaceholder, \
    FewShotChatMessagePromptTemplate, PromptTemplate
from pydantic import BaseModel, Field

from langchain_demo.my_llm import llm

from langchain_core.prompts import ChatPromptTemplate


# 生成一个笑话的段子： 三个属性，

# 使用pydantic定义一个类
class Joke(BaseModel):
    """笑话（搞笑段子）的结构类(数据模型类 POVO)"""

    setup: str = Field(description="笑话的开头部分")  # 笑话的铺垫部分
    punchline: str = Field(description="笑话的包袱/笑点")  # 笑话的爆笑部分
    rating: Optional[int] = Field(description="笑话的有趣程度评分，范围1到10")  # 可选的笑话评分字段


prompt_template = PromptTemplate.from_template("帮我生成一个关于{topic}的笑话。")
runnable = llm.with_structured_output(Joke)     # 只有一些模型支持with_structured_output，并不是所有模型都支持。比如deepseek-reasoner不支持  with_structured_output

chain = prompt_template | runnable
resp = chain.invoke({"topic": "猫"})
# resp = chain.invoke("讲个笑话")
print(resp)
print(str(resp))

print(resp.__dict__)

json_str = json.dumps(resp.__dict__)
print(json_str)