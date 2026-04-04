from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda

from langchain_demo.my_llm import llm

# 提示词 --> llm --> 文本 --- 提示词2  --> llm --> 评分

prompt1 = PromptTemplate.from_template('给我写一篇关于{key_word}的{type}，字数不超过{count}')

prompt2 = PromptTemplate.from_template('请检单评价一下这篇短文，如果总分是10分，请给这篇短文打分： {text_content}')

# 整个需求的第一段，组成一个链chain
chain1 = prompt1 | llm | StrOutputParser()

# 第一种
# chain2 = {'text_content': chain1} | prompt2 | llm | StrOutputParser()

# print(chain2.invoke({'key_word': '青春', 'type': '散文', 'count': 400}))

# 第二种
def print_chain1(input1):
    print(input1)
    print('---' * 30)
    return {'text_content': input1}

chain2 = chain1 | RunnableLambda(print_chain1) | prompt2 | llm | StrOutputParser()
print(chain2.invoke({'key_word': '青春', 'type': '散文', 'count': 400}))
