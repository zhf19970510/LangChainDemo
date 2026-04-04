from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RouterRunnable, RunnableSequence

from langchain_demo.my_llm import llm

# 需求：用户会问到各种领域（数学、物理、历史等）的问题，根据不同的领域，定义不同的提示词模板。动态地选择合适的任务模板去完成

# 定义物理任务模板
physics_template = ChatPromptTemplate.from_template(
    "你是一位物理学教授，擅长用简洁易懂的方式回答物理问题。以下是问题内容：{input}"
)

# 定义数学任务模板
math_template = ChatPromptTemplate.from_template(
    "你是一位数学家，擅长分步骤解决数学问题，并提供详细的解决过程。以下是问题内容：{input}"
)

# 定义历史任务模板
history_template = ChatPromptTemplate.from_template(
    "你是一位历史学家，对历史事件和背景有深入研究。以下是问题内容：{input}"
)

# 定义计算机科学任务模板
computer_science_template = ChatPromptTemplate.from_template(
    "你是一位计算机科学专家，擅长算法、数据结构和编程问题，以下是问题内容：{input}"
)

default_template = ChatPromptTemplate.from_template(
    "输入内容无法归类，请直接回答：{input}"
)

default_chain = default_template | llm
physics_chain = physics_template | llm
math_chain = math_template | llm
history_chain = history_template | llm
computer_science_chain = computer_science_template | llm

# 动态路由的chain
def route(input):
    """根据大模型第一次分类处理之后的输出来动态判断各种领域的任务"""
    if '物理' in input['type']:
        print('1号')
        return {"key": 'physics', "input": input['input']}
    elif '数学' in input['type']:
        print('2号')
        return {"key": 'math', "input": input['input']}
    elif '历史' in input['type']:
        print('3号')
        return {"key": 'history', "input": input['input']}
    elif '计算机科学' in input['type']:
        print('4号')
        return {"key": 'computer_science', "input": input['input']}
    else:
        print('5号')
        return {"key": 'default', "input": input['input']}

# 创建一个路由节点
route_runnable = RunnableLambda(route)

# 路由调度器
router = RouterRunnable(runnables={ # 所有各个领域对应的任务 字典
    'physics': physics_chain,
    'math': math_chain,
    'history': history_chain,
    'computer_science': computer_science_chain,
    'default': default_chain
})

# 第一个提示词模板：
first_prompt = ChatPromptTemplate.from_template(
    "不要回答下面用户的问题，只要根据用户的输入来判断分类，一共有[物理、历史、计算机科学，数学]5种类别 \n\n \
    用户的输入：{input} \n\n \
    最后的输出包含分类的类别和用户输入的内容，输出格式为json，其中，类别的key为type，用户输入的key为input"
)

chain1 = first_prompt | llm | JsonOutputParser()

chain = RunnableSequence(chain1, route_runnable, router, StrOutputParser())    # 替代写法：chain = chain1 | route_runnable | router

inputs = [
    {"input": "什么是黑体辐射"},   # 物理问题
    {"input": "计算 2 + 2 的结果。"}, # 数学问题
    {"input": "介绍第一次世界大战的背景。"},  # 历史问题
    {"input": "如何实现快速排序算法？"}    # 计算机科学问题
]

for inp in inputs:
    result = chain.invoke(inp)
    print(f'问题：{inp} \n 回答： {result} \n')