from langchain_core.prompts import PromptTemplate

from langchain_demo.my_llm import llm

# prompt_template = PromptTemplate.from_template("帮我生成一个简短的，关于{topic}的报幕词。")

prompt = (  # 外层的模板
    PromptTemplate.from_template("帮我生成一个简短的，关于{topic}的报幕词。")
    + ", 要求： 1、内容搞笑一点；"
    + "2、输出的内容采用{language}。"
)


# res = prompt_template.invoke({"topic": "相声"})
# print(res)

chain = prompt | llm
resp = chain.invoke({"topic": "相声", "language": "English"})
print(resp)