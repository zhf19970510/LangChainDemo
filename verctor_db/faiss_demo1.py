from typing import List

import faiss
from langchain_community.docstore import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from embeddings_demo.custom_embedding import CustomQwen3Embeddings

# model_name = "BAAI/bge-large-zh-v1.5"
# model_kwargs = {'device': 'cpu'}
# encode_kwargs = {'normalize_embeddings': True} # set True to compute cosine similarity
#
# bge_hf_embedding = HuggingFaceEmbeddings(
#     model_name=model_name,
#     model_kwargs=model_kwargs,
#     encode_kwargs=encode_kwargs
# )


qwen_embedding = CustomQwen3Embeddings("Qwen/Qwen3-Embedding-0.6B")

# 向量数据不等于关系型数据库
# FAISS向量数据库： pip install faiss-cpu

# 1、初始化数据库
# 先创建索引
# index = faiss.IndexFlatL2(1024)
index = faiss.IndexFlatL2(len(qwen_embedding.embed_query('Hello world!')))
vector_store = FAISS(
    embedding_function=qwen_embedding,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={}
)


# 2、 准备数据（Document）
document_1 = Document(
    page_content="今天早餐我吃了巧克力薄煎饼和炒蛋。",
    metadata={"source": "tweet", "time": "上午"},
)

document_2 = Document(
    page_content="明天的天气预报是阴天多云，最高气温62华氏度。",
    metadata={"source": "news"},
)

document_3 = Document(
    page_content="正在用LangChain构建一个激动人心的新项目——快来看看吧！",
    metadata={"source": "tweet"},
)

document_4 = Document(
    page_content="劫匪闯入城市银行，盗走了100万美元现金。",
    metadata={"source": "news"},
)

document_5 = Document(
    page_content="哇！那部电影太精彩了，我已经迫不及待想再看一遍。",
    metadata={"source": "tweet"},
)

document_6 = Document(
    page_content="新iPhone值得这个价格吗？阅读这篇评测一探究竟。",
    metadata={"source": "website"},
)

document_7 = Document(
    page_content="当今世界排名前十的足球运动员。",
    metadata={"source": "website"},
)

document_8 = Document(
    page_content="LangGraph是构建有状态智能体应用的最佳框架！",
    metadata={"source": "tweet"},
)

document_9 = Document(
    page_content="由于对经济衰退的担忧，今日股市下跌500点。",
    metadata={"source": "news"},
)

document_10 = Document(
    page_content="我有种不好的预感，我要被删除了 :(",
    metadata={"source": "tweet"},
)

documents = [
    document_1,
    document_2,
    document_3,
    document_4,
    document_5,
    document_6,
    document_7,
    document_8,
    document_9,
    document_10,
]

ids = [ 'id'+str(i+1) for i in range(len(documents))]

vector_store.add_documents(documents, ids=ids)


# results = vector_store.similarity_search('今天的金融投资新闻', k=2)
results = vector_store.similarity_search('有美食的内容吗', k=2)
for res in results:
    print(type(res))
    print(res.id)
    print(f"* {res.page_content} [{res.metadata}]")

