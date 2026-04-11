from langchain_chroma import Chroma
from langchain_core.documents import Document

from embeddings_demo.custom_embedding import CustomQwen3Embeddings

#  pip install -qU "langchain-chroma>=0.1.2"

qwen_embedding = CustomQwen3Embeddings("Qwen/Qwen3-Embedding-0.6B")


vector_store = Chroma(
    collection_name='t_news',
    embedding_function=qwen_embedding,
    persist_directory='../chroma_db'
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
results = vector_store.similarity_search_with_score('有美食的内容吗', k=4, filter={"source": 'tweet'})  # 带分数
for res, score in results:
    print(type(res))
    print(res.id)
    print(f"* [Score={score:3f}] {res.page_content} [{res.metadata}]")