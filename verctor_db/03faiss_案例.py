from typing import List

import faiss
from langchain_community.docstore import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from embeddings_demo.custom_embedding import CustomQwen3Embeddings

qwen_embedding = CustomQwen3Embeddings("Qwen/Qwen3-Embedding-0.6B")

# 把数据库写入磁盘
vector_store = FAISS.load_local('../faiss_db', embeddings=qwen_embedding, allow_dangerous_deserialization=True)

vector_store.delete(ids=['id10'])

# results = vector_store.similarity_search('今天的金融投资新闻', k=2)
results = vector_store.similarity_search_with_score('有美食的内容吗', k=4, filter={"source": 'tweet'})  # 带分数
for res, score in results:
    print(type(res))
    print(res.id)
    print(f"* [Score={score:3f}] {res.page_content} [{res.metadata}]")


