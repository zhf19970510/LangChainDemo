import bs4

from langchain_chroma import Chroma
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.history_aware_retriever import create_history_aware_retriever
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter

from embeddings_demo.custom_embedding import CustomQwen3Embeddings
from langchain_demo.my_llm import llm

qwen_embedding = CustomQwen3Embeddings("Qwen/Qwen3-Embedding-0.6B")

# 构建向量数据库
vector_store = Chroma(
    collection_name='t_agent_blog',
    embedding_function=qwen_embedding,
    persist_directory='../chroma_db'
)


def create_dense_db():
    """把网络的关于Agent的博客数据写入向量数据库"""
    loader = WebBaseLoader(
        web_path=('https://lilianweng.github.io/posts/2023-06-23-agent/',),
        bs_kwargs=dict(
            #  pip install beautifulsoup4
            parse_only=bs4.SoupStrainer(  # 使用BeautifulSoup解析器，只解析特定class的内容
                class_=("post-content", "post-title", "post-header")  # 指定要解析的HTML类名
            )
        )
    )

    docs_list = loader.load()

    # 切割
    # 初始化文本分割器，设置块大小1000，重叠200
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    # 分割文档
    splits = text_splitter.split_documents(docs_list)

    print('doc的数量为:', len(splits))
    ids = ['id' + str(i + 1) for i in range(len(splits))]
    # 把doc写到向量数据库
    vector_store.add_documents(documents=splits, ids=ids)


# create_dense_db()

### 问题上下文化 ###
# 系统提示词：用于将带有聊天历史的问题转化为独立问题
contextualize_q_system_prompt = (
    "给定聊天历史和最新的用户问题（可能引用聊天历史中的上下文），"
    "将其重新表述为一个独立的问题（不需要聊天历史也能理解）。"
    "不要回答问题，只需在需要时重新表述问题，否则保持原样。"
)

# 创建聊天提示模板
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),  # 系统角色提示
        MessagesPlaceholder("chat_history"),  # 聊天历史占位符
        ("human", "{input}"),  # 用户输入占位符
    ]
)

# 创建一个向量数据库的检索器
retriever = vector_store.as_retriever(search_kwargs={'k': 2})

# 创建一个上下文感知的检索器
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

# RAG的代码

### 回答问题 ###
# 系统提示词：定义助手的行为和回答规范
system_prompt = (
    "你是一个问答任务助手。"
    "使用以下检索到的上下文来回答问题。"
    "如果不知道答案，就说你不知道。"
    "回答最多三句话，保持简洁。"
    "\n\n"
    "{context}"  # 从向量数据库中检索出来的doc
)
# 创建问答提示模板
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),  # 系统角色提示
        MessagesPlaceholder("chat_history"),  # 聊天历史占位符
        ("human", "{input}"),  # 用户输入占位符
    ]
)

# 创建文档处理链
question_chain = create_stuff_documents_chain(llm, qa_prompt)

# 创建RAG检索链
rag_chain = create_retrieval_chain(history_aware_retriever, question_chain)

store = {}  # 用来保存历史消息, key : 会话ID session_id


def get_session_history(session_id: str):
    """从内存中的历史消息列表中 返回当前会话 的所有历史消息"""
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


# 创建带历史记录功能的处理链
conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key='input',
    history_messages_key='chat_history',
    output_messages_key='answer',  # 输出消息的键
)

# 调用会话式RAG链，询问"什么是任务分解？"
resp1 = conversational_rag_chain.invoke(
    {"input": "What is Task Decomposition?"},  # 用户输入问题
    config={
        "configurable": {"session_id": "abc123"}  # 使用会话ID "abc123" 保持对话历史
    }
)

print(resp1['answer'])

resp2 = conversational_rag_chain.invoke(
    {"input": "What are common ways of doing it?"},  # 用户输入问题
    config={
        "configurable": {"session_id": "abc123"}  # 使用会话ID "abc123" 保持对话历史
    }
)

print(resp2['answer'])
