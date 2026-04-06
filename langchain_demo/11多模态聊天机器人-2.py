from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.stores import InMemoryStore

from langchain_demo.my_llm import llm

# 提示词模板
prompt = ChatPromptTemplate.from_messages([
    ('system', '你是一个乐于助人的助手。尽你所能回答所有问题。提供的聊天历史包含与你对话用户的相关信息。'),
    MessagesPlaceholder(variable_name='chat_history', optional=True),
    # ("placeholder", "{chat_history}")
    ('human', '{input}')
])

chain = prompt | llm  # 基础的执行链


# 存储聊天记录：（内存、关系型数据库或者redis数据库）

# 用来保存历史记录，key: 会话ID session_id

def get_session_history(session_id: str):
    """从关系型数据库的历史聊天消息列表中 返回当前会话 的所有历史消息"""
    return SQLChatMessageHistory(
        session_id=session_id,
        connection_string='sqlite:///chat_history.db',
    )


"""
所有消息都继承BaseMessage，总共有五种消息，分别是：
SystemMessage
HumanMessage
AIMessage
ToolMessage
"""

# 创建带有历史记录的处理链
chain_with_message_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key='input',
    history_messages_key='chat_history'
)

result1 = chain_with_message_history.invoke({'input': '你好，我是张学良？'},
                                            config={"configurable": {"session_id": "user123"}})
print(result1)

result1 = chain_with_message_history.invoke({'input': '你好，我的名字叫什么？'},
                                            config={"configurable": {"session_id": "user123"}})
print(result1)

result2 = chain_with_message_history.invoke({'input': '历史上，和我同名的人有哪些？'},
                                            config={"configurable": {"session_id": "user123"}})
print(result2)
