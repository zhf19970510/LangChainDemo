from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory, RunnablePassthrough
from langchain_core.stores import InMemoryStore

from langchain_demo.my_llm import llm

# 1. 提示词模板
prompt = ChatPromptTemplate.from_messages([
    ('system', '你是一个乐于助人的助手。尽你所能回答所有问题。提供的聊天历史包含与你对话用户的相关信息。'),
    MessagesPlaceholder(variable_name='chat_history', optional=True),
    # ("placeholder", "{chat_history}")
    ('human', '{input}')
])

chain = prompt | llm  # 基础的执行链


# 2. 存储聊天记录：（内存、关系型数据库或者redis数据库）

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

# 3. 创建带有历史记录的处理链
chain_with_message_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key='input',
    history_messages_key='chat_history'
)


# 4. 剪辑和摘要上下文，历史记录：保留最近的前2条消息不要做摘要，也不要删掉；把之前的所有消息形成摘要
def summarize_messages(current_input):
    """剪辑和摘要上下文的历史聊天记录"""
    session_id = current_input['config']['configurable']['session_id']
    if not session_id:
        raise ValueError("必须通过config参数提供session_id")

    # 获取当前会话ID的所有历史聊天记录
    chat_history = get_session_history(session_id)
    stored_messages = chat_history.messages
    if len(stored_messages) <= 2:  # 保留最近2条消息的阈值
        return False

    # 剪辑消息列表
    last_two_messages = stored_messages[-2:]  # 保留的2条消息
    messages_to_summarize = stored_messages[:-2]  # 需要进行摘要的消息列表
    summarization_prompt = ChatPromptTemplate.from_messages([
        ('system', '请将以下对话历史压缩为一条关键信息的摘要信息。'),
        ('placeholder', '{chat_history}'),
        ('human', '请生成包含上述对话核心内容的摘要，保留重要事实和决策。')
    ])
    summarization_chain = summarization_prompt | llm
    # 生成摘要
    summary_message = summarization_chain.invoke({'chat_history': messages_to_summarize})

    # 重建历史记录：摘要 + 最后2条原始消息
    chat_history.clear()
    chat_history.add_message(summary_message)
    for msg in last_two_messages:
        chat_history.add_message(msg)
    return True


# 最终的链
# RunnablePassthrough 默认会将输入数据原样传递给下游，而 .assign() 方法允许在保留原始输入的同时，通过指定键值对（如messages_summarized=summarize_messages 向输入字典中加入新字段
final_chain = (RunnablePassthrough.assign(messages_summarized=summarize_messages) | chain_with_message_history)


result1 = final_chain.invoke({'input': '你好，我是张学良？', 'config': {'configurable': {'session_id': 'user123'}}},
                                            config={"configurable": {"session_id": "user123"}})
print(result1)

result1 = final_chain.invoke({'input': '你好，我的名字叫什么？', 'config': {'configurable': {'session_id': 'user123'}}},
                                            config={"configurable": {"session_id": "user123"}})
print(result1)

result2 = final_chain.invoke({'input': '历史上，和我同名的人有哪些？', 'config': {'configurable': {'session_id': 'user123'}}},
                                            config={"configurable": {"session_id": "user123"}})
print(result2)
