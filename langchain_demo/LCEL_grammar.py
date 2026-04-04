import time

from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_core.tracers import Run


def test1(x: int) -> int:
    return x + 10


# 节点：标准：Runnable

r1 = RunnableLambda(test1)  # 把test1函数封装为一个节点或者组件

# invoke 普通调用
res = r1.invoke(4)
print(res)

# 批量调用
res = r1.batch([4, 5])
print(res)


# 流式运行
def test2(prompt: str):
    for item in prompt.split(' '):
        yield item


r1 = RunnableLambda(test2)
res = r1.stream('This is a dog.')  # 返回的是一个生成器

for chunk in res:
    print(chunk)

# 4. 组合链
r1 = RunnableLambda(test1)
r2 = RunnableLambda(lambda x: x * 2)

chain1 = r1 | r2
print(f'组合链结果:{chain1.invoke(2)}')

# 5. 并行运行
chain2 = RunnableParallel(r1=r1, r2=r2)
# max_concurrency：最大并发数
res = chain2.invoke(2, config={'max_concurrency': 2})
print(f'并行运行结果:{res}')

# 组合链和并行运行结合使用
chain = chain1 | chain2
res = chain.invoke(2)
print(f'组合链和并行运行结合使用结果:{res}')
chain.get_graph().print_ascii()  # 打印链的图像描述

# 6. 合并输入并处理中间数据
# RunnablePassthrough:  # 允许传递输入的数据，可以保持不变 或 添加额外的键，必须传入一个字典数据，还可以过滤
r1 = RunnableLambda(lambda x: {'key1': x})
r2 = RunnableLambda(lambda x: x['key1'] + 10)
chain = r1 | r2
print(chain.invoke(2))
chain2 = r1 | RunnablePassthrough.assign(new_key=r2)  # new_key, 随意定制的，代表输出的key
print(chain2.invoke(2))
chain3 = r1 | RunnablePassthrough() | RunnablePassthrough.assign(new_key=r2)  # new_key, 随意定制的，代表输出的key
print(chain3.invoke(2))

chain3 = r1 | RunnableParallel(foo=RunnablePassthrough(), new_key=RunnablePassthrough.assign(key2=r2))
print(chain3.invoke(2))

chain4 = r1 | RunnableParallel(foo=RunnablePassthrough(),
                               new_key=RunnablePassthrough.assign(key2=r2)) | RunnablePassthrough().pick(['new_key'])
print(chain4.invoke(2))

# 7. 后备选项：后备选项是一种可以在紧急情况下使用的替代方案。
r1 = RunnableLambda(test1)
r2 = RunnableLambda(lambda x: int(x) + 20)
# 在加法计算中的后备选项
chain = r1.with_fallbacks([r2])  # r2是r1的后备方案，r1报错的情况下会执行r2
print(chain.invoke('2'))

# 根据条件，动态地构建链
r1 = RunnableLambda(test1)
r2 = RunnableLambda(lambda x: [x] * 2)
# 根据r1的输出结果，判断是否要执行r2（判断本身也是一个节点）
chain = r1 | RunnableLambda(lambda x: r2 if x > 12 else RunnableLambda(lambda y: y))
print(chain.invoke(2))


# 生命周期管理
def test4(n: int) -> int:
    time.sleep(n)
    return n * 2


def on_start(run_obj: Run):
    """当r1节点启动的时候，自动调用"""
    print('r1启动的时间：', run_obj.start_time)


def on_end(run_obj: Run):
    """当r1节点已经运行结束的时候，自动调用"""
    print('r1结束的时间：', run_obj.end_time)


r1 = RunnableLambda(test4)

chain = r1.with_listeners(on_start=on_start, on_end=on_end)
print(chain.invoke(2))