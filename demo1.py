from openai import OpenAI

client = OpenAI(base_url="http://localhost:6006/v1", api_key="xxxx")

resp = client.chat.completions.create(
    model='qwen3-8b',
    messages=[{'role': 'user', 'content': '请介绍一下什么是深度学习？'}],
    temperature=0.8,
    presence_penalty=1.5,
    # qwen3特有的参数：enable_thinking    表示开启深度思考
    extra_body={'chat_template_kwargs': {'enable_thinking': True}},
)

print(resp)
print(resp.choices[0].message.content)
