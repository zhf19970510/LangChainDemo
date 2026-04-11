from openai import OpenAI

from env_utils import OPENAI_API_KEY, OPENAI_BASE_URL

# 不用langchain
client = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL
)

text = 'I like large language models'
resp = client.embeddings.create(
    model='text-embedding-3-large',
    dimensions=512,
    input=text
)
print(resp.data[0].embedding)
print(len(resp.data[0].embedding))






