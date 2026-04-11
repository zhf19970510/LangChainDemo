from langchain_openai import OpenAIEmbeddings

from env_utils import OPENAI_API_KEY, OPENAI_BASE_URL

openai_embeddings = OpenAIEmbeddings(
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL,
    model='text-embedding-3-large',
    dimensions=256
)

resp = openai_embeddings.embed_documents(
    [
        'I like large language models',
        '今天的天气非常不错'
    ]
)

print(resp[0])
print(len(resp[0]))

