from zhipuai import ZhipuAI

from env_utils import ZHIPU_API_KEY

client = ZhipuAI(api_key=ZHIPU_API_KEY)  # 填写您自己的APIKey
with open(r"G:\code\LangChainProject\langchain_demo\happy_birthday.wav", "rb") as audio_file:
    resp = client.audio.transcriptions.create(
        model="glm-asr",
        file=audio_file,
        stream=False
    )
    # print(resp)
    print(resp.model_extra['text'])