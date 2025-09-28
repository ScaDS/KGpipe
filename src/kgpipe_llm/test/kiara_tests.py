from typing import AnyStr
from kgpipe_llm.common import LLMClient
from dotenv import load_dotenv
import os
from pydantic import BaseModel
import requests
import time
import numpy as np
load_dotenv()

# client = LLMClient(endpoint_url="https://kiara.sc.uni-leipzig.de/openai/chat/completions", model_name="vllm-llama-4-scout-17b-16e-instruct", token=os.getenv("KIARA_API_KEY") or "")

# print(client.send_message([{"role": "user", "content": "Hello, world!"}], "raw"))    



class CarDescription(BaseModel):
    name: str
    description: str

token = os.getenv("KIARA_API_KEY") or ""


# payload = {
#     "model": "vllm-llama-4-scout-17b-16e-instruct",
#     "messages": [
#         {
#             "role": "user",
#             "content": "Generate an example email address for Alan Turing, who works in Enigma. End in .com and new line. Example result: alan.turing@enigma.com\n",
#         }
#     ],
#     "extra_body": {"guided_regex": r"at+@\w+\.com\n", "stop": ["\n"]},
#     "stream": False,
#     # "messages": [{"role": "user", "content": "Make a car description for a red car."}],
#     # "stream": False,
#     # "response_format": {
#     #     "type": "json_schema",
#     #     "json_schema": {
#     #         "name": "car-description",
#     #         "schema": CarDescription.model_json_schema()
#     #     },
#     # },
# }

# headers = {
#     "Content-Type": "application/json",
# }

# if token and token != "":
#     headers["Authorization"] = f"Bearer {token}"

# response = requests.post(
#     "https://kiara.sc.uni-leipzig.de/openai/chat/completions",
#     headers=headers,
#     json=payload,
# )

# print(response.json())
            

# curl -i https://kiara.sc.uni-leipzig.de/api/embeddings   -H "Authorization: Bearer $API_KEY"   -H "Content-Type: application/json"   -d '{"model":"vllm-baai-bge-m3","input":["hello2","hello3"]}'

headers = {
    "Authorization": f"Bearer {token}",
    "Content-Type": "application/json",
}

# test_text_list = [f"hello{s}" for s in range(100)]

test_text_list = ["2025-05-02", "May 2nd 2025"]

payload = {
    "model": "vllm-baai-bge-m3",
    "input": test_text_list,
}

t0 = time.time()
response = requests.post(
    "https://kiara.sc.uni-leipzig.de/api/embeddings",
    headers=headers,
    json=payload,
)

embedding_map = {text: embedding["embedding"] for text, embedding in zip(test_text_list, response.json()["data"])}

t1 = time.time()
print(f"Time taken: {t1 - t0} seconds")

# print(len(embedding_map["The film of the titanic"]))
# print(len(embedding_map["Titanic 1997"]))

for i in range(len(test_text_list)):
    for j in range(i + 1, len(test_text_list)):
        print(test_text_list[i], test_text_list[j])
        print(np.dot(embedding_map[test_text_list[i]], embedding_map[test_text_list[j]]))
        print()