import requests
import os
from typing import List, Dict
import numpy as np
from kgpipe.util.embeddings.emb import Embedder
from tqdm import tqdm

TOKEN = None

def get_token():
    global TOKEN
    if TOKEN is None:
        TOKEN = os.getenv("OPENWEBUI_API_KEY")
    return TOKEN

class OpenWebUIEmbedder(Embedder):
    def __init__(self):
        super().__init__("openwebui")
    
    def encode_partially(self, text_list: List[str]) -> np.ndarray:

        headers = {
            "Authorization": f"Bearer {get_token()}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": "vllm-baai-bge-m3",
            "input": text_list,
        }

        response = requests.post(
            "https://kiara.sc.uni-leipzig.de/api/embeddings",
            headers=headers,
            json=payload,
        )

        return [ e["embedding"] for e in response.json()["data"] ]        

    def encode(self, text_list: List[str]) -> np.ndarray: 
        """Encode all texts in batches of 100 with tqdm progress bar."""
        all_embeddings = []
        batch_size = 100

        for i in range(0, len(text_list), batch_size): #, desc="Encoding text batches"):
            batch = text_list[i : i + batch_size]
            embeddings = self.encode_partially(batch)
            all_embeddings.extend(embeddings)

        return np.array(all_embeddings)


    def encode_as_dict(self, text_list: List[str]) -> Dict[str, np.ndarray]:
        embeddings = self.encode(text_list)
        embedding_map = {text: embedding for text, embedding in zip(text_list, embeddings)}
        return embedding_map