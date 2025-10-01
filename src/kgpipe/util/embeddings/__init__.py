import os
from typing import List, Dict
import kgpipe.util.embeddings.openwebui_emb as openwebui_emb
import kgpipe.util.embeddings.st_emb as st_emb
from kgpipe.util.embeddings.emb import Embedder
import numpy as np

embedder = None

def get_embedder() -> Embedder:
    global embedder
    if embedder is None:
        if os.getenv("EMBEDDER") == "openwebui":
            embedder = openwebui_emb.OpenWebUIEmbedder()
        elif os.getenv("EMBEDDER") == "sentence-transformer":
            embedder = st_emb.SentenceTransformerEmbedder()
        else:
            raise ValueError(f"Invalid embedder: {os.getenv('EMBEDDER')}")
    return embedder


def global_encode_as_dict(texts: List[str]) -> Dict[str, np.ndarray]:
    return get_embedder().encode_as_dict(texts)

def global_encode(texts: List[str]) -> np.ndarray:
    return get_embedder().encode(texts)