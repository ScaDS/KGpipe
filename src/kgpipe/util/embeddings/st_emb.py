from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List
from kgpipe.util.embeddings.emb import Embedder
import numpy as np

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

# lazy load the model
model = None

def get_model() -> SentenceTransformer:
    global model
    if model is None:
        from sentence_transformers import SentenceTransformer
        import torch

        # disable pulling newer images
        model = SentenceTransformer("all-MiniLM-L6-v2")
        # if possible move to cuda if available
        if torch.cuda.is_available():
            model.to(torch.cuda.current_device())
    return model

class SentenceTransformerEmbedder(Embedder):
    def __init__(self):
        super().__init__("sentence-transformer")

    def encode_as_dict(self, text_list: List[str]) -> Dict[str, np.ndarray]:
        embedding_map = {}
        embeddings = self.encode(text_list)
        embedding_map = {text: embedding for text, embedding in zip(text_list, embeddings)}
        return embedding_map

    def encode(self, text_list: List[str]) -> np.ndarray:
        embeddings = get_model().encode(text_list, show_progress_bar=False)
        return embeddings
