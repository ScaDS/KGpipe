from typing import List, Dict
import os
from abc import ABC, abstractmethod
import numpy as np

class Embedder(ABC):
    def __init__(self, embedder_name: str):
        self.embedder_name = embedder_name
    
    @abstractmethod
    def encode_as_dict(self, texts: List[str]) -> Dict[str, np.ndarray]:
        pass

    @abstractmethod
    def encode(self, texts: List[str]) -> np.ndarray:
        pass

