import os
from typing import List, Dict
import kgpipe.util.embeddings.openwebui_emb as openwebui_emb
import kgpipe.util.embeddings.st_emb as st_emb
from kgpipe.util.embeddings.emb import Embedder
import numpy as np

embedder = None


# in your module where get_embedder() lives
from kgpipe.util.embeddings.cache import (
    CachingEmbedder, InMemoryLRUCache, SQLiteCache
)

def _make_cache_from_env():
    spec = os.getenv("EMBED_CACHE", "memory").strip()
    ttl = int(os.getenv("EMBED_CACHE_TTL", "0")) or None  # seconds; 0 = no TTL

    if spec == "memory":
        print("Using in-memory cache")
        capacity = int(os.getenv("EMBED_CACHE_CAPACITY", "50000"))
        return InMemoryLRUCache(capacity=capacity, ttl_seconds=ttl)

    if spec.startswith("sqlite://"):
        print("Using sqlite cache")
        # e.g. EMBED_CACHE="sqlite:///tmp/emb_cache.db"
        path = spec[len("sqlite://"):]
        return SQLiteCache(path=path, ttl_seconds=ttl)

    if spec.startswith("redis://"):
        print("Using redis cache")
        # e.g. EMBED_CACHE="redis://localhost:6379"
        url = spec
        from kgpipe.util.embeddings.chache_redis import RedisCache
        return RedisCache(url=url, ttl_seconds=ttl)

    raise ValueError(f"Unknown EMBED_CACHE backend: {spec}")

def get_embedder() -> Embedder:
    global embedder
    if embedder is None:
        # pick the real embedder first
        if os.getenv("EMBEDDER") == "openwebui":
            base = openwebui_emb.OpenWebUIEmbedder()
            model_id = getattr(base, "model_id", os.getenv("OPENWEBUI_MODEL", "openwebui-default"))
        elif os.getenv("EMBEDDER") == "sentence-transformer":
            base = st_emb.SentenceTransformerEmbedder()
            model_id = getattr(base, "model_name", os.getenv("ST_MODEL", "st-default"))
        else:
            raise ValueError(f"Invalid embedder: {os.getenv('EMBEDDER')}")

        # wrap with cache if enabled
        if os.getenv("EMBED_CACHE_DISABLE", "0") == "1":
            embedder = base
        else:
            cache = _make_cache_from_env()
            ttl = int(os.getenv("EMBED_CACHE_TTL", "0")) or None
            embedder = CachingEmbedder(inner=base, cache=cache, model_id=model_id, default_ttl_seconds=ttl)
    return embedder


def global_encode_as_dict(texts: List[str]) -> Dict[str, np.ndarray]:
    return get_embedder().encode_as_dict(texts)

def global_encode(texts: List[str]) -> np.ndarray:
    return get_embedder().encode(texts)