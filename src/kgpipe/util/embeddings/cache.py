import os
import sqlite3
import time
import io
import hashlib
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Iterable, List, Dict, Optional, Tuple, Protocol

import numpy as np
from kgpipe.util.embeddings.emb import Embedder


# ---------- Cache backend protocol ----------

class CacheBackend(Protocol):
    def get_many(self, keys: List[str]) -> Dict[str, np.ndarray]:
        ...
    def set_many(self, items: Dict[str, np.ndarray], ttl_seconds: Optional[int] = None) -> None:
        ...
    def close(self) -> None:
        ...


# ---------- Keying helpers ----------

def _normalize_text(t: str) -> str:
    # Keep simple & deterministic. You can add unicode normalization if needed.
    return t.strip()

def _hash_text(text: str) -> str:
    h = hashlib.blake2b(digest_size=16)
    h.update(text.encode("utf-8"))
    return h.hexdigest()

def make_cache_key(embedder_name: str, model_id: str, text: str) -> str:
    # Include embedder + model in the key so you can switch models safely.
    norm = _normalize_text(text)
    return f"{embedder_name}:{model_id}:{_hash_text(norm)}"


# ---------- In-memory LRU cache ----------

class InMemoryLRUCache(CacheBackend):
    def __init__(self, capacity: int = 50_000, ttl_seconds: Optional[int] = None):
        self.capacity = capacity
        self.ttl = ttl_seconds
        self._lock = threading.RLock()
        # value: (np.ndarray, expiry_ts_or_none)
        self._store: "OrderedDict[str, Tuple[np.ndarray, Optional[float]]]" = OrderedDict()

    def _prune(self):
        while len(self._store) > self.capacity:
            self._store.popitem(last=False)

    def get_many(self, keys: List[str]) -> Dict[str, np.ndarray]:
        now = time.time()
        out: Dict[str, np.ndarray] = {}
        with self._lock:
            for k in keys:
                val = self._store.get(k)
                if val is None:
                    continue
                arr, expiry = val
                if expiry is not None and expiry < now:
                    # expired: delete
                    try:
                        del self._store[k]
                    except KeyError:
                        pass
                    continue
                # move to end (recently used)
                self._store.move_to_end(k, last=True)
                out[k] = arr
        return out

    def set_many(self, items: Dict[str, np.ndarray], ttl_seconds: Optional[int] = None) -> None:
        ttl = ttl_seconds if ttl_seconds is not None else self.ttl
        expiry = (time.time() + ttl) if ttl else None
        with self._lock:
            for k, arr in items.items():
                self._store[k] = (arr, expiry)
                self._store.move_to_end(k, last=True)
            self._prune()

    def close(self) -> None:
        with self._lock:
            self._store.clear()


# ---------- SQLite cache (persistent, safe for multi-process reads) ----------

_SQL_SCHEMA = """
CREATE TABLE IF NOT EXISTS embeddings_cache (
    key TEXT PRIMARY KEY,
    value BLOB NOT NULL,
    expiry REAL
);
CREATE INDEX IF NOT EXISTS idx_expiry ON embeddings_cache(expiry);
"""

def _np_to_blob(arr: np.ndarray) -> bytes:
    bio = io.BytesIO()
    # allow_pickle=False for safety
    np.save(bio, arr, allow_pickle=False)
    return bio.getvalue()

def _blob_to_np(blob: bytes) -> np.ndarray:
    bio = io.BytesIO(blob)
    return np.load(bio, allow_pickle=False)

class SQLiteCache(CacheBackend):
    def __init__(self, path: str, ttl_seconds: Optional[int] = None):
        self.ttl = ttl_seconds
        self.path = path
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.path, isolation_level=None, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA synchronous=NORMAL;")
        for stmt in filter(None, _SQL_SCHEMA.split(";")):
            s = stmt.strip()
            if s:
                self._conn.execute(s + ";")

    def get_many(self, keys: List[str]) -> Dict[str, np.ndarray]:
        if not keys:
            return {}
        now = time.time()
        placeholders = ",".join("?" for _ in keys)
        sql = f"SELECT key, value, expiry FROM embeddings_cache WHERE key IN ({placeholders})"
        out: Dict[str, np.ndarray] = {}
        with self._lock:
            for k, blob, expiry in self._conn.execute(sql, keys):
                if expiry is not None and expiry < now:
                    # lazily purge expired
                    self._conn.execute("DELETE FROM embeddings_cache WHERE key=?", (k,))
                    continue
                out[k] = _blob_to_np(blob)
        return out

    def set_many(self, items: Dict[str, np.ndarray], ttl_seconds: Optional[int] = None) -> None:
        if not items:
            return
        ttl = ttl_seconds if ttl_seconds is not None else self.ttl
        expiry = (time.time() + ttl) if ttl else None
        rows = [(k, _np_to_blob(v), expiry) for k, v in items.items()]
        with self._lock:
            self._conn.executemany(
                "INSERT OR REPLACE INTO embeddings_cache(key, value, expiry) VALUES (?, ?, ?)", rows
            )

    def close(self) -> None:
        with self._lock:
            self._conn.close()


# ---------- Caching wrapper that implements your Embedder ----------

@dataclass
class CachingEmbedder(Embedder):
    inner: Embedder
    cache: CacheBackend
    # model_id helps with invalidation when you change models
    model_id: str
    # optional TTL per call
    default_ttl_seconds: Optional[int] = None

    @property
    def name(self) -> str:
        # include the inner class name for uniqueness in keys
        return f"{self.inner.__class__.__name__}"

    def encode(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0))
        keys = [make_cache_key(self.name, self.model_id, t) for t in texts]

        hits = self.cache.get_many(keys)
        # figure out which indices are misses
        miss_indices = [i for i, k in enumerate(keys) if k not in hits]
        miss_texts = [texts[i] for i in miss_indices]

        if miss_texts:
            miss_embs = self.inner.encode(miss_texts)  # shape: (m, d)
            # store back
            to_store: Dict[str, np.ndarray] = {}
            for i, emb in zip(miss_indices, miss_embs):
                to_store[keys[i]] = np.asarray(emb)
                hits[keys[i]] = np.asarray(emb)
            self.cache.set_many(to_store, ttl_seconds=self.default_ttl_seconds)

        # reconstruct in original order
        arrs = [hits[k] for k in keys]
        return np.stack(arrs, axis=0)

    def encode_as_dict(self, texts: List[str]) -> Dict[str, np.ndarray]:
        # build on encode() to guarantee identical behavior
        embeddings = self.encode(texts)
        return {t: embeddings[i] for i, t in enumerate(texts)}
