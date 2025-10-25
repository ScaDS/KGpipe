# # pip install redis
# import redis
# class RedisCache(CacheBackend):
#     def __init__(self, url="redis://localhost:6379/0", ttl_seconds: Optional[int]=None):
#         self.ttl = ttl_seconds
#         self.r = redis.Redis.from_url(url)

#     def get_many(self, keys: List[str]) -> Dict[str, np.ndarray]:
#         vals = self.r.mget(keys)
#         out = {}
#         for k, v in zip(keys, vals):
#             if v is not None:
#                 out[k] = _blob_to_np(v)
#         return out

#     def set_many(self, items: Dict[str, np.ndarray], ttl_seconds: Optional[int]=None) -> None:
#         pipe = self.r.pipeline()
#         ttl = ttl_seconds if ttl_seconds is not None else self.ttl
#         for k, arr in items.items():
#             blob = _np_to_blob(arr)
#             if ttl:
#                 pipe.setex(k, ttl, blob)
#             else:
#                 pipe.set(k, blob)
#         pipe.execute()

#     def close(self) -> None:
#         self.r.close()
