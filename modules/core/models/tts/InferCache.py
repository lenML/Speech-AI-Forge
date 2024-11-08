from typing import Dict

import torch
from cachetools import LRUCache
from cachetools import keys as cache_keys

from modules.core.spk.TTSSpeaker import TTSSpeaker


def hash_tensor(tensor: torch.Tensor):
    """
    NOTE: 在不同执行虚拟机中此函数可能不稳定
    NOTE: 但是用来计算 cache 足够了
    """

    return hash(tuple(tensor.reshape(-1).tolist()))


class InferCache:
    caches: Dict[str, LRUCache] = {}

    @classmethod
    def get_cache(cls, model_id: str) -> LRUCache:
        if model_id in InferCache.caches:
            return InferCache.caches.get(model_id)
        cache = LRUCache(maxsize=128)
        InferCache.caches[model_id] = cache
        return cache

    @classmethod
    def get_hash_key(cls, *args, **kwargs):
        for i, arg in enumerate(args):
            if isinstance(arg, TTSSpeaker):
                args[i] = str(arg.id)
            if isinstance(arg, torch.Tensor):
                args[i] = hash_tensor(arg)

        for key, value in kwargs.items():
            if isinstance(value, TTSSpeaker):
                kwargs[key] = str(value.id)
            if isinstance(value, torch.Tensor):
                kwargs[key] = hash_tensor(value)

        cachekey = cache_keys.hashkey(*args, **kwargs)
        return cachekey

    @classmethod
    def get_cache_val(cls, model_id: str, *args, **kwargs):
        key = cls.get_hash_key(*args, **kwargs)
        cache = InferCache.get_cache(model_id)

        if key in cache:
            return cache[key]

        return None

    @classmethod
    def set_cache_val(cls, model_id: str, value, *args, **kwargs):
        key = cls.get_hash_key(*args, **kwargs)
        cache = InferCache.get_cache(model_id)

        cache[key] = value

    @classmethod
    def cached(cls, model_id: str, should_cache: callable = None):
        """
        包装器

        should_cache 用于提供 条件化 cache
        """

        def decorator(func):
            def wrapper(*args, **kwargs):
                if should_cache is not None and not should_cache(*args, **kwargs):
                    return func(*args, **kwargs)

                cached = cls.get_cache_val(model_id, *args, **kwargs)
                if cached is not None:
                    return cached

                result = func(*args, **kwargs)
                cls.set_cache_val(model_id, result, *args, **kwargs)
                return result

            return wrapper

        return decorator
