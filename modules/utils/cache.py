from functools import lru_cache
from typing import Callable


def conditional_cache(condition: Callable):
    def decorator(func):
        @lru_cache(None)
        def cached_func(*args, **kwargs):
            return func(*args, **kwargs)

        def wrapper(*args, **kwargs):
            if condition(*args, **kwargs):
                return cached_func(*args, **kwargs)
            else:
                return func(*args, **kwargs)

        return wrapper

    return decorator
