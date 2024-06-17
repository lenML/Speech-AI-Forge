from functools import _CacheInfo, lru_cache
from typing import Any, Callable, TypeVar

from typing_extensions import ParamSpec


def conditional_cache(maxsize: int, condition: Callable):
    def decorator(func):
        @lru_cache_ext(maxsize=maxsize)
        def cached_func(*args, **kwargs):
            return func(*args, **kwargs)

        def wrapper(*args, **kwargs):
            if condition(*args, **kwargs):
                return cached_func(*args, **kwargs)
            else:
                return func(*args, **kwargs)

        return wrapper

    return decorator


def hash_list(l: list) -> int:
    __hash = 0
    for i, e in enumerate(l):
        __hash = hash((__hash, i, hash_item(e)))
    return __hash


def hash_dict(d: dict) -> int:
    __hash = 0
    for k, v in d.items():
        __hash = hash((__hash, k, hash_item(v)))
    return __hash


def hash_item(e) -> int:
    if hasattr(e, "__hash__") and callable(e.__hash__):
        try:
            return hash(e)
        except TypeError:
            pass
    if isinstance(e, (list, set, tuple)):
        return hash_list(list(e))
    elif isinstance(e, (dict)):
        return hash_dict(e)
    else:
        raise TypeError(f"unhashable type: {e.__class__}")


PT = ParamSpec("PT")
RT = TypeVar("RT")


def lru_cache_ext(
    *opts, hashfunc: Callable[..., int] = hash_item, **kwopts
) -> Callable[[Callable[PT, RT]], Callable[PT, RT]]:
    def decorator(func: Callable[PT, RT]) -> Callable[PT, RT]:
        class _lru_cache_ext_wrapper:
            args: tuple
            kwargs: dict[str, Any]

            def cache_info(self) -> _CacheInfo: ...
            def cache_clear(self) -> None: ...

            @classmethod
            @lru_cache(*opts, **kwopts)
            def cached_func(cls, args_hash: int) -> RT:
                return func(*cls.args, **cls.kwargs)

            @classmethod
            def __call__(cls, *args: PT.args, **kwargs: PT.kwargs) -> RT:
                __hash = hashfunc(
                    (
                        id(func),
                        *[hashfunc(a) for a in args],
                        *[(hashfunc(k), hashfunc(v)) for k, v in kwargs.items()],
                    )
                )

                cls.args = args
                cls.kwargs = kwargs

                cls.cache_info = cls.cached_func.cache_info
                cls.cache_clear = cls.cached_func.cache_clear

                return cls.cached_func(__hash)

        return _lru_cache_ext_wrapper()

    return decorator
