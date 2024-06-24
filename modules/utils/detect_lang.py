from functools import lru_cache
from typing import Literal


@lru_cache(maxsize=64)
def is_chinese(text):
    for char in text:
        if "\u4e00" <= char <= "\u9fff":
            return True
    return False


@lru_cache(maxsize=64)
def is_eng(text):
    for char in text:
        if "a" <= char.lower() <= "z":
            return True
    return False


@lru_cache(maxsize=64)
def guess_lang(text) -> Literal["zh", "en"]:
    if is_chinese(text):
        return "zh"
    if is_eng(text):
        return "en"
    return "zh"
