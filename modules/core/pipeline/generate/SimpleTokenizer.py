import re


class TokenizerLike:

    def encode(self, text: str) -> list[int]:
        pass

    def decode(self, ids: list[int]) -> str:
        pass


class SimpleTokenizer(TokenizerLike):

    def encode(self, text: str) -> list[int]:
        return [ord(char) for char in text]

    def decode(self, ids: list[int]) -> str:
        return "".join([chr(id) for id in ids])


class RegexpTokenizer(SimpleTokenizer):

    def encode(self, text: str) -> list[int]:
        # NOTE: 略微比纯基于char的好一点，因为只是给 spliter 用，所以大概能计算出一个结果即可
        tokens = re.findall(r"\w{1,4}|[^\w\s]{1,2}", text, re.UNICODE)
        return [ord(char[0]) for char in tokens]
