import copy
from dataclasses import dataclass
from typing import Callable, Dict, Literal, Optional

from langdetect import LangDetectException, detect_langs

from modules.core.handler.datacls.tn_model import TNConfig
from modules.utils.detect_lang import guess_lang


@dataclass(frozen=True, repr=False)
class GuessLang:
    zh_or_en: Literal["zh", "en"]
    detected: Dict[str, float]


class TNBlock:

    def __init__(self, name: str):
        self.name = name
        self.enabled = True

    def process(self, text: str, guess_lang: GuessLang):
        raise NotImplementedError


TNBlockFnType = Callable[[str, GuessLang], str]


class TNBlockFn(TNBlock):
    def __init__(self, name: str, fn: TNBlockFnType):
        super().__init__(name)
        self.fn = fn

    def process(self, text: str, guess_lang: GuessLang):
        return self.fn(text, guess_lang)


@dataclass(frozen=True, repr=False)
class TNText:
    text: str
    type: Literal["normal", "freeze"]


class TNPipeline:
    """文本归一化管道类"""

    SEP_CHAR = "\n"

    def __init__(self):
        self.blocks: list[TNBlock] = []
        self.freeze_tokens: list[str] = []

    def block(self, name: str = None, enabled: bool = True):
        block = TNBlockFn(name=name, fn=None)
        block.enabled = enabled
        self.blocks.append(block)

        def decorator(fn: TNBlockFnType):
            block.fn = fn
            if not block.name:
                block.name = fn.__name__
            return fn

        return decorator

    def append_block(self, fn: TNBlockFnType, enabled: bool = True):
        name = fn.__name__
        block = TNBlockFn(name=name, fn=fn)
        block.enabled = enabled
        self.blocks.append(block)

    def remove_block(self, name: str):
        self.blocks = [b for b in self.blocks if b.name != name]

    def clone(self):
        return copy.deepcopy(self)

    def split_string_with_freeze(
        self, text: str, freeze_strs: list[str]
    ) -> list[TNText]:
        if not freeze_strs:
            return [TNText(text=text, type="normal")]

        result: list[TNText] = []
        buffer = ""

        for char in text:
            buffer += char

            for freeze_str in freeze_strs:
                if buffer.endswith(freeze_str):
                    result.append(
                        TNText(text=buffer[: -len(freeze_str)], type="normal")
                    )
                    result.append(TNText(text=freeze_str, type="freeze"))
                    buffer = ""
                    break

        if buffer:
            result.append(TNText(text=buffer, type="normal"))

        return result

    def normalize(self, text: str, config: Optional[TNConfig] = None) -> str:
        texts: list[TNText] = self.split_string_with_freeze(text, self.freeze_tokens)

        result = ""

        for tn_text in texts:
            if tn_text.type == "normal":
                result += self._normalize(tn_text.text, config)
            else:
                result += tn_text.text
            result += self.SEP_CHAR

        return result.strip()

    def guess_langs(self, text: str):
        zh_or_en = guess_lang(text)
        try:
            detected_langs = detect_langs(text)
            detected = {lang.lang: lang.prob for lang in detected_langs}
        except LangDetectException:
            detected = {
                "zh": 1.0 if zh_or_en == "zh" else 0.0,
                "en": 1.0 if zh_or_en == "en" else 0.0,
            }
        guess = GuessLang(zh_or_en=zh_or_en, detected=detected)
        return guess

    def _normalize(self, text: str, config: Optional[TNConfig] = TNConfig()):
        if config is None:
            config = TNConfig()
        enabled_block = config.enabled if config.enabled else []
        disabled_block = config.disabled if config.disabled else []

        guess = self.guess_langs(text)

        for block in self.blocks:
            enabled = block.enabled

            if block.name in enabled_block:
                enabled = True
            if block.name in disabled_block:
                enabled = False

            if not enabled:
                continue
            # print(text)
            # print("---", block.name)
            text = block.process(text=text, guess_lang=guess)
            # print("---")
            # print(text)

        return text
