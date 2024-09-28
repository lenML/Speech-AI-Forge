import re

from modules.core.tn.TNPipeline import GuessLang
from modules.repos_static.cosyvoice.cosyvoice.utils.frontend_utils import (
    remove_bracket,
    replace_blank,
    replace_corner_mark,
)

from .base_tn import BaseTN

CosyVoiceTN = BaseTN.clone()
CosyVoiceTN.freeze_tokens = [
    "[laughter]",
    "[breath]",
    "<laughter>",
    "</laughter>",
    "<storng>",
    "</storng>",
    # <|zh|><|en|><|jp|><|yue|><|ko|> for Chinese/English/Japanese/Cantonese/Korean
    "<|zh|>",
    "<|en|>",
    "<|jp|>",
    "<|yue|>",
    "<|ko|>",
]


@CosyVoiceTN.block()
def cv_tn(text: str, guess_lang: GuessLang) -> str:
    text = text.strip()
    if guess_lang.zh_or_en == "en":
        return text
    # NOTE: 这个在这里大概率不会触发，因为 tn 之前会 chunker split
    text = text.replace("\n", "")  # 源代码这里把 \n 全部去掉了??? 这样不会有问题吗？
    text = replace_blank(text)
    text = replace_corner_mark(text)
    text = text.replace(".", "、")
    text = text.replace(" - ", "，")
    text = remove_bracket(text)
    text = re.sub(r"[，,]+$", "。", text)
    return text


if __name__ == "__main__":
    pass
