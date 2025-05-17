from modules.core.tn.TNPipeline import GuessLang

from .base_tn import BaseTN
from .pinyin_ton3_list import pinyin_ton3

f5_pinyin_annos = [f"({p})" for p in pinyin_ton3]

F5TtsTN = BaseTN.clone()
F5TtsTN.freeze_tokens = [
    # 所有拼音标注
    *f5_pinyin_annos
]
F5TtsTN.SEP_CHAR = ""


@F5TtsTN.block()
def something_tn(text: str, guess_lang: GuessLang) -> str:
    # NOTE: 预留位置
    return text


if __name__ == "__main__":
    pass
