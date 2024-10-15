from modules.core.tn.TNPipeline import GuessLang
from .base_tn import BaseTN


F5TtsTN = BaseTN.clone()
F5TtsTN.freeze_tokens = [
    # TODO: 好像没有
]
F5TtsTN.SEP_CHAR = ","


@F5TtsTN.block()
def something_tn(text: str, guess_lang: GuessLang) -> str:
    # NOTE: 预留位置
    return text


if __name__ == "__main__":
    pass
