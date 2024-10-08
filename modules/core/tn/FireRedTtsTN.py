from modules.core.tn.TNPipeline import GuessLang

from .base_tn import BaseTN
from .whisper import specials_token as whisper_specials_token

FireRedTtsTN = BaseTN.clone()
FireRedTtsTN.freeze_tokens = [
    # TODO: 应该从 tiktoken 中取值
    "[char_rep]",
    "[word_rep]",
    "[elong]",
    "[oralsii]",
    "[tsk]",
    "[breath]",
    "(filled pause)",
    "(confirmation)",
    "(realization)",
    "(surprise)",
    *whisper_specials_token,
]


@FireRedTtsTN.block()
def something_tn(text: str, guess_lang: GuessLang) -> str:
    # NOTE: 暂时用不着 留个位置
    # NOTE: FireRedTTS 的 tn 是在 tokenizer 执行时调用
    return text


if __name__ == "__main__":
    pass
