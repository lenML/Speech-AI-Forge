from modules.core.tn.TNPipeline import GuessLang, TNPipeline

from .base_tn import BaseTN
from .whisper import specials_token as whisper_specials_token

FireRedTtsTN = BaseTN.clone()
# FireRedTtsTN = TNPipeline()
FireRedTtsTN.freeze_tokens = [
    # TODO: 应该从 tiktoken 中取值
    # NOTE: 貌似还用不了，具体情况追踪这个 issues： https://github.com/FireRedTeam/FireRedTTS/issues/12
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
    # 特殊语气符号 (现在好像没什么用...)
    # "@",
    # "^",
    *whisper_specials_token,
]
# NOTE: 不能用换行 因为 fire red 内部的 tn 会把换行替换成逗号
FireRedTtsTN.SEP_CHAR = " "


@FireRedTtsTN.block()
def something_tn(text: str, guess_lang: GuessLang) -> str:
    # NOTE: 暂时用不着 留个位置
    # NOTE: FireRedTTS 的 tn 是在 tokenizer 执行时调用
    # TODO: 最好是把内部的 TN 移出来，不然不好控制
    return text


if __name__ == "__main__":
    pass
