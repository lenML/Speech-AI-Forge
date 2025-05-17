from modules.core.tn.TNPipeline import GuessLang, TNPipeline
from modules.repos_static.index_tts.indextts.utils.front import TextNormalizer
from modules.utils.HomophonesReplacer import HomophonesReplacer
from modules.utils.html import remove_html_tags as _remove_html_tags
from modules.utils.markdown import markdown_to_text

from .base_tn import BaseTN

IndexTTSTN = BaseTN.clone()
IndexTTSTN.freeze_tokens = []

IndexTTSTN.remove_block("apply_character_map")


# TODO: 尝试使用这个 tn，因为这个基本都是基于 wetext 的，安装有点问题，我感觉用我们自己的也够了，并且之后还得支持 pinyin 语法
# tts_front_tn = TextNormalizer()
# tts_front_tn.load()
# @IndexTTSTN.block()
# def fs_clean_text(text: str, guess_lang: GuessLang) -> str:
#     return tts_front_tn.normalize(text)
