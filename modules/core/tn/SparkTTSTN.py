from modules.core.tn.TNPipeline import GuessLang, TNPipeline
from modules.repos_static.index_tts.indextts.utils.front import TextNormalizer
from modules.utils.HomophonesReplacer import HomophonesReplacer
from modules.utils.html import remove_html_tags as _remove_html_tags
from modules.utils.markdown import markdown_to_text

from .base_tn import BaseTN

SparkTTSTN = BaseTN.clone()
SparkTTSTN.freeze_tokens = []

SparkTTSTN.remove_block("apply_character_map")


# TODO: 可能需要自行TN
