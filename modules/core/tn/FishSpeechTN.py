from modules.core.tn.TNPipeline import GuessLang
from modules.repos_static.fish_speech.fish_speech.text.clean import clean_text

from .base_tn import BaseTN

FishSpeechTN = BaseTN.clone()
FishSpeechTN.freeze_tokens = []

FishSpeechTN.remove_block("apply_character_map")


@FishSpeechTN.block()
def fs_clean_text(text: str, guess_lang: GuessLang) -> str:
    return clean_text(text)
