from modules.core.models.tts.F5.F5Annotation import F5Annotation

annotation = F5Annotation()


def convert_char_to_pinyin(text_list: list[str], polyphone=True) -> list[list[str]]:
    return [annotation.convert_to_pinyin(text) for text in text_list]
