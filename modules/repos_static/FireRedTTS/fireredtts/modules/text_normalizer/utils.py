from fireredtts.modules.text_normalizer.regex_common import *


def contains_chinese(text):
    return bool(chinese_regex.search(text))


def strip_kaomoji(text):
    return kaomoji_regex.sub(" ", text)


def is_chinese(char):
    return chinese_char_regex.match(char)


def is_eng_and_digit(char):
    return eng_and_digit_char_regex.match(char)


def is_upper_eng_and_digit(text):
    return upper_eng_and_digit_regex.match(text)


def is_valid_char(char):
    return valid_char_regex.match(char)


def is_digit(text):
    return digit_regex.match(text)


def contains_chinese(text):
    return bool(chinese_regex.search(text))


def f2b(ustr, exemption="。，："):
    half = []
    for u in ustr:
        num = ord(u)
        if num == 0x3000:
            half.append(" ")
        elif u in exemption:  # exemption
            half.append(u)
        elif 0xFF01 <= num <= 0xFF5E:
            num -= 0xFEE0
            half.append(chr(num))
        else:
            half.append(u)
    return "".join(half)


symbol_reduction = {
    "「": '"',
    "」": '"',
    "｀": '"',
    "〝": '"',
    "〞": '"',
    "‟": '"',
    "„": '"',
    "｛": "(",
    "｝": ")",
    "【": "(",
    "】": ")",
    "〖": "(",
    "〗": ")",
    "〔": "(",
    "〕": ")",
    "〘": "(",
    "〙": ")",
    "《": "(",
    "》": ")",
    "｟": "(",
    "｠": ")",
    "〚": "(",
    "〛": ")",
    "『": '"',
    "』": '"',
    "｢": '"',
    "｣": '"',
    "{": "(",
    "}": ")",
    "〈": "(",
    "〉": ")",
    "•": "·",
    "‧": "·",
    "〰": "…",
    "﹏": "…",
    "〜": "~",
    "～": "~",
    "＋": "+",
    "､": "、",
    "｡": "。",
    "︐": "，",
    "﹐": "，",
    "︑": "、",
    "﹑": "、",
    "︒": "。",
    "︓": "：",
    "﹕": "：",
    "︔": "；",
    "﹔": "；",
    "︕": "！",
    "﹗": "！",
    "︖": "？",
    "﹖": "？",
    "﹙": "(",
    "﹚": ")",
    "﹪": "%",
    "﹠": "&",
    "＞": ">",
    "|": "、",
    "＝": "=",
    "‐": "-",
    "‑": "-",
    "‒": "-",
    "–": "-",
    "—": "-",
    "―": "-",
    "％": "%",
    "μ": "u",
}
