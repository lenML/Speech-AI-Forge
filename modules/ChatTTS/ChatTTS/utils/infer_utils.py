import re

import torch
import torch.nn.functional as F


class CustomRepetitionPenaltyLogitsProcessorRepeat:

    def __init__(self, penalty: float, max_input_ids, past_window):
        if not isinstance(penalty, float) or not (penalty > 0):
            raise ValueError(
                f"`penalty` has to be a strictly positive float, but is {penalty}"
            )

        self.penalty = penalty
        self.max_input_ids = max_input_ids
        self.past_window = past_window

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:

        input_ids = input_ids[:, -self.past_window :]
        freq = F.one_hot(input_ids, scores.size(1)).sum(1)
        freq[self.max_input_ids :] = 0
        alpha = self.penalty**freq
        scores = scores.contiguous()
        scores = torch.where(scores < 0, scores * alpha, scores / alpha)

        return scores


class CustomRepetitionPenaltyLogitsProcessor:

    def __init__(self, penalty: float, max_input_ids, past_window):
        if not isinstance(penalty, float) or not (penalty > 0):
            raise ValueError(
                f"`penalty` has to be a strictly positive float, but is {penalty}"
            )

        self.penalty = penalty
        self.max_input_ids = max_input_ids
        self.past_window = past_window

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:

        input_ids = input_ids[:, -self.past_window :]
        score = torch.gather(scores, 1, input_ids)
        _score = score.detach().clone()
        score = torch.where(score < 0, score * self.penalty, score / self.penalty)
        score[input_ids >= self.max_input_ids] = _score[input_ids >= self.max_input_ids]
        scores.scatter_(1, input_ids, score)

        return scores


def count_invalid_characters(s, reserved_tokens: list = []):
    escaped_tokens = [re.escape(token) for token in reserved_tokens]
    reserved_pattern = "|".join(escaped_tokens)
    s = re.sub(rf"{reserved_pattern}", "", s)
    pattern = re.compile(r"[^\u4e00-\u9fffA-Za-z，。、,\. ]")
    non_alphabetic_chinese_chars = pattern.findall(s)
    return set(non_alphabetic_chinese_chars)


def detect_language(sentence):

    chinese_char_pattern = re.compile(r"[\u4e00-\u9fff]")
    english_word_pattern = re.compile(r"\b[A-Za-z]+\b")

    chinese_chars = chinese_char_pattern.findall(sentence)
    english_words = english_word_pattern.findall(sentence)

    if len(chinese_chars) > len(english_words):
        return "zh"
    else:
        return "en"


character_map = {
    "：": "，",
    "；": "，",
    "！": "。",
    "（": "，",
    "）": "，",
    "【": "，",
    "】": "，",
    "『": "，",
    "』": "，",
    "「": "，",
    "」": "，",
    "《": "，",
    "》": "，",
    "－": "，",
    "‘": "",
    "“": "",
    "’": "",
    "”": "",
    ":": ",",
    ";": ",",
    "!": ".",
    "(": ",",
    ")": ",",
    "[": ",",
    "]": ",",
    ">": ",",
    "<": ",",
    "-": ",",
}

halfwidth_2_fullwidth_map = {
    "!": "！",
    '"': "“",
    "'": "‘",
    "#": "＃",
    "$": "＄",
    "%": "％",
    "&": "＆",
    "(": "（",
    ")": "）",
    ",": "，",
    "-": "－",
    "*": "＊",
    "+": "＋",
    ".": "。",
    "/": "／",
    ":": "：",
    ";": "；",
    "<": "＜",
    "=": "＝",
    ">": "＞",
    "?": "？",
    "@": "＠",
    "[": "［",
    "\\": "＼",
    "]": "］",
    "^": "＾",
    "_": "＿",
    "`": "｀",
    "{": "｛",
    "|": "｜",
    "}": "｝",
    "~": "～",
}


def replace_unsupported_chars(text, replace_dict, reserved_tokens: list = []):
    escaped_tokens = [re.escape(token) for token in reserved_tokens]
    special_tokens_pattern = "|".join(escaped_tokens)
    tokens = re.split(f"({special_tokens_pattern})", text)

    def replace_chars(segment):
        for old_char, new_char in replace_dict.items():
            segment = segment.replace(old_char, new_char)
        return segment

    result = "".join(
        (
            replace_chars(segment)
            if not re.match(special_tokens_pattern, segment)
            else segment
        )
        for segment in tokens
    )

    return result


def apply_half2full_map(text, reserved_tokens: list = []):
    return replace_unsupported_chars(
        text, halfwidth_2_fullwidth_map, reserved_tokens=reserved_tokens
    )


def apply_character_map(text, reserved_tokens: list = []):
    return replace_unsupported_chars(
        text, character_map, reserved_tokens=reserved_tokens
    )
