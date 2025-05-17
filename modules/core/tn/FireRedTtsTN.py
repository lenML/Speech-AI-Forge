import re
import unicodedata
from builtins import str as unicode

import inflect
import regex

from modules.core.tn.TNPipeline import GuessLang, TNPipeline
from modules.repos_static.FireRedTTS.fireredtts.modules.text_normalizer.utils import (
    f2b,
    is_chinese,
    is_valid_char,
    strip_kaomoji,
    symbol_reduction,
)

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


def preprocess_text(sentence: str):
    # preprocessing
    sentence = bytes(sentence, "utf-8").decode("utf-8", "ignore")
    sentence = regex.sub("[\p{Cf}--[\u200d]]", "", sentence, flags=regex.V1)
    sentence = regex.sub("\p{Co}", "", sentence)
    sentence = sentence.replace("\u00a0", " ")
    sentence = sentence.replace("\ufffd", "")
    sentence = regex.sub("\p{Zl}", "\n", sentence)
    sentence = regex.sub("\p{Zp}", "\n", sentence)

    sentence = unicode(sentence)
    sentence = "".join(
        char
        for char in unicodedata.normalize("NFD", sentence)
        if unicodedata.category(char) != "Mn"
    )  # Strip accents

    sentence = strip_kaomoji(sentence)
    # full to half with exemption (to be converted after number TN): 。，：
    sentence = f2b(sentence, exemption="。，：")

    # clean spaces
    sentence = sentence.replace("\n", "，")
    sentence = sentence.replace("\t", "，")
    sentence = sentence.replace("\r", "，")
    sentence = re.sub(r"[。.]{3,}", "…", sentence)
    sentence = re.sub(r"[…⋯]{1,}", "…", sentence)
    sentence = re.sub(r"[ ]+", " ", sentence)
    sentence = sentence.strip()

    # punctuation reduction
    result = ""
    for idx, char in enumerate(sentence):
        if char in symbol_reduction:
            char = symbol_reduction[char]

        if char == " ":
            if idx == 0:
                continue
            if is_chinese(sentence[idx + 1]) and (
                is_chinese(sentence[idx - 1]) or sentence[idx - 1] in '") '
            ):
                result += "，"
            else:
                result += " "
            continue

        if is_valid_char(char):
            result += char
    result = re.sub(r"[ ]+", " ", result)
    return result


def rettt(sentence: str):
    # handle abbreviations for all languages
    sentence = sentence.replace("&nd", "and")
    sentence = sentence.replace("Jan.", "january")
    sentence = sentence.replace("Feb.", "febrary")
    sentence = sentence.replace("Mar.", "march")
    sentence = sentence.replace("Apr.", "april")
    sentence = sentence.replace("May.", "may")
    sentence = sentence.replace("Jun.", "june")
    sentence = sentence.replace("Jul.", "july")
    sentence = sentence.replace("Aug.", "august")
    sentence = sentence.replace("Sept.", "september")
    sentence = sentence.replace("Sep.", "september")
    sentence = sentence.replace("Oct.", "october")
    sentence = sentence.replace("Nov.", "november")
    sentence = sentence.replace("Dec.", "december")
    sentence = sentence.replace("Mon.", "monday")
    sentence = sentence.replace("Tues.", "tuesday")
    sentence = sentence.replace("Wed.", "wednesday")
    sentence = sentence.replace("Thur.", "thursday")
    sentence = sentence.replace("Fri.", "friday")
    sentence = sentence.replace("Sat.", "saturday")
    if sentence != "Sun.":
        sentence = sentence.replace("Sun.", "sunday")
    sentence = re.sub(r" St\. ([A-Z])", r" saint \1", sentence)
    sentence = re.sub(r" St\.", " street", sentence)
    sentence = re.sub(r" Rd\.", " road", sentence)
    sentence = re.sub(r"[Aa]\.[Mm]\.", "A_M", sentence)
    sentence = re.sub(r"[Pp]\.[Mm]\.", "P_M", sentence)
    sentence = re.sub(r"[Bb]\.[Cc]\.", "B_C", sentence)
    sentence = re.sub(r"[Ad]\.[Dd]\.", "A_D", sentence)
    sentence = sentence.replace("Mr.", "mister")
    sentence = sentence.replace("Ms.", "miss")
    sentence = sentence.replace("Mrs.", "misses")
    sentence = sentence.replace("Ph.D", "P_H_D")
    sentence = sentence.replace("i.e.", "that is")
    sentence = sentence.replace("e.g.", "for example")
    sentence = sentence.replace("btw.", "by the way")
    sentence = sentence.replace("btw", "by the way")
    sentence = sentence.replace("b.t.w.", "by the way")
    sentence = sentence.replace("@", " at ")
    return sentence


@FireRedTtsTN.block()
def something_tn(text: str, guess_lang: GuessLang) -> str:
    text = preprocess_text(text)
    text = rettt(text)
    return text

inflect_parser = inflect.engine()


@FireRedTtsTN.block()
def clean_text(text: str, guess_lang: GuessLang) -> str:
    if guess_lang.zh_or_en == "zh":
        text = text.replace("\n", "")
        text = re.sub(r"[，,]+$", "。", text)
    else:
        text = re.sub(r"[^ 0-9A-Za-z\[\]'.,:?!_\-]", "", text)
        # fallback number normalization
        pieces = re.split(r"(\d+)", text)
        text = "".join(
            [
                inflect_parser.number_to_words(p) if p.isnumeric() else p
                for p in pieces
                if len(p) > 0
            ]
        )

    # cleanup
    text = text.replace("_", " ")
    text = re.sub(r"[ ]+", " ", text)

    # spell caplital words
    pieces = re.split(r"([A-Z]{2,4}|[ ])", text)
    for idx, p in enumerate(pieces):
        if re.match("[A-Z]{2,4}", p):
            pieces[idx] = " ".join(p)
    text = " ".join([p for p in pieces if p != " "])

    # post TN full to half
    text = text.replace("。", ".")
    text = text.replace("，", ",")
    text = text.replace("：", ":")

    # model limitations
    text = text.lower().strip()
    text = text.replace('"', "")
    text = text.replace("·", " ")
    text = re.sub("[…~、！，？：；!?:;]+", ",", text)
    text = re.sub("[,]+", ",", text)
    text = re.sub(r"[,. ]+$", ".", text)
    if len(text) > 0 and text[-1] != ".":
        text = text + "."
    return text


if __name__ == "__main__":
    pass
