import re
import regex
import inflect
import unicodedata
from lingua import Language, LanguageDetectorBuilder
from builtins import str as unicode

from tn.chinese.normalizer import Normalizer as ZhNormalizer
from tn.english.normalizer import Normalizer as EnNormalizer

from fireredtts.modules.text_normalizer.regex_common import *
from fireredtts.modules.text_normalizer.utils import *


def preprocess_text(sentence):
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


def rettt(sentence):
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


class TextNormalizer:
    def __init__(self):
        self.language_detector = LanguageDetectorBuilder.from_languages(
            Language.ENGLISH, Language.CHINESE
        ).build()
        self.zh_normalizer = ZhNormalizer()
        self.en_normalizer = EnNormalizer()
        self.inflect_parser = inflect.engine()
        self.lang2token = {Language.ENGLISH: "en", Language.CHINESE: "zh"}

    def tn(self, text):
        text = preprocess_text(text)
        text = rettt(text)  # regex replacements
        # for non chinese languages
        language = self.language_detector.detect_language_of(text)
        # enforce chinese if text contains any chinese character
        if contains_chinese(text):
            language = Language.CHINESE
        text_lang = self.lang2token.get(language, "zh")

        if is_upper_eng_and_digit(text):
            language = Language.CHINESE

        if language == Language.CHINESE:
            text = self.zh_normalizer.normalize(text)
            text = text.replace("\n", "")
            text = re.sub(r"[，,]+$", "。", text)
        else:
            text = re.sub(r"[^ 0-9A-Za-z\[\]'.,:?!_\-]", "", text)
            text = self.en_normalizer.normalize(text)
            # fallback number normalization
            pieces = re.split(r"(\d+)", text)
            text = "".join(
                [
                    self.inflect_parser.number_to_words(p) if p.isnumeric() else p
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

        return text, text_lang
