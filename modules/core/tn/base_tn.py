import html
import os
import re

import emojiswitch
import ftfy

from modules.core.tn.TNPipeline import GuessLang, TNPipeline
from modules.repos_static.zh_normalization.text_normlization import TextNormalizer
from modules.utils.HomophonesReplacer import HomophonesReplacer
from modules.utils.html import remove_html_tags as _remove_html_tags
from modules.utils.markdown import markdown_to_text

BaseTN = TNPipeline()

# ------- UTILS ---------


def is_markdown(text):
    markdown_patterns = [
        r"(^|\s)#[^#]",  # 标题
        r"\*\*.*?\*\*",  # 加粗
        r"\*.*?\*",  # 斜体
        r"!\[.*?\]\(.*?\)",  # 图片
        r"\[.*?\]\(.*?\)",  # 链接
        r"`[^`]+`",  # 行内代码
        r"```[\s\S]*?```",  # 代码块
        r"(^|\s)\* ",  # 无序列表
        r"(^|\s)\d+\. ",  # 有序列表
        r"(^|\s)> ",  # 引用
        r"(^|\s)---",  # 分隔线
    ]

    for pattern in markdown_patterns:
        if re.search(pattern, text, re.MULTILINE):
            return True

    return False


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
    "‘": " ",
    "“": " ",
    "’": " ",
    "”": " ",
    '"': " ",
    "'": " ",
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
    "~": " ",
    "～": " ",
    "/": " ",
    "·": " ",
}

# -----------------------


@BaseTN.block()
def html_unescape(text: str, guess_lang: GuessLang):
    text = html.unescape(text)
    text = html.unescape(text)
    return text


@BaseTN.block()
def fix_text(text: str, guess_lang: GuessLang):
    return ftfy.fix_text(text=text)


@BaseTN.block()
def apply_markdown_to_text(text: str, guess_lang: GuessLang):
    if is_markdown(text):
        text = markdown_to_text(text)
    return text


@BaseTN.block()
def remove_html_tags(text: str, guess_lang: GuessLang):
    return _remove_html_tags(text)


# 将 "xxx" => \nxxx\n
# 将 'xxx' => \nxxx\n
@BaseTN.block()
def replace_quotes(text: str, guess_lang: GuessLang):
    repl = r"\n\1\n"
    patterns = [
        ['"', '"'],
        ["'", "'"],
        ["“", "”"],
        ["‘", "’"],
    ]
    for p in patterns:
        text = re.sub(rf"({p[0]}[^{p[0]}{p[1]}]+?{p[1]})", repl, text)
    return text


# ---- main normalize ----


@BaseTN.block(name="tx_zh", enabled=True)
def tx_normalize(text: str, guss_lang: GuessLang):
    if guss_lang.zh_or_en != "zh":
        return text
    # NOTE: 这个是魔改过的 TextNormalizer 来自 PaddlePaddle
    tx = TextNormalizer()
    # NOTE: 为什么要分行？因为我们需要保留 "\n" 作为 chunker 的分割信号
    lines = [line for line in text.split("\n") if line.strip() != ""]
    texts: list[str] = []
    for line in lines:
        ts = tx.normalize(line)
        texts.append("".join(ts))
    return "\n".join(texts)


@BaseTN.block(name="wetext_en", enabled=True)
def wetext_normalize(text: str, guss_lang: GuessLang):
    # NOTE: wetext 依赖 pynini 无法在 windows 上安装，所以这里只在 linux 上启用
    if os.name == "nt":
        return text
    if guss_lang.zh_or_en == "en":
        from pywrapfst import FstOpError
        from tn.english.normalizer import Normalizer as EnNormalizer

        en_tn_model = EnNormalizer(overwrite_cache=False)
        try:
            return en_tn_model.normalize(text)
        except FstOpError:
            # NOTE: 不太理解为什么 tn 都能出错...
            pass
    return text


# ---- end main normalize ----


@BaseTN.block()
def apply_character_map(text: str, guess_lang: GuessLang):
    translation_table = str.maketrans(character_map)
    return text.translate(translation_table)


@BaseTN.block()
def apply_emoji_map(text: str, guess_lang: GuessLang):
    return emojiswitch.demojize(text, delimiters=("", ""), lang=guess_lang.zh_or_en)


@BaseTN.block()
def insert_spaces_between_uppercase(text: str, guess_lang: GuessLang):
    # 使用正则表达式在每个相邻的大写字母之间插入空格
    return re.sub(
        r"(?<=[A-Z])(?=[A-Z])|(?<=[a-z])(?=[A-Z])|(?<=[\u4e00-\u9fa5])(?=[A-Z])|(?<=[A-Z])(?=[\u4e00-\u9fa5])",
        " ",
        text,
    )


homo_replacer = HomophonesReplacer(
    map_file_path="./modules/repos_static/ChatTTS/ChatTTS/res/homophones_map.json"
)


@BaseTN.block()
def replace_homophones(text: str, guess_lang: GuessLang):
    if guess_lang.zh_or_en == "zh":
        text = homo_replacer.replace(text)
    return text
