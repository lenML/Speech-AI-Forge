import html
import os
import re

import emojiswitch
import ftfy
from pywrapfst import FstOpError

from modules.core.models import zoo
from modules.core.tn.TNPipeline import GuessLang, TNPipeline
from modules.repos_static.zh_normalization.text_normlization import TextNormalizer
from modules.utils.HomophonesReplacer import HomophonesReplacer
from modules.utils.html import remove_html_tags as _remove_html_tags
from modules.utils.markdown import markdown_to_text

DISABLE_UNK_TOKEN_CHECK = False

ChatTtsTN = TNPipeline()
ChatTtsTN.freeze_strs = [
    "[Sasr]",
    "[Pasr]",
    "[Easr]",
    "[Stts]",
    "[Ptts]",
    "[Etts]",
    "[Sbreak]",
    "[Pbreak]",
    "[Ebreak]",
    "[uv_break]",
    "[v_break]",
    "[lbreak]",
    "[llbreak]",
    "[undefine]",
    "[laugh]",
    "[spk_emb]",
    "[empty_spk]",
    "[music]",
    "[pure]",
    "[break_0]",
    "[break_1]",
    "[break_2]",
    "[break_3]",
    "[break_4]",
    "[break_5]",
    "[break_6]",
    "[break_7]",
    "[laugh_0]",
    "[laugh_1]",
    "[laugh_2]",
    "[oral_0]",
    "[oral_1]",
    "[oral_2]",
    "[oral_3]",
    "[oral_4]",
    "[oral_5]",
    "[oral_6]",
    "[oral_7]",
    "[oral_8]",
    "[oral_9]",
    "[speed_0]",
    "[speed_1]",
    "[speed_2]",
    "[speed_3]",
    "[speed_4]",
    "[speed_5]",
    "[speed_6]",
    "[speed_7]",
    "[speed_8]",
    "[speed_9]",
]

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


@ChatTtsTN.block()
def html_unescape(text: str, guess_lang: GuessLang):
    text = html.unescape(text)
    text = html.unescape(text)
    return text


@ChatTtsTN.block()
def fix_text(text: str, guess_lang: GuessLang):
    return ftfy.fix_text(text=text)


@ChatTtsTN.block()
def apply_markdown_to_text(text: str, guess_lang: GuessLang):
    if is_markdown(text):
        text = markdown_to_text(text)
    return text


@ChatTtsTN.block()
def remove_html_tags(text: str, guess_lang: GuessLang):
    return _remove_html_tags(text)


# 将 "xxx" => \nxxx\n
# 将 'xxx' => \nxxx\n
@ChatTtsTN.block()
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


@ChatTtsTN.block(name="tx_zh", enabled=True)
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


@ChatTtsTN.block(name="wetext_en", enabled=True)
def wetext_normalize(text: str, guss_lang: GuessLang):
    # NOTE: wetext 依赖 pynini 无法在 windows 上安装，所以这里只在 linux 上启用
    if os.name == "nt":
        return text
    if guss_lang.zh_or_en == "en":
        from tn.english.normalizer import Normalizer as EnNormalizer

        en_tn_model = EnNormalizer(overwrite_cache=False)
        try:
            return en_tn_model.normalize(text)
        except FstOpError:
            # NOTE: 不太理解为什么 tn 都能出错...
            pass
    return text


# ---- end main normalize ----


@ChatTtsTN.block()
def apply_character_map(text: str, guess_lang: GuessLang):
    translation_table = str.maketrans(character_map)
    return text.translate(translation_table)


@ChatTtsTN.block()
def apply_emoji_map(text: str, guess_lang: GuessLang):
    return emojiswitch.demojize(text, delimiters=("", ""), lang=guess_lang.zh_or_en)


@ChatTtsTN.block()
def insert_spaces_between_uppercase(text: str, guess_lang: GuessLang):
    # 使用正则表达式在每个相邻的大写字母之间插入空格
    return re.sub(
        r"(?<=[A-Z])(?=[A-Z])|(?<=[a-z])(?=[A-Z])|(?<=[\u4e00-\u9fa5])(?=[A-Z])|(?<=[A-Z])(?=[\u4e00-\u9fa5])",
        " ",
        text,
    )


@ChatTtsTN.block()
def replace_unk_tokens(text: str, guess_lang: GuessLang):
    """
    把不在字典里的字符替换为 " , "

    FIXME: 总感觉不太好...但是没有遇到问题的话暂时留着...
    """
    if DISABLE_UNK_TOKEN_CHECK:
        return text
    chat_tts = zoo.ChatTTS.load_chat_tts()
    if "tokenizer" not in chat_tts.pretrain_models:
        # 这个地方只有在 huggingface spaces 中才会触发
        # 因为 hugggingface 自动处理模型卸载加载，所以如果拿不到就算了...
        return text
    tokenizer = zoo.ChatTTS.get_tokenizer()
    vocab = tokenizer.get_vocab()
    vocab_set = set(vocab.keys())
    # 添加所有英语字符
    vocab_set.update(set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"))
    vocab_set.update(set(" \n\r\t"))
    replaced_chars = [char if char in vocab_set else " , " for char in text]
    output_text = "".join(replaced_chars)
    return output_text


homo_replacer = HomophonesReplacer(
    map_file_path="./modules/ChatTTS/ChatTTS/res/homophones_map.json"
)


@ChatTtsTN.block()
def replace_homophones(text: str, guess_lang: GuessLang):
    if guess_lang.zh_or_en == "zh":
        text = homo_replacer.replace(text)
    return text


if __name__ == "__main__":
    from modules.devices import devices

    DISABLE_UNK_TOKEN_CHECK = True

    devices.reset_device()
    test_cases = [
        "ChatTTS是专门为对话场景设计的文本转语音模型，例如LLM助手对话任务。它支持英文和中文两种语言。最大的模型使用了10万小时以上的中英文数据进行训练。在HuggingFace中开源的版本为4万小时训练且未SFT的版本.",
        " [oral_9] [laugh_0] [break_0] 电 [speed_0] 影 [speed_0] 中 梁朝伟 [speed_9] 扮演的陈永仁的编号27149",
        " 明天有62％的概率降雨",
        "大🍌，一条大🍌，嘿，你的感觉真的很奇妙  [lbreak]",
        "I like eating 🍏",
        """
# 你好，世界
```js
console.log('1')
```
**加粗**

*一条文本*
        """,
        """
在沙漠、岩石、雪地上行走了很长的时间以后，小王子终于发现了一条大路。所有的大路都是通往人住的地方的。
“你们好。”小王子说。
这是一个玫瑰盛开的花园。
“你好。”玫瑰花说道。
小王子瞅着这些花，它们全都和他的那朵花一样。
“你们是什么花？”小王子惊奇地问。
“我们是玫瑰花。”花儿们说道。
“啊！”小王子说……。
        """,
        """
State-of-the-art Machine Learning for PyTorch, TensorFlow, and JAX.

🤗 Transformers provides APIs and tools to easily download and train state-of-the-art pretrained models. Using pretrained models can reduce your compute costs, carbon footprint, and save you the time and resources required to train a model from scratch. These models support common tasks in different modalities, such as:

📝 Natural Language Processing: text classification, named entity recognition, question answering, language modeling, summarization, translation, multiple choice, and text generation.
🖼️ Computer Vision: image classification, object detection, and segmentation.
🗣️ Audio: automatic speech recognition and audio classification.
🐙 Multimodal: table question answering, optical character recognition, information extraction from scanned documents, video classification, and visual question answering.
        """,
        """
120米
有12%的概率会下雨
埃隆·马斯克
""",
    ]

    for i, test_case in enumerate(test_cases):
        print(f"case {i}:\n", {"x": ChatTtsTN.normalize(test_case)})
