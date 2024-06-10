from functools import lru_cache
from modules.utils.zh_normalization.text_normlization import *
import emojiswitch
from modules.utils.markdown import markdown_to_text
from modules import models
import re

# 是否关闭 unk token 检查
# NOTE: 单测的时候用于跳过模型加载
DISABLE_UNK_TOKEN_CHECK = False


@lru_cache(maxsize=64)
def is_chinese(text):
    # 中文字符的 Unicode 范围是 \u4e00-\u9fff
    chinese_pattern = re.compile(r"[\u4e00-\u9fff]")
    return bool(chinese_pattern.search(text))


@lru_cache(maxsize=64)
def is_eng(text):
    eng_pattern = re.compile(r"[a-zA-Z]")
    return bool(eng_pattern.search(text))


@lru_cache(maxsize=64)
def guess_lang(text):
    if is_chinese(text):
        return "zh"
    if is_eng(text):
        return "en"
    return "zh"


post_normalize_pipeline = []
pre_normalize_pipeline = []


def post_normalize():
    def decorator(func):
        post_normalize_pipeline.append(func)
        return func

    return decorator


def pre_normalize():
    def decorator(func):
        pre_normalize_pipeline.append(func)
        return func

    return decorator


def apply_pre_normalize(text):
    for func in pre_normalize_pipeline:
        text = func(text)
    return text


def apply_post_normalize(text):
    for func in post_normalize_pipeline:
        text = func(text)
    return text


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
}

character_to_word = {
    " & ": " and ",
}

## ---------- post normalize ----------


@post_normalize()
def apply_character_to_word(text):
    for k, v in character_to_word.items():
        text = text.replace(k, v)
    return text


@post_normalize()
def apply_character_map(text):
    translation_table = str.maketrans(character_map)
    return text.translate(translation_table)


@post_normalize()
def apply_emoji_map(text):
    lang = guess_lang(text)
    return emojiswitch.demojize(text, delimiters=("", ""), lang=lang)


@post_normalize()
def insert_spaces_between_uppercase(s):
    # 使用正则表达式在每个相邻的大写字母之间插入空格
    return re.sub(
        r"(?<=[A-Z])(?=[A-Z])|(?<=[a-z])(?=[A-Z])|(?<=[\u4e00-\u9fa5])(?=[A-Z])|(?<=[A-Z])(?=[\u4e00-\u9fa5])",
        " ",
        s,
    )


@post_normalize()
def replace_unk_tokens(text):
    """
    把不在字典里的字符替换为 " , "
    """
    if DISABLE_UNK_TOKEN_CHECK:
        return text
    chat_tts = models.load_chat_tts()
    if "tokenizer" not in chat_tts.pretrain_models:
        # 这个地方只有在 huggingface spaces 中才会触发
        # 因为 hugggingface 自动处理模型卸载加载，所以如果拿不到就算了...
        return text
    tokenizer = chat_tts.pretrain_models["tokenizer"]
    vocab = tokenizer.get_vocab()
    vocab_set = set(vocab.keys())
    # 添加所有英语字符
    vocab_set.update(set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"))
    vocab_set.update(set(" \n\r\t"))
    replaced_chars = [char if char in vocab_set else " , " for char in text]
    output_text = "".join(replaced_chars)
    return output_text


## ---------- pre normalize ----------


@pre_normalize()
def apply_markdown_to_text(text):
    if is_markdown(text):
        text = markdown_to_text(text)
    return text


# 将 "xxx" => \nxxx\n
# 将 'xxx' => \nxxx\n
@pre_normalize()
def replace_quotes(text):
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


def ensure_suffix(a: str, b: str, c: str):
    a = a.strip()
    if not a.endswith(b):
        a += c
    return a


email_domain_map = {
    "outlook.com": "Out look",
    "hotmail.com": "Hot mail",
    "yahoo.com": "雅虎",
}


# 找到所有 email 并将 name 分割为单个字母，@替换为 at ，. 替换为 dot，常见域名替换为单词
#
# 例如:
# zhzluke96@outlook.com => z h z l u k e 9 6 at out look dot com
def email_detect(text):
    email_pattern = re.compile(r"([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})")

    def replace(match):
        email = match.group(1)
        name, domain = email.split("@")
        name = " ".join(name)
        if domain in email_domain_map:
            domain = email_domain_map[domain]
        domain = domain.replace(".", " dot ")
        return f"{name} at {domain}"

    return email_pattern.sub(replace, text)


def sentence_normalize(sentence_text: str):
    # https://github.com/PaddlePaddle/PaddleSpeech/tree/develop/paddlespeech/t2s/frontend/zh_normalization
    tx = TextNormalizer()

    # 匹配 \[.+?\] 的部分
    pattern = re.compile(r"(\[.+?\])|([^[]+)")

    def normalize_part(part):
        sentences = tx.normalize(part) if guess_lang(part) == "zh" else [part]
        dest_text = ""
        for sentence in sentences:
            sentence = apply_post_normalize(sentence)
            dest_text += sentence
        return dest_text

    def replace(match):
        if match.group(1):
            return f" {match.group(1)} "
        else:
            return normalize_part(match.group(2))

    result = pattern.sub(replace, sentence_text)

    # NOTE: 加了会有杂音...
    # if is_end:
    # 加这个是为了防止吞字
    # result = ensure_suffix(result, "[uv_break]", "。。。[uv_break]。。。")

    return result


def text_normalize(text, is_end=False):
    text = apply_pre_normalize(text)
    lines = text.split("\n")
    lines = [line.strip() for line in lines]
    lines = [line for line in lines if line]
    lines = [sentence_normalize(line) for line in lines]
    content = "\n".join(lines)
    return content


if __name__ == "__main__":
    test_cases = [
        "ChatTTS是专门为对话场景设计的文本转语音模型，例如LLM助手对话任务。它支持英文和中文两种语言。最大的模型使用了10万小时以上的中英文数据进行训练。在HuggingFace中开源的版本为4万小时训练且未SFT的版本.",
        " [oral_9] [laugh_0] [break_0] 电 [speed_0] 影 [speed_0] 中 梁朝伟 [speed_9] 扮演的陈永仁的编号27149",
        " 明天有62％的概率降雨",
        "大🍌，一条大🍌，嘿，你的感觉真的很奇妙  [lbreak]",
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
""",
    ]

    for i, test_case in enumerate(test_cases):
        print(f"case {i}:\n", {"x": text_normalize(test_case, is_end=True)})
