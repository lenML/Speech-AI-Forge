from modules.utils.zh_normalization.text_normlization import *
import emojiswitch
from modules.utils.markdown import markdown_to_text

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
    ":": ",",
    ";": ",",
    "!": ".",
    "(": ",",
    ")": ",",
    # '[': ',',
    # ']': ',',
    ">": ",",
    "<": ",",
    "-": ",",
}

character_to_word = {
    " & ": " and ",
}


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
    return emojiswitch.demojize(text, delimiters=("", ""), lang="zh")


@pre_normalize()
def apply_markdown_to_text(text):
    if is_markdown(text):
        text = markdown_to_text(text)
    return text


@post_normalize()
def insert_spaces_between_uppercase(s):
    # 使用正则表达式在每个相邻的大写字母之间插入空格
    return re.sub(
        r"(?<=[A-Z])(?=[A-Z])|(?<=[a-z])(?=[A-Z])|(?<=[\u4e00-\u9fa5])(?=[A-Z])|(?<=[A-Z])(?=[\u4e00-\u9fa5])",
        " ",
        s,
    )


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
        sentences = tx.normalize(part)
        dest_text = ""
        for sentence in sentences:
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
    content = apply_post_normalize(content)
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
    ]

    for i, test_case in enumerate(test_cases):
        print(f"case {i}:\n", {"x": text_normalize(test_case, is_end=True)})
