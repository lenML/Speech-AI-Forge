from modules.utils.zh_normalization.text_normlization import *

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


def apply_character_to_word(text):
    for k, v in character_to_word.items():
        text = text.replace(k, v)
    return text


def apply_character_map(text):
    translation_table = str.maketrans(character_map)
    return text.translate(translation_table)


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


def pre_normalize(text):
    # NOTE: 效果一般...
    # text = email_detect(text)
    return text


def post_normalize(text):
    text = insert_spaces_between_uppercase(text)
    text = apply_character_map(text)
    text = apply_character_to_word(text)
    return text


def text_normalize(text, is_end=False):
    # https://github.com/PaddlePaddle/PaddleSpeech/tree/develop/paddlespeech/t2s/frontend/zh_normalization
    tx = TextNormalizer()

    # 匹配 \[.+?\] 的部分
    pattern = re.compile(r"(\[.+?\])|([^[]+)")

    def normalize_part(part):
        part = pre_normalize(part)
        sentences = tx.normalize(part)
        dest_text = ""
        for sentence in sentences:
            dest_text += post_normalize(sentence)
        return dest_text

    def replace(match):
        if match.group(1):
            return f" {match.group(1)} "
        else:
            return normalize_part(match.group(2))

    result = pattern.sub(replace, text)

    # NOTE: 加了会有杂音...
    # if is_end:
    # 加这个是为了防止吞字
    # result = ensure_suffix(result, "[uv_break]", "。。。[uv_break]。。。")

    return result


if __name__ == "__main__":
    print(
        text_normalize(
            "ChatTTS是专门为对话场景设计的文本转语音模型，例如LLM助手对话任务。它支持英文和中文两种语言。最大的模型使用了10万小时以上的中英文数据进行训练。在HuggingFace中开源的版本为4万小时训练且未SFT的版本."
        )
    )
    print(
        text_normalize(
            " [oral_9] [laugh_0] [break_0] 电 [speed_0] 影 [speed_0] 中 梁朝伟 [speed_9] 扮演的陈永仁的编号27149"
        )
    )
    print(text_normalize(" 明天有62％的概率降雨"))
