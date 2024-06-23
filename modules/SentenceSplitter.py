import re

import langdetect
import zhon


def split_zhon_sentence(text):
    result = []
    pattern = re.compile(zhon.hanzi.sentence)
    start = 0
    for match in pattern.finditer(text):
        # 获取匹配的中文句子
        end = match.end()
        result.append(text[start:end])
        start = end

    # 最后一个中文句子后面的内容（如果有）也需要添加到结果中
    if start < len(text):
        result.append(text[start:])

    result = [t for t in result if t.strip()]
    return result


def split_en_sentence(text):
    """
    Split English text into sentences.
    """
    # Define a regex pattern for English sentence splitting
    pattern = re.compile(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s")
    result = pattern.split(text)

    # Filter out any empty strings or strings that are just whitespace
    result = [sentence.strip() for sentence in result if sentence.strip()]

    return result


def is_eng_sentence(text):
    try:
        return langdetect.detect(text) == "en"
    except langdetect.LangDetectException:
        return False


def split_zhon_paragraph(text):
    lines = text.split("\n")
    result = []
    for line in lines:
        if is_eng_sentence(line):
            result.extend(split_en_sentence(line))
        else:
            result.extend(split_zhon_sentence(line))
    return result


# 解析文本 并根据停止符号分割成句子
# 可以设置最大阈值，即如果分割片段小于这个阈值会与下一段合并
class SentenceSplitter:
    def __init__(self, threshold=100):
        self.sentence_threshold = threshold

    def parse(self, text):
        sentences = split_zhon_paragraph(text)

        # 合并小于最大阈值的片段
        merged_sentences = []
        temp_sentence = []
        for sentence in sentences:
            if len(sentence) < self.sentence_threshold:
                temp_sentence.extend(sentence)
                if len(temp_sentence) >= self.sentence_threshold:
                    merged_sentences.append(temp_sentence)
                    temp_sentence = []
            else:
                if temp_sentence:
                    merged_sentences.append(temp_sentence)
                    temp_sentence = []
                merged_sentences.append(sentence)

        if temp_sentence:
            merged_sentences.append(temp_sentence)

        joind_sentences = [
            "".join(sentence) for sentence in merged_sentences if sentence
        ]
        return joind_sentences


if __name__ == "__main__":
    max_threshold = 100
    parser = SentenceSplitter(max_threshold)
    text = """
中华美食，作为世界饮食文化的瑰宝，以其丰富的种类、独特的风味和精湛的烹饪技艺而闻名于世。中国地大物博，各地区的饮食习惯和烹饪方法各具特色，形成了独树一帜的美食体系。从北方的京鲁菜、东北菜，到南方的粤菜、闽菜，无不展现出中华美食的多样性。

在中华美食的世界里，五味调和，色香味俱全。无论是辣味浓郁的川菜，还是清淡鲜美的淮扬菜，都能够满足不同人的口味需求。除了味道上的独特，中华美食还注重色彩的搭配和形态的美感，让每一道菜品不仅是味觉的享受，更是一场视觉的盛宴。

中华美食不仅仅是食物，更是一种文化的传承。每一道菜背后都有着深厚的历史背景和文化故事。比如，北京的烤鸭，代表着皇家气派；而西安的羊肉泡馍，则体现了浓郁的地方风情。中华美食的精髓在于它追求的“天人合一”，讲究食材的自然性和烹饪过程中的和谐。

总之，中华美食博大精深，其丰富的口感和多样的烹饪技艺，构成了一个充满魅力和无限可能的美食世界。无论你来自哪里，都会被这独特的美食文化所吸引和感动。
    """
    result = parser.parse(text)
    for idx, sentence in enumerate(result):
        print(f"Sentence {idx + 1}: {sentence}")
