import re

import zhon

from modules.utils.detect_lang import guess_lang


def char_tokenizer(text: str):
    return [ord(char) for char in text]


# 解析文本 并根据停止符号分割成句子
# 可以设置最大阈值，即如果分割片段小于这个阈值会与下一段合并
class SentenceSplitter:
    # 分隔符 用于连接句子 sentence1 + SEP_TOKEN + sentence2
    SEP_TOKEN = " "

    def __init__(self, threshold=100, tokenizer=char_tokenizer):
        assert (
            isinstance(threshold, int) and threshold > 0
        ), "Threshold must be greater than 0."

        self.sentence_threshold = threshold
        self.tokenizer = tokenizer

    def len(self, text: str):
        """
        Get the length of tokenized text.
        """
        return len(self.tokenizer(text))

    def parse(self, text: str):
        sentences = self.split_paragraph(text)
        sentences = self.merge_text_by_threshold(sentences)

        return sentences

    def merge_text_by_threshold(self, setences: list[str]):
        """
        Merge text by threshold.

        If the length of the text is less than the threshold, merge it with the next text.
        """
        merged_sentences: list[str] = []
        temp_sentence = ""
        for sentence in setences:
            if self.len(temp_sentence) + self.len(sentence) < self.sentence_threshold:
                temp_sentence += SentenceSplitter.SEP_TOKEN + sentence
            else:
                merged_sentences.append(temp_sentence)
                temp_sentence = sentence

        if temp_sentence:
            merged_sentences.append(temp_sentence)
        return merged_sentences

    def split_paragraph(self, text: str):
        """
        Split text into sentences.
        """
        lines = text.split("\n")
        sentences: list[str] = []
        for line in lines:
            if self.is_eng_sentence(line):
                sentences.extend(self.split_en_sentence(line))
            else:
                sentences.extend(self.split_zhon_sentence(line))
        return sentences

    def is_eng_sentence(self, text: str):
        return guess_lang(text) == "en"

    def split_en_sentence(self, text: str):
        """
        Split English text into sentences.
        """
        pattern = re.compile(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s")
        sentences = pattern.split(text)

        sentences = [sentence.strip() for sentence in sentences if sentence.strip()]

        return sentences

    def split_zhon_sentence(self, text: str):
        """
        Split Chinese text into sentences.
        """
        sentences: list[str] = []
        pattern = re.compile(zhon.hanzi.sentence)
        start = 0
        for match in pattern.finditer(text):
            end = match.end()
            sentences.append(text[start:end])
            start = end

        if start < len(text):
            sentences.append(text[start:])

        sentences = [t for t in sentences if t.strip()]
        return sentences


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
