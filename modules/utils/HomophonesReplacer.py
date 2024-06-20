import json


# ref: https://github.com/2noise/ChatTTS/commit/ce1c962b6235bd7d0c637fbdcda5e2dccdbac80d
class HomophonesReplacer:
    """
    Homophones Replacer

    Replace the mispronounced characters with correctly pronounced ones.

    Creation process of homophones_map.json:

    1. Establish a word corpus using the [Tencent AI Lab Embedding Corpora v0.2.0 large] with 12 million entries. After cleaning, approximately 1.8 million entries remain. Use ChatTTS to infer the text.
    2. Record discrepancies between the inferred and input text, identifying about 180,000 misread words.
    3. Create a pinyin to common characters mapping using correctly read characters by ChatTTS.
    4. For each discrepancy, extract the correct pinyin using [python-pinyin] and find homophones with the correct pronunciation from the mapping.

    Thanks to:
    [Tencent AI Lab Embedding Corpora for Chinese and English Words and Phrases](https://ai.tencent.com/ailab/nlp/en/embedding.html)
    [python-pinyin](https://github.com/mozillazg/python-pinyin)

    """

    def __init__(self, map_file_path):
        self.homophones_map = self.load_homophones_map(map_file_path)

    def load_homophones_map(self, map_file_path):
        with open(map_file_path, "r", encoding="utf-8") as f:
            homophones_map = json.load(f)
        return homophones_map

    def replace(self, text):
        result = []
        for char in text:
            if char in self.homophones_map:
                result.append(self.homophones_map[char])
            else:
                result.append(char)
        return "".join(result)
