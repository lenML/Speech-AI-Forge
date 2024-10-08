import os
import torch
from fireredtts.modules.tokenizer.whisper_tokenizer import get_tokenizer
from fireredtts.modules.text_normalizer.normalize import TextNormalizer


DEFAULT_VOCAB_FILE = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "../data/tokenizer.json"
)


class VoiceBpeTokenizer:
    def __init__(self):
        self.tokenizer = get_tokenizer(multilingual=True)
        self.tn_engine = TextNormalizer()

    def redtts_text_cleaner(self, text):
        text = text.strip()
        text, text_lang = self.tn_engine.tn(text)
        # print("---text after tn:", text)
        return text, text_lang

    def encode(self, text, lang="auto"):
        text, text_lang = self.redtts_text_cleaner(text=text)
        if lang == "auto":
            lang = text_lang
        text = f"[{lang}]{text}"
        return self.tokenizer.encode(text)

    def decode(self, seq):
        if isinstance(seq, torch.Tensor):
            seq = seq.cpu().numpy()
        text = self.tokenizer.decode(seq)
        return text

    def __len__(self):
        return self.tokenizer.get_vocab_size()

    def get_number_tokens(self):
        return self.tokenizer.get_vocab_size()


if __name__ == "__main__":
    tok = VoiceBpeTokenizer()
    codes = tok.encode("我、真是hello USA啊？谢谢你world！")
    print([tok.decode([c]) for c in codes])
