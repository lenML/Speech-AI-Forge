try:
    # NOTE: 这里调一下是为了方便 test
    from modules.repos_static.sys_paths import setup_repos_paths

    setup_repos_paths()
except:
    pass

import torch

from modules.repos_static.FireRedTTS.fireredtts.modules.tokenizer.whisper_tokenizer import (
    get_tokenizer,
)
from modules.utils.detect_lang import guess_lang


# NOTE: 为什么需要这个: https://github.com/lenML/Speech-AI-Forge/issues/178
class FRBepTokenizer:
    def __init__(self):
        self.tokenizer = get_tokenizer(multilingual=True)

    def encode(self, text, lang="auto"):
        if lang == "auto":
            lang = guess_lang(text)
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
    tok = FRBepTokenizer()
    codes = tok.encode("我、真是hello USA啊？谢谢你world！")
    print([tok.decode([c]) for c in codes])
