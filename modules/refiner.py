from typing import Generator

import torch

from modules.core.models.tts.ChatTTS import ChatTTS
from modules.core.models.tts.ChatTTS.ChatTTSInfer import ChatTTSInfer
from modules.core.tools.SentenceSplitter import SentenceSplitter
from modules.utils.SeedContext import SeedContext

"""
TODO: 应该重构一下，增加 refine model 而不是从这里调用
"""
@torch.inference_mode()
def refine_text(
    text: str,
    prompt="[oral_2][laugh_0][break_6]",
    seed=-1,
    top_P=0.7,
    top_K=20,
    temperature=0.7,
    repetition_penalty=1.0,
    max_new_token=384,
    spliter_threshold=300,
) -> str:
    chat_tts = ChatTTS.load_chat_tts()

    spliter = SentenceSplitter(spliter_threshold)
    sentences = spliter.parse(text)

    with SeedContext(seed):
        infer = ChatTTSInfer(chat_tts)
        results: list[str] = []
        for senc in sentences:
            refined_text = infer.refine_text(
                text=senc,
                prompt=prompt,
                top_P=top_P,
                top_K=top_K,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                max_new_token=max_new_token,
            )
            if isinstance(refined_text, Generator):
                refined_text = list(refined_text)
            if isinstance(refined_text, list):
                refined_text = "\n".join(refined_text)
            results.append(refined_text)
        return "\n".join(results).strip()
