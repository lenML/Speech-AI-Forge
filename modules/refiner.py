from typing import Generator

import numpy as np
import torch

from modules import config, models
from modules.ChatTTSInfer import ChatTTSInfer
from modules.utils.SeedContext import SeedContext


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
) -> str:
    chat_tts = models.load_chat_tts()

    with SeedContext(seed):
        infer = ChatTTSInfer(chat_tts)
        refined_text = infer.refine_text(
            text=text,
            prompt=prompt,
            top_P=top_P,
            top_K=top_K,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            max_new_token=max_new_token,
        )
        if isinstance(refined_text, Generator):
            raise NotImplementedError(
                "Refiner is not yet implemented for generator output"
            )
        if isinstance(refined_text, list):
            refined_text = "\n".join(refined_text)
        return refined_text
