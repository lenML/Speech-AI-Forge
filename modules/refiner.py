import numpy as np
import torch

from modules import config, models
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
        refined_text = chat_tts.refiner_prompt(
            text,
            {
                "prompt": prompt,
                "top_K": top_K,
                "top_P": top_P,
                "temperature": temperature,
                "repetition_penalty": repetition_penalty,
                "max_new_token": max_new_token,
                "disable_tqdm": config.runtime_env_vars.off_tqdm,
            },
        )
        return refined_text
