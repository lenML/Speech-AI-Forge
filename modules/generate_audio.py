import numpy as np
import torch

from modules.speaker import Speaker
from modules.utils.SeedContext import SeedContext

from modules import models


@torch.inference_mode()
def generate_audio(
    text: str,
    temperature: float = 0.3,
    top_P: float = 0.7,
    top_K: float = 20,
    spk: int | Speaker = -1,
    infer_seed: int = -1,
    use_decoder: bool = True,
    prompt1: str = "",
    prompt2: str = "",
    prefix: str = "",
):
    chat_tts = models.load_chat_tts()
    params_infer_code = {
        "spk_emb": None,
        "temperature": temperature,
        "top_P": top_P,
        "top_K": top_K,
        "prompt1": prompt1 or "",
        "prompt2": prompt2 or "",
        "prefix": prefix or "",
    }

    if isinstance(spk, int):
        with SeedContext(spk):
            params_infer_code["spk_emb"] = chat_tts.sample_random_speaker()
        print("spk", spk)
    elif isinstance(spk, Speaker):
        params_infer_code["spk_emb"] = spk.emb
        print("spk", spk.name)

    # print(
    #     {
    #         "text": text,
    #         "infer_seed": infer_seed,
    #         "temperature": temperature,
    #         "top_P": top_P,
    #         "top_K": top_K,
    #         "prompt1": prompt1 or "",
    #         "prompt2": prompt2 or "",
    #         "prefix": prefix or "",
    #     }
    # )

    with SeedContext(infer_seed):
        wav = chat_tts.generate_audio(text, params_infer_code, use_decoder=use_decoder)

    audio_data = np.array(wav[0]).flatten()
    sample_rate = 24000

    return sample_rate, audio_data
