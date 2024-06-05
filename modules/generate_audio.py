import numpy as np
import torch

from modules.speaker import Speaker
from modules.utils.SeedContext import SeedContext

from modules import models, config

import logging

from modules import devices

logger = logging.getLogger(__name__)


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
    (sample_rate, wav) = generate_audio_batch(
        [text],
        temperature=temperature,
        top_P=top_P,
        top_K=top_K,
        spk=spk,
        infer_seed=infer_seed,
        use_decoder=use_decoder,
        prompt1=prompt1,
        prompt2=prompt2,
        prefix=prefix,
    )[0]

    return (sample_rate, wav)


@torch.inference_mode()
def generate_audio_batch(
    texts: list[str],
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
        "repetition_penalty": 1.0,
        "disable_tqdm": config.disable_tqdm,
    }

    if isinstance(spk, int):
        with SeedContext(spk):
            params_infer_code["spk_emb"] = chat_tts.sample_random_speaker()
        logger.info(("spk", spk))
    elif isinstance(spk, Speaker):
        params_infer_code["spk_emb"] = spk.emb
        logger.info(("spk", spk.name))
    else:
        raise ValueError("spk must be int or Speaker")

    logger.info(
        {
            "text": texts,
            "infer_seed": infer_seed,
            "temperature": temperature,
            "top_P": top_P,
            "top_K": top_K,
            "prompt1": prompt1 or "",
            "prompt2": prompt2 or "",
            "prefix": prefix or "",
        }
    )

    with SeedContext(infer_seed):
        wavs = chat_tts.generate_audio(
            texts, params_infer_code, use_decoder=use_decoder
        )

    sample_rate = 24000

    devices.torch_gc()

    return [(sample_rate, np.array(wav).flatten().astype(np.float32)) for wav in wavs]


if __name__ == "__main__":
    import soundfile as sf

    # 测试batch生成
    inputs = ["你好[lbreak]", "再见[lbreak]", "长度不同的文本片段[lbreak]"]
    outputs = generate_audio_batch(inputs, spk=5, infer_seed=42)

    for i, (sample_rate, wav) in enumerate(outputs):
        print(i, sample_rate, wav.shape)

        sf.write(f"batch_{i}.wav", wav, sample_rate, format="wav")

    # 单独生成
    for i, text in enumerate(inputs):
        sample_rate, wav = generate_audio(text, spk=5, infer_seed=42)
        print(i, sample_rate, wav.shape)

        sf.write(f"one_{i}.wav", wav, sample_rate, format="wav")
