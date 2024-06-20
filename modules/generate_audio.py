import gc
import logging
from typing import Generator, Union

import numpy as np
import torch

from modules import config, models
from modules.ChatTTS import ChatTTS
from modules.devices import devices
from modules.speaker import Speaker
from modules.utils.cache import conditional_cache
from modules.utils.SeedContext import SeedContext

logger = logging.getLogger(__name__)

SAMPLE_RATE = 24000


def generate_audio(
    text: str,
    temperature: float = 0.3,
    top_P: float = 0.7,
    top_K: float = 20,
    spk: Union[int, Speaker] = -1,
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


def parse_infer_params(
    texts: list[str],
    chat_tts: ChatTTS.Chat,
    temperature: float = 0.3,
    top_P: float = 0.7,
    top_K: float = 20,
    spk: Union[int, Speaker] = -1,
    infer_seed: int = -1,
    prompt1: str = "",
    prompt2: str = "",
    prefix: str = "",
):
    params_infer_code = {
        "spk_emb": None,
        "temperature": temperature,
        "top_P": top_P,
        "top_K": top_K,
        "prompt1": prompt1 or "",
        "prompt2": prompt2 or "",
        "prefix": prefix or "",
        "repetition_penalty": 1.0,
        "disable_tqdm": config.runtime_env_vars.off_tqdm,
    }

    if isinstance(spk, int):
        with SeedContext(spk, True):
            params_infer_code["spk_emb"] = chat_tts.sample_random_speaker()
        logger.debug(("spk", spk))
    elif isinstance(spk, Speaker):
        if not isinstance(spk.emb, torch.Tensor):
            raise ValueError("spk.pt is broken, please retrain the model.")
        params_infer_code["spk_emb"] = spk.emb
        logger.debug(("spk", spk.name))
    else:
        logger.warn(
            f"spk must be int or Speaker, but: <{type(spk)}> {spk}, wiil set to default voice"
        )
        with SeedContext(2, True):
            params_infer_code["spk_emb"] = chat_tts.sample_random_speaker()

    logger.debug(
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

    return params_infer_code


@torch.inference_mode()
def generate_audio_batch(
    texts: list[str],
    temperature: float = 0.3,
    top_P: float = 0.7,
    top_K: float = 20,
    spk: Union[int, Speaker] = -1,
    infer_seed: int = -1,
    use_decoder: bool = True,
    prompt1: str = "",
    prompt2: str = "",
    prefix: str = "",
):
    chat_tts = models.load_chat_tts()
    params_infer_code = parse_infer_params(
        texts=texts,
        chat_tts=chat_tts,
        temperature=temperature,
        top_P=top_P,
        top_K=top_K,
        spk=spk,
        infer_seed=infer_seed,
        prompt1=prompt1,
        prompt2=prompt2,
        prefix=prefix,
    )

    with SeedContext(infer_seed, True):
        wavs = chat_tts.generate_audio(
            texts=texts, params_infer_code=params_infer_code, use_decoder=use_decoder
        )

    if config.auto_gc:
        devices.torch_gc()
        gc.collect()

    return [(SAMPLE_RATE, np.array(wav).flatten().astype(np.float32)) for wav in wavs]


# TODO: generate_audio_stream 也应该支持 lru cache
@torch.inference_mode()
def generate_audio_stream(
    text: str,
    temperature: float = 0.3,
    top_P: float = 0.7,
    top_K: float = 20,
    spk: Union[int, Speaker] = -1,
    infer_seed: int = -1,
    use_decoder: bool = True,
    prompt1: str = "",
    prompt2: str = "",
    prefix: str = "",
) -> Generator[tuple[int, np.ndarray], None, None]:
    chat_tts = models.load_chat_tts()
    texts = [text]
    params_infer_code = parse_infer_params(
        texts=texts,
        chat_tts=chat_tts,
        temperature=temperature,
        top_P=top_P,
        top_K=top_K,
        spk=spk,
        infer_seed=infer_seed,
        prompt1=prompt1,
        prompt2=prompt2,
        prefix=prefix,
    )

    with SeedContext(infer_seed, True):
        wavs_gen = chat_tts.generate_audio(
            prompt=texts,
            params_infer_code=params_infer_code,
            use_decoder=use_decoder,
            stream=True,
        )

        for wav in wavs_gen:
            yield [SAMPLE_RATE, np.array(wav).flatten().astype(np.float32)]

    if config.auto_gc:
        devices.torch_gc()
        gc.collect()

    return


lru_cache_enabled = False


def setup_lru_cache():
    global generate_audio_batch
    global lru_cache_enabled

    if lru_cache_enabled:
        return
    lru_cache_enabled = True

    def should_cache(*args, **kwargs):
        spk_seed = kwargs.get("spk", -1)
        infer_seed = kwargs.get("infer_seed", -1)
        return spk_seed != -1 and infer_seed != -1

    lru_size = config.runtime_env_vars.lru_size
    if isinstance(lru_size, int):
        generate_audio_batch = conditional_cache(lru_size, should_cache)(
            generate_audio_batch
        )
        logger.info(f"LRU cache enabled with size {lru_size}")
    else:
        logger.debug(f"LRU cache failed to enable, invalid size {lru_size}")


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
