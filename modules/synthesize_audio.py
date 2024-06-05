from modules.SentenceSplitter import SentenceSplitter
from modules.normalization import text_normalize

from modules import generate_audio as generate


import numpy as np

from modules.speaker import Speaker


def synthesize_audio(
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
    batch_size: int = 1,
    spliter_threshold: int = 100,
):
    if batch_size == 1:
        return generate.generate_audio(
            text,
            temperature=temperature,
            top_P=top_P,
            top_K=top_K,
            spk=spk,
            infer_seed=infer_seed,
            use_decoder=use_decoder,
            prompt1=prompt1,
            prompt2=prompt2,
            prefix=prefix,
        )
    else:
        spliter = SentenceSplitter(spliter_threshold)
        sentences = spliter.parse(text)
        sentences = [text_normalize(s) for s in sentences]
        audio_data_batch = generate.generate_audio_batch(
            texts=sentences,
            temperature=temperature,
            top_P=top_P,
            top_K=top_K,
            spk=spk,
            infer_seed=infer_seed,
            use_decoder=use_decoder,
            prompt1=prompt1,
            prompt2=prompt2,
            prefix=prefix,
        )
        sample_rate = audio_data_batch[0][0]
        audio_data = np.concatenate([data for _, data in audio_data_batch])

        return sample_rate, audio_data
