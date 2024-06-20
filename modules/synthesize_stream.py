import io
from typing import Generator, Union

import numpy as np

from modules import generate_audio as generate
from modules.SentenceSplitter import SentenceSplitter
from modules.speaker import Speaker


def synthesize_stream(
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
    spliter_threshold: int = 100,
    end_of_sentence="",
) -> Generator[tuple[int, np.ndarray], None, None]:
    spliter = SentenceSplitter(spliter_threshold)
    sentences = spliter.parse(text)

    for sentence in sentences:
        wav_gen = generate.generate_audio_stream(
            text=sentence + end_of_sentence,
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
        for sr, wav in wav_gen:
            yield sr, wav
