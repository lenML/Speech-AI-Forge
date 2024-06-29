import io
from typing import Generator, Optional, Union

import numpy as np
import torch
from cachetools import LRUCache
from cachetools import keys as cache_keys

from modules import generate_audio as generate
from modules.SentenceSplitter import SentenceSplitter
from modules.speaker import Speaker


def handle_chunks(
    wav_gen: np.ndarray,
    wav_gen_prev: Optional[np.ndarray],
    wav_overlap: Optional[np.ndarray],
    overlap_len: int,
):
    """Handle chunk formatting in streaming mode"""
    wav_chunk = wav_gen[:-overlap_len]
    if wav_gen_prev is not None:
        wav_chunk = wav_gen[(wav_gen_prev.shape[0] - overlap_len) : -overlap_len]
    if wav_overlap is not None:
        # cross fade the overlap section
        if overlap_len > len(wav_chunk):
            # wav_chunk is smaller than overlap_len, pass on last wav_gen
            if wav_gen_prev is not None:
                wav_chunk = wav_gen[(wav_gen_prev.shape[0] - overlap_len) :]
            else:
                # not expecting will hit here as problem happens on last chunk
                wav_chunk = wav_gen[-overlap_len:]
            return wav_chunk, wav_gen, None
        else:
            crossfade_wav = wav_chunk[:overlap_len]
            crossfade_wav = crossfade_wav * np.linspace(0.0, 1.0, overlap_len)
            wav_chunk[:overlap_len] = wav_overlap * np.linspace(1.0, 0.0, overlap_len)
            wav_chunk[:overlap_len] += crossfade_wav

    wav_overlap = wav_gen[-overlap_len:]
    wav_gen_prev = wav_gen
    return wav_chunk, wav_gen_prev, wav_overlap


cache = LRUCache(maxsize=128)


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
    end_of_sentence: str = "",
    overlap_wav_len: int = 1024,
) -> Generator[tuple[int, np.ndarray], None, None]:
    cachekey = cache_keys.hashkey(
        text,
        temperature,
        top_P,
        top_K,
        spk=spk if isinstance(spk, int) else spk.id,
        infer_seed=infer_seed,
        use_decoder=use_decoder,
        prompt1=prompt1,
        prompt2=prompt2,
        prefix=prefix,
        spliter_threshold=spliter_threshold,
        end_of_sentence=end_of_sentence,
        overlap_wav_len=overlap_wav_len,
    )

    if cachekey in cache:
        for sr, wav in cache[cachekey]:
            yield sr, wav
        return

    spliter = SentenceSplitter(spliter_threshold)
    sentences = spliter.parse(text)

    wav_gen_prev = None
    wav_overlap = None

    total_wavs = []
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

        # for sr, wav in wav_gen:
        #     yield sr, wav

        # NOTE: 作用很微妙对质量有一点改善
        for sr, wav in wav_gen:
            total_wavs.append((sr, wav))
            wav_gen = (
                wav if wav_gen_prev is None else np.concatenate([wav_gen_prev, wav])
            )
            wav_chunk, wav_gen_prev, wav_overlap = handle_chunks(
                wav_gen, wav_gen_prev, wav_overlap, overlap_wav_len
            )
            yield sr, wav_chunk

    cache[cachekey] = [(sr, np.concatenate([wav for sr, wav in total_wavs]))]
