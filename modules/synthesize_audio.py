from typing import Union

from modules.SentenceSplitter import SentenceSplitter
from modules.speaker import Speaker
from modules.ssml_parser.SSMLParser import SSMLSegment
from modules.SynthesizeSegments import SynthesizeSegments, combine_audio_segments
from modules.utils import audio


def synthesize_audio(
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
    batch_size: int = 1,
    spliter_threshold: int = 100,
    end_of_sentence="",
):
    spliter = SentenceSplitter(spliter_threshold)
    sentences = spliter.parse(text)

    text_segments = [
        SSMLSegment(
            text=s,
            params={
                "temperature": temperature,
                "top_P": top_P,
                "top_K": top_K,
                "spk": spk,
                "infer_seed": infer_seed,
                "use_decoder": use_decoder,
                "prompt1": prompt1,
                "prompt2": prompt2,
                "prefix": prefix,
            },
        )
        for s in sentences
    ]
    synthesizer = SynthesizeSegments(
        batch_size=batch_size, eos=end_of_sentence, spliter_thr=spliter_threshold
    )
    audio_segments = synthesizer.synthesize_segments(text_segments)

    combined_audio = combine_audio_segments(audio_segments)

    return audio.pydub_to_np(combined_audio)
