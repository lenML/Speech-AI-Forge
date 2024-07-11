from typing import Union

from modules.core.speaker import Speaker
from modules.core.ssml.SSMLParser import SSMLSegment
from modules.core.ssml.SynthesizeSSML import SynthesizeSSML
from modules.core.tools.SentenceSplitter import SentenceSplitter


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
    synthesizer = SynthesizeSSML(
        batch_size=batch_size, eos=end_of_sentence, spliter_thr=spliter_threshold
    )
    return synthesizer.synthesize_combine_np(text_segments)
