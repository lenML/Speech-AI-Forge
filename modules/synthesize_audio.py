import io
from typing import Union
from modules.SentenceSplitter import SentenceSplitter
from modules.SynthesizeSegments import SynthesizeSegments, combine_audio_segments

from modules import generate_audio as generate


from modules.speaker import Speaker
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

        text_segments = [
            {
                "text": s,
                "params": {
                    "text": s,
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
            }
            for s in sentences
        ]
        synthesizer = SynthesizeSegments(batch_size)
        audio_segments = synthesizer.synthesize_segments(text_segments)

        combined_audio = combine_audio_segments(audio_segments)

        return audio.pydub_to_np(combined_audio)
