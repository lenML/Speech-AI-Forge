from typing import Any, Generator

import numpy as np

from modules.core.models.TTSModel import TTSModel
from modules.core.models.zoo.ChatTTS import ChatTTS, load_chat_tts, unload_chat_tts
from modules.core.models.zoo.ChatTTSInfer import ChatTTSInfer
from modules.core.models.zoo.InerCache import InferCache
from modules.core.pipeline.dcls import TTSPipelineContext
from modules.core.pipeline.pipeline import TTSSegment
from modules.core.pipeline.processor import NP_AUDIO
from modules.utils.SeedContext import SeedContext


class ChatTTSModel(TTSModel):
    model_id = "chat-tts"

    def __init__(self) -> None:
        super().__init__("chat-tts-4w")
        self.chat: ChatTTS = None

    def load(self, context: TTSPipelineContext) -> ChatTTS:
        self.chat = load_chat_tts()
        return self.chat

    def unload(self, context: TTSPipelineContext) -> None:
        unload_chat_tts(self.chat)
        self.chat = None

    def generate_batch(
        self, segments: list[TTSSegment], context: TTSPipelineContext
    ) -> list[NP_AUDIO]:
        return self.generate_batch_base(segments, context, stream=False)

    def generate_batch_stream(
        self, segments: list[TTSSegment], context: TTSPipelineContext
    ) -> Generator[list[NP_AUDIO], Any, None]:
        return self.generate_batch_base(segments, context, stream=True)

    def get_infer(self, context: TTSPipelineContext):
        return ChatTTSInfer(self.load(context=context))

    def get_cache_kwargs(self, segments: list[TTSSegment], context: TTSPipelineContext):
        texts = [segment.text for segment in segments]

        seg0 = segments[0]
        spk_emb = seg0.spk.emb if seg0.spk else None
        top_P = seg0.top_p
        top_K = seg0.top_k
        temperature = seg0.temperature
        # repetition_penalty = seg0.repetition_penalty
        # max_new_token = seg0.max_new_token
        prompt1 = seg0.prompt1
        prompt2 = seg0.prompt2
        prefix = seg0.prefix
        # use_decoder = seg0.use_decoder
        seed = seg0.infer_seed
        chunk_size = context.infer_config.stream_chunk_size

        kwargs = dict(
            text="|".join(texts),
            spk_emb=spk_emb,
            top_P=top_P,
            top_K=top_K,
            temperature=temperature,
            repetition_penalty=None,
            max_new_token=None,
            prompt1=prompt1,
            prompt2=prompt2,
            prefix=prefix,
            stream_chunk_size=chunk_size,
            seed=seed,
        )
        return kwargs

    def get_cache(
        self, segments: list[TTSSegment], context: TTSPipelineContext
    ) -> list[NP_AUDIO]:
        kwargs = self.get_cache_kwargs(segments=segments, context=context)

        if InferCache.get_cache_val(model_id=self.model_id, **kwargs):
            return InferCache.get_cache_val(model_id=self.model_id, **kwargs)

        return None

    def set_cache(
        self,
        segments: list[TTSSegment],
        context: TTSPipelineContext,
        value: list[NP_AUDIO],
    ):
        kwargs = self.get_cache_kwargs(segments=segments, context=context)
        InferCache.set_cache_val(model_id=self.model_id, value=value, **kwargs)

    def generate_batch_base(
        self, segments: list[TTSSegment], context: TTSPipelineContext, stream=False
    ) -> list[NP_AUDIO] | Generator[list[NP_AUDIO], Any, None]:
        cached = self.get_cache(segments=segments, context=context)
        if cached is not None:
            if not stream:
                return cached

            def _gen():
                yield cached

            return _gen()

        infer = self.get_infer(context)

        texts = [segment.text for segment in segments]

        seg0 = segments[0]
        spk_emb = seg0.spk.emb if seg0.spk else None
        top_P = seg0.top_p
        top_K = seg0.top_k
        temperature = seg0.temperature
        # repetition_penalty = seg0.repetition_penalty
        # max_new_token = seg0.max_new_token
        prompt1 = seg0.prompt1
        prompt2 = seg0.prompt2
        prefix = seg0.prefix
        # use_decoder = seg0.use_decoder
        seed = seg0.infer_seed
        chunk_size = context.infer_config.stream_chunk_size

        sr = 24000

        if not stream:
            with SeedContext(seed, cudnn_deterministic=False):
                results = infer.generate_audio(
                    text=texts,
                    spk_emb=spk_emb,
                    top_P=top_P,
                    top_K=top_K,
                    temperature=temperature,
                    prompt1=prompt1,
                    prompt2=prompt2,
                    prefix=prefix,
                )
                audio_arr: list[NP_AUDIO] = [(sr, data[0]) for data in results]

                self.set_cache(segments=segments, context=context, value=audio_arr)
                return audio_arr
        else:

            def _gen() -> Generator[list[NP_AUDIO], None, None]:
                audio_arr_buff = None
                with SeedContext(seed, cudnn_deterministic=False):
                    for results in infer.generate_audio_stream(
                        text=texts,
                        spk_emb=spk_emb,
                        top_P=top_P,
                        top_K=top_K,
                        temperature=temperature,
                        prompt1=prompt1,
                        prompt2=prompt2,
                        prefix=prefix,
                        stream_chunk_size=chunk_size,
                    ):
                        audio_arr: list[NP_AUDIO] = [(sr, data[0]) for data in results]
                        yield audio_arr

                        if audio_arr_buff is None:
                            audio_arr_buff = audio_arr
                        else:
                            for i, data in enumerate(results):
                                sr1, before = audio_arr_buff[i]
                                buff = np.concatenate([before, data[0]], axis=0)
                                audio_arr_buff[i] = (sr1, buff)
                self.set_cache(segments=segments, context=context, value=audio_arr_buff)

            return _gen(infer)
