from typing import Generator

from modules.core.models.TTSModel import TTSModel
from modules.core.models.zoo.ChatTTS import ChatTTS, load_chat_tts, unload_chat_tts
from modules.core.models.zoo.ChatTTSInfer import ChatTTSInfer
from modules.core.pipeline.dcls import TTSPipelineContext
from modules.core.pipeline.pipeline import TTSSegment
from modules.core.pipeline.processor import NP_AUDIO
from modules.utils.SeedContext import SeedContext


class ChatTTSModel(TTSModel):

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
    ) -> Generator[list[NP_AUDIO], None, None]:
        return self.generate_batch_base(segments, context, stream=True)

    def get_infer(self, context: TTSPipelineContext):
        return ChatTTSInfer(self.load(context=context))

    def generate_batch_base(
        self, segments: list[TTSSegment], context: TTSPipelineContext, stream=False
    ):
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
                return [(sr, data[0]) for data in results]
        else:

            def _gen():
                with SeedContext(seed, cudnn_deterministic=False):
                    for data in infer.generate_audio_stream(
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
                        yield (sr, data[0])

            return _gen(infer)
