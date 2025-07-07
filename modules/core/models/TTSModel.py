from typing import Generator, Union

from modules.core.models.AudioReshaper import AudioReshaper
from modules.core.models.BaseZooModel import BaseZooModel
from modules.core.models.tts.InferCache import InferCache
from modules.core.pipeline.dcls import TTSSegment
from modules.core.pipeline.processor import NP_AUDIO, TTSPipelineContext
from modules.core.spk.TTSSpeaker import TTSSpeaker
from modules.devices import devices
from modules.utils import audio_utils

import numpy.typing as npt

class TTSModel(BaseZooModel):

    def __init__(self, model_id: str) -> None:
        super().__init__(model_id=model_id)

    def encode(self, text: str) -> list[int]:
        return [ord(char) for char in text]

    def decode(self, ids: list[int]) -> str:
        return "".join([chr(id) for id in ids])

    def get_sample_rate(self) -> int:
        return 24000

    def generate(self, segment: TTSSegment, context: TTSPipelineContext) -> NP_AUDIO:
        return self.generate_batch([segment], context=context)[0]

    def compute_spk_features(self, spk: TTSSpeaker) -> npt.NDArray:
        """
        计算说话人特征，只有部分模型支持

        用于音色合并或者其他用途
        """
        raise NotImplementedError(
            f"Model {self.model_id} is not support extracting speaker features"
        )

    # NOTE: 这里会有假设，所有的 segments 除了文本以外所有配置相同，具体调用逻辑在 core.pipeline.generate 中
    def generate_batch(
        self, segments: list[TTSSegment], context: TTSPipelineContext
    ) -> list[NP_AUDIO]:
        raise NotImplementedError("generate_batch method is not implemented")

    def generate_stream(
        self, segment: TTSSegment, context: TTSPipelineContext
    ) -> Generator[NP_AUDIO, None, None]:
        for batch in self.generate_batch_stream([segment], context=context):
            yield batch[0]

    def generate_batch_stream(
        self, segments: list[TTSSegment], context: TTSPipelineContext
    ) -> Generator[list[NP_AUDIO], None, None]:
        raise NotImplementedError("generate_batch_stream method is not implemented")

    def get_spk_emb(self, segment: TTSSegment, context: TTSPipelineContext):
        if segment.spk is None:
            return None
        token = segment.spk.get_token(self.model_id)
        if token is None:
            return None
        return token.tokens[0]

    def get_cache_kwargs(self, segments: list[TTSSegment], context: TTSPipelineContext):
        texts = [segment.text for segment in segments]

        # NOTE: 这里感觉有点蛋疼...如果增加新属性，都得改这里，但是如果改用 JSON 直接作为 cache key 的话，又可能太长了...
        # TODO: 得换个好点的方法

        seg0 = segments[0]
        spk = seg0.spk
        spk_id = spk.id if spk else None
        top_P = seg0.top_p
        top_K = seg0.top_k
        temperature = seg0.temperature
        # repetition_penalty = seg0.repetition_penalty
        # max_new_token = seg0.max_new_token
        prompt1 = seg0.prompt1
        prompt2 = seg0.prompt2
        prefix = seg0.prefix
        emotion = seg0.emotion
        # use_decoder = seg0.use_decoder
        seed = seg0.infer_seed
        chunk_size = context.infer_config.stream_chunk_size

        kwargs = dict(
            text="|".join(texts),
            spk_id=spk_id,
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
            emotion=emotion,
        )
        return kwargs

    def _is_skip_cache(self, segments: list[TTSSegment], context: TTSPipelineContext):
        no_cache = context.infer_config.no_cache
        if no_cache:
            return True

        is_random_generate = context.infer_config.seed == -1
        if is_random_generate:
            return True

        return False

    def get_cache(
        self, segments: list[TTSSegment], context: TTSPipelineContext
    ) -> Union[list[NP_AUDIO], None]:
        if self._is_skip_cache(segments=segments, context=context):
            return None

        kwargs = self.get_cache_kwargs(segments=segments, context=context)

        cached = InferCache.get_cache_val(model_id=self.model_id, **kwargs)
        return cached

    def set_cache(
        self,
        segments: list[TTSSegment],
        context: TTSPipelineContext,
        value: list[NP_AUDIO],
    ):
        if self._is_skip_cache(segments=segments, context=context):
            return

        kwargs = self.get_cache_kwargs(segments=segments, context=context)
        InferCache.set_cache_val(model_id=self.model_id, value=value, **kwargs)

    def get_ref_wav(self, segment: TTSSegment):
        spk = segment.spk
        if spk is None:
            return None, None
        emotion = segment.emotion
        ref_data = spk.get_ref(lambda x: x.emotion == emotion)
        if ref_data is None:
            return None, None
        wav = audio_utils.bytes_to_librosa_array(
            audio_bytes=ref_data.wav, sample_rate=ref_data.wav_sr
        )
        _, wav = AudioReshaper.normalize_audio(
            audio=(ref_data.wav_sr, wav),
            target_sr=self.get_sample_rate(),
        )
        text = ref_data.text
        return wav, text
