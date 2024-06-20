from typing import Generator
import numpy as np

from modules.api.impl.handler.AudioHandler import AudioHandler
from modules.api.impl.model.audio_model import AdjustConfig
from modules.api.impl.model.chattts_model import ChatTTSConfig, InferConfig
from modules.api.impl.model.enhancer_model import EnhancerConfig
from modules.Enhancer.ResembleEnhance import apply_audio_enhance_full
from modules.normalization import text_normalize
from modules.speaker import Speaker
from modules.synthesize_audio import synthesize_audio
from modules.synthesize_stream import synthesize_stream
from modules.utils.audio import apply_prosody_to_audio_data

import logging

logger = logging.getLogger(__name__)


class TTSHandler(AudioHandler):
    def __init__(
        self,
        text_content: str,
        spk: Speaker,
        tts_config: ChatTTSConfig,
        infer_config: InferConfig,
        adjust_config: AdjustConfig,
        enhancer_config: EnhancerConfig,
    ):
        assert isinstance(text_content, str), "text_content should be str"
        assert isinstance(spk, Speaker), "spk should be Speaker"
        assert isinstance(
            tts_config, ChatTTSConfig
        ), "tts_config should be ChatTTSConfig"
        assert isinstance(
            infer_config, InferConfig
        ), "infer_config should be InferConfig"
        assert isinstance(
            adjust_config, AdjustConfig
        ), "adjest_config should be AdjustConfig"
        assert isinstance(
            enhancer_config, EnhancerConfig
        ), "enhancer_config should be EnhancerConfig"

        self.text_content = text_content
        self.spk = spk
        self.tts_config = tts_config
        self.infer_config = infer_config
        self.adjest_config = adjust_config
        self.enhancer_config = enhancer_config

        self.validate()

    def validate(self):
        # TODO params checker
        pass

    def enqueue(self) -> tuple[np.ndarray, int]:
        text = text_normalize(self.text_content)
        tts_config = self.tts_config
        infer_config = self.infer_config
        adjust_config = self.adjest_config
        enhancer_config = self.enhancer_config

        sample_rate, audio_data = synthesize_audio(
            text,
            spk=self.spk,
            temperature=tts_config.temperature,
            top_P=tts_config.top_p,
            top_K=tts_config.top_k,
            prompt1=tts_config.prompt1,
            prompt2=tts_config.prompt2,
            prefix=tts_config.prefix,
            infer_seed=infer_config.seed,
            batch_size=infer_config.batch_size,
            spliter_threshold=infer_config.spliter_threshold,
            end_of_sentence=infer_config.eos,
        )

        if enhancer_config.enabled:
            nfe = enhancer_config.nfe
            solver = enhancer_config.solver
            lambd = enhancer_config.lambd
            tau = enhancer_config.tau

            audio_data, sample_rate = apply_audio_enhance_full(
                audio_data=audio_data,
                sr=sample_rate,
                nfe=nfe,
                solver=solver,
                lambd=lambd,
                tau=tau,
            )

        audio_data = apply_prosody_to_audio_data(
            audio_data=audio_data,
            rate=adjust_config.speed_rate,
            pitch=adjust_config.pitch,
            volume=adjust_config.volume_gain_db,
            sr=sample_rate,
        )

        return audio_data, sample_rate

    def enqueue_stream(self) -> Generator[tuple[np.ndarray, int], None, None]:
        text = text_normalize(self.text_content)
        tts_config = self.tts_config
        infer_config = self.infer_config
        adjust_config = self.adjest_config
        enhancer_config = self.enhancer_config

        if enhancer_config.enabled:
            logger.warning(
                "enhancer_config is enabled, but it is not supported in stream mode"
            )

        gen = synthesize_stream(
            text,
            spk=self.spk,
            temperature=tts_config.temperature,
            top_P=tts_config.top_p,
            top_K=tts_config.top_k,
            prompt1=tts_config.prompt1,
            prompt2=tts_config.prompt2,
            prefix=tts_config.prefix,
            infer_seed=infer_config.seed,
            spliter_threshold=infer_config.spliter_threshold,
            end_of_sentence=infer_config.eos,
        )

        for sr, wav in gen:
            yield wav, sr
