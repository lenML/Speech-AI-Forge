import numpy as np
from fastapi import HTTPException

from modules.api.impl.handler.AudioHandler import AudioHandler
from modules.api.impl.model.audio_model import AdjustConfig
from modules.api.impl.model.chattts_model import InferConfig
from modules.api.impl.model.enhancer_model import EnhancerConfig
from modules.Enhancer.ResembleEnhance import apply_audio_enhance_full
from modules.normalization import text_normalize
from modules.ssml_parser.SSMLParser import create_ssml_parser
from modules.SynthesizeSegments import SynthesizeSegments, combine_audio_segments
from modules.utils import audio


class SSMLHandler(AudioHandler):
    def __init__(
        self,
        ssml_content: str,
        infer_config: InferConfig,
        adjust_config: AdjustConfig,
        enhancer_config: EnhancerConfig,
    ) -> None:
        assert isinstance(ssml_content, str), "ssml_content must be a string."
        assert isinstance(
            infer_config, InferConfig
        ), "infer_config must be an InferConfig object."
        assert isinstance(
            adjust_config, AdjustConfig
        ), "adjest_config should be AdjustConfig"
        assert isinstance(
            enhancer_config, EnhancerConfig
        ), "enhancer_config must be an EnhancerConfig object."

        self.ssml_content = ssml_content
        self.infer_config = infer_config
        self.adjest_config = adjust_config
        self.enhancer_config = enhancer_config

        self.validate()

    def validate(self):
        # TODO params checker
        pass

    def enqueue(self) -> tuple[np.ndarray, int]:
        ssml_content = self.ssml_content
        infer_config = self.infer_config
        adjust_config = self.adjest_config
        enhancer_config = self.enhancer_config

        parser = create_ssml_parser()
        segments = parser.parse(ssml_content)
        for seg in segments:
            seg["text"] = text_normalize(seg["text"], is_end=True)

        if len(segments) == 0:
            raise HTTPException(
                status_code=422, detail="The SSML text is empty or parsing failed."
            )

        synthesize = SynthesizeSegments(
            batch_size=infer_config.batch_size,
            eos=infer_config.eos,
            spliter_thr=infer_config.spliter_threshold,
        )
        audio_segments = synthesize.synthesize_segments(segments)
        combined_audio = combine_audio_segments(audio_segments)

        sample_rate, audio_data = audio.pydub_to_np(combined_audio)

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

        audio_data = audio.apply_prosody_to_audio_data(
            audio_data=audio_data,
            rate=adjust_config.speed_rate,
            pitch=adjust_config.pitch,
            volume=adjust_config.volume_gain_db,
            sr=sample_rate,
        )

        if adjust_config.normalize:
            sample_rate, audio_data = audio.apply_normalize(
                audio_data=audio_data, headroom=adjust_config.headroom, sr=sample_rate
            )

        return audio_data, sample_rate
