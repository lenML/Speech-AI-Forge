import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import torch

from modules.core.models.tts.FireRed.FireRedInfer import (
    FireRedTTSInfer,
    FireRedTTSParams,
)
from modules.core.models.TTSModel import TTSModel
from modules.core.pipeline.dcls import TTSPipelineContext
from modules.core.pipeline.pipeline import TTSSegment
from modules.core.pipeline.processor import NP_AUDIO
from modules.devices import devices
from modules.utils.SeedContext import SeedContext

logger = logging.getLogger(__name__)

# TODO: 这个模型没有写cache
class FireRedTTSModel(TTSModel):
    model_id = "fire-red-tts"

    def __init__(self) -> None:
        super().__init__(FireRedTTSModel.model_id)

        self.fire_red: FireRedTTSInfer = None

    def is_downloaded(self) -> bool:
        return Path("models/FireRedTTS").exists()

    def load(self):
        if self.fire_red:
            return self.fire_red
        logger.info("loadding FireRedTTS...")
        self.fire_red = FireRedTTSInfer(
            config_path="./modules/repos_static/FireRedTTS/config_24k.json",
            pretrained_path="./models/FireRedTTS",
            device=self.get_device(),
            dtype=self.get_dtype(),
        )
        logger.info("FireRedTTS model loaded.")
        return self.fire_red

    def unload(self) -> None:
        if self.fire_red is None:
            return
        self.fire_red.unload_models()
        self.fire_red = None
        devices.torch_gc()
        logger.info("FireRedTTS model unloaded.")

    def get_sample_rate(self):
        return 24000

    def generate(
        self, segment: TTSSegment, context: TTSPipelineContext
    ) -> Tuple[NP_AUDIO]:
        model = self.load()

        seg0 = segment
        spk_emb = self.get_spk_emb(segment=seg0, context=context) if seg0.spk else None
        spk_wav, txt_smp = self.get_ref_wav(seg0)
        spk_smp = (
            model.extract_spk_embeddings(spk_wav, audio_sr=self.get_sample_rate())
            if spk_wav is not None
            else None
        )
        top_P = seg0.top_p
        top_K = seg0.top_k
        temperature = seg0.temperature
        # repetition_penalty = seg0.repetition_penalty
        # max_new_token = seg0.max_new_token
        prompt = seg0.prompt
        prompt1 = seg0.prompt1
        prompt2 = seg0.prompt2
        prefix = seg0.prefix
        # use_decoder = seg0.use_decoder
        # seed = seg0.infer_seed
        # chunk_size = context.infer_config.stream_chunk_size

        with SeedContext(seed=seg0.infer_seed), torch.cuda.amp.autocast(
            dtype=self.get_dtype()
        ):
            syn_audio = model.synthesize(
                audio=spk_wav,
                audio_sr=self.get_sample_rate(),
                text=seg0.text,
                # lang="auto",
                params=FireRedTTSParams(
                    top_p=top_P,
                    top_k=top_K,
                    temperature=temperature,
                ),
            )

        wav: np.ndarray = syn_audio.float().cpu().squeeze().numpy()

        return self.get_sample_rate(), wav

    def generate_batch(
        self, segments: list[TTSSegment], context: TTSPipelineContext
    ) -> list[Tuple[NP_AUDIO]]:
        # NOTE: 原生不支持 batch 所以，就是简单的循环

        ret = []
        for seg in segments:
            ret.append(self.generate(segment=seg, context=context))

        return ret
