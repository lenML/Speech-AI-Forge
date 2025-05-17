if __name__ == "__main__":
    from modules.repos_static.sys_paths import setup_repos_paths

    setup_repos_paths()

import logging
import threading
from pathlib import Path
from typing import Generator, Optional, Union

import hydra
import torch
import torchaudio
from hydra import compose, initialize
from hydra.utils import instantiate

from modules import config
from modules.core.models.tts.fishspeech.FF14_infer import FF14_infer
from modules.core.models.tts.fishspeech.FF14_llama import FF14_llama
from modules.core.models.tts.fishspeech.FF14_vqgan import FF14_vqgan
from modules.core.models.tts.FishSpeechInfer import FishSpeechInfer
from modules.core.models.TTSModel import TTSModel
from modules.core.pipeline.dcls import TTSPipelineContext, TTSSegment
from modules.core.pipeline.processor import NP_AUDIO
from modules.devices import devices
from modules.repos_static.fish_speech.fish_speech.models.text2semantic.llama import (
    DualARTransformer,
    NaiveTransformer,
)
from modules.repos_static.fish_speech.fish_speech.models.vqgan.modules.firefly import (
    FireflyArchitecture,
)
from modules.repos_static.fish_speech.tools.llama.generate import (
    load_model as load_llama_model,
)
from modules.repos_static.fish_speech.tools.vqgan.inference import (
    load_model as load_vqgan_model,
)
from modules.utils.SeedContext import SeedContext

logger = logging.getLogger(__name__)


class FishSpeech14Model(TTSModel):
    lock = threading.Lock()

    def __init__(self) -> None:
        model_id = "fish-speech"
        super().__init__(model_id)

        self.model: Optional[FF14_infer] = None

    def is_downloaded(self):
        return FF14_llama.MODEL_PATH.exists() and FF14_vqgan.MODEL_PATH.exists()

    def load(self) -> None:
        with self.lock:
            if self.model is not None:
                logger.info("Model is already loaded")
                return
            self.model = FF14_infer()
            logger.info("Model is loaded")

    def unload(self) -> None:
        if self.model is None:
            return
        self.model.unload()
        del self.model
        self.model = None
        devices.torch_gc()

    def is_loaded(self) -> bool:
        return self.model is not None

    def get_sample_rate(self) -> int:
        # 来自 modules/repos_static/fish_speech/fish_speech/configs/firefly_gan_vq.yaml
        return 44100

    def generate_batch(
        self, segments: list[TTSSegment], context: TTSPipelineContext
    ) -> list[NP_AUDIO]:
        generator = self.generate_batch_stream(segments, context)
        return next(generator)

    # NOTE: 不支持batch生成 所以基本上是同步的
    def generate_batch_stream(
        self, segments: list[TTSSegment], context: TTSPipelineContext
    ) -> Generator[list[NP_AUDIO], None, None]:
        cached = self.get_cache(segments=segments, context=context)
        if cached is not None:
            yield cached
            return

        self.load()
        model = self.model

        seg0 = segments[0]
        infer_seed = seg0.infer_seed
        top_p = seg0.top_p
        temperature = seg0.temperature
        # repetition_penalty = seg0.repetition_penalty

        ref_wav, ref_txt = self.get_ref_wav(seg0)

        sr = self.get_sample_rate()
        ret = []
        for segment in segments:
            if context.stop:
                break

            with SeedContext(seed=infer_seed):
                generated = model.generate(
                    text=segment.text,
                    ref_wav=ref_wav,
                    ref_text=ref_txt,
                    top_p=top_p,
                    temperature=temperature,
                    # repetition_penalty=repetition_penalty,
                )

            ret.append((sr, generated))
        if not context.stop:
            self.set_cache(segments=segments, context=context, value=ret)
        yield ret


if __name__ == "__main__":
    import numpy as np
    import soundfile as sf

    from modules.core.spk.SpkMgr import spk_mgr

    logger.setLevel(logging.DEBUG)

    # 测试模型
    tts_model = FishSpeech14Model()
    # tts_model.load()

    spk = spk_mgr.get_speaker("mona")

    def create_seg(text: str, seed=42):
        return TTSSegment(_type="text", text=text, infer_seed=seed, spk=spk)

    sr, audio_data = tts_model.generate(
        segment=create_seg(text="你好"),
        context=TTSPipelineContext(),
    )

    sf.write(f"test_fish_speech_14.wav", audio_data, sr)
