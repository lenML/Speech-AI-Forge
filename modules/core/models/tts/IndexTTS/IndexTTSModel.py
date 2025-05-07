import io
import os
from typing import Generator
import numpy as np
from omegaconf import OmegaConf
import torch
from modules.core.models.TTSModel import TTSModel
from modules.core.pipeline.dcls import TTSPipelineContext, TTSSegment
from modules.core.pipeline.processor import NP_AUDIO
from modules.devices import devices
from modules.repos_static.index_tts.indextts.infer import IndexTTS
from modules.repos_static.index_tts.indextts.utils.front import TextTokenizer
import soundfile as sf

from modules.utils.SeedContext import SeedContext


class IndexTTSModel(TTSModel):
    model_id = "index-tts"

    def __init__(self):
        super().__init__("index-tts")
        self.tts: IndexTTS = None
        self.device = devices.get_device_for("index-tts")
        self.dtype = devices.dtype
        self.cfg_path = "./modules/repos_static/index_tts/checkpoints/config.yaml"
        self.model_dir = "./models/Index-TTS"
        self.cfg = OmegaConf.load(self.cfg_path)
        self.bpe_path = os.path.join(self.model_dir, self.cfg.dataset["bpe_model"])
        self.tokenizer = TextTokenizer(self.bpe_path)

    def encode(self, text):
        return self.tokenizer.encode(text)

    def decode(self, ids):
        return self.tokenizer.decode(ids)

    def get_sample_rate(self):
        # 来自 modules/repos_static/index_tts/checkpoints/config.yaml
        return 24000

    def is_loaded(self):
        return self.tts is not None

    def load(self):
        if self.tts:
            return
        self.tts = IndexTTS(
            cfg_path=self.cfg_path,
            model_dir=self.model_dir,
            is_fp16=self.dtype == torch.float16,
            # 这个好像可以加速，但是需要冷启动
            # TODO: 暂时不实现，有需要的自行打开即可
            use_cuda_kernel=False,
            device=self.device,
        )

    @devices.after_gc()
    def unload(self):
        if self.tts is not None:
            del self.tts.gpt
            del self.tts.bigvgan
            self.tts = None

    def generate_batch(
        self, segments: list[TTSSegment], context: TTSPipelineContext
    ) -> list[NP_AUDIO]:
        generator = self.generate_batch_stream(segments, context)
        return next(generator)

    def generate_batch_stream(
        self, segments, context
    ) -> Generator[list[NP_AUDIO], None, None]:
        cached = self.get_cache(segments=segments, context=context)
        if cached is not None:
            yield cached
            return

        self.load()

        seg0 = segments[0]
        infer_seed = seg0.infer_seed
        top_p = seg0.top_p
        temperature = seg0.temperature
        # repetition_penalty = seg0.repetition_penalty

        # NOTE: 这个模型不需要 ref_txt
        sr = self.get_sample_rate()
        ref_wav, ref_txt = self.get_ref_wav(seg0)
        prompt_wav = io.BytesIO()
        sf.write(prompt_wav, ref_wav, sr, format="WAV")
        prompt_wav.seek(0)
        ret = []
        for segment in segments:
            # NOTE: 目前暂时没有释放其他配置参数...因为它官方repo里面居然是写死的...
            # TODO: 完全实现带参数的 infer
            with SeedContext(infer_seed):
                wav_sr, wav_data = self.tts.infer_fast(
                    audio_prompt=prompt_wav,
                    text=segment.text,
                    verbose=False,
                    output_path=None,
                )
            wav_data: np.ndarray = wav_data
            wav_data = wav_data.reshape(-1).astype(np.float32) / np.iinfo(np.int16).max
            ret.append((wav_sr, wav_data))
        if not context.stop:
            self.set_cache(segments=segments, context=context, value=ret)
        yield ret
