import io
import os
from pathlib import Path
from typing import Generator

import numpy as np
import soundfile as sf
import torch
from omegaconf import OmegaConf

from modules.core.models.TTSModel import TTSModel
from modules.core.pipeline.dcls import TTSPipelineContext, TTSSegment
from modules.core.pipeline.processor import NP_AUDIO
from modules.devices import devices
from modules.repos_static.index_tts.indextts.infer import IndexTTS
from modules.repos_static.index_tts.indextts.utils.front import TextTokenizer
from modules.utils.SeedContext import SeedContext

import logging

logger = logging.getLogger(__name__)

class IndexTTSModel(TTSModel):
    model_id = "index-tts"

    def __init__(self):
        super().__init__("index-tts")
        self.tts: IndexTTS = None
        model_v1_dir = Path("./models/Index-TTS")
        model_v1_5_dir = Path("./models/Index-TTS-1.5")
        self.model_dir = model_v1_5_dir if model_v1_5_dir.exists() else model_v1_dir
        self.tokenizer: TextTokenizer = None

        if model_v1_dir.exists() and not model_v1_5_dir.exists():
            logger.warning(
                "Index-TTS 模型已经更新，建议使用 Index-TTS-1.5 模型，使用 1.5 下载脚本下载最新模型即可使用。"
            )

    def is_downloaded(self):
        return self.model_dir.exists()

    def is_loaded(self):
        return self.tts is not None

    def load_tokenizer(self):
        if self.tokenizer is None:
            cfg_path = self.model_dir / "config.yaml"
            cfg = OmegaConf.load(cfg_path)
            bpe_path = self.model_dir / cfg.dataset["bpe_model"]
            self.tokenizer = TextTokenizer(str(bpe_path))
        return self.tokenizer

    def encode(self, text):
        self.load_tokenizer()
        return self.tokenizer.encode(text)

    def decode(self, ids):
        self.load_tokenizer()
        return self.tokenizer.decode(ids)

    def get_sample_rate(self):
        # 来自 modules/repos_static/index_tts/checkpoints/config.yaml
        # NOTE: 其实应该从 config.yaml 里取，但是 v1 v1.5 都一样，所以直接写死得了，因为加载配置需要外部依赖库
        return 24000

    def is_loaded(self):
        return self.tts is not None

    def load(self):
        if self.tts:
            return
        cfg_path = self.model_dir / "config.yaml"
        self.tts = IndexTTS(
            cfg_path=str(cfg_path),
            model_dir=str(self.model_dir),
            is_fp16=self.get_dtype() == torch.float16,
            # 这个好像可以加速，但是需要冷启动
            # TODO: 暂时不实现，有需要的自行打开即可
            use_cuda_kernel=False,
            device=self.get_device(),
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
        if ref_wav is None:
            # NOTE: 必须要有 reference audio
            raise RuntimeError("Reference audio not found.")
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
