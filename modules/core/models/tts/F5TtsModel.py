import io
import logging
import threading
from pathlib import Path
from typing import Generator, Optional

import soundfile as sf

from modules.core.models.tts.F5.F5Annotation import F5Annotation
from modules.core.models.tts.F5.F5ttsApi import F5TTS
from modules.core.models.TTSModel import TTSModel
from modules.core.pipeline.dcls import TTSPipelineContext, TTSSegment
from modules.core.pipeline.processor import NP_AUDIO
from modules.devices import devices
from modules.utils.SeedContext import SeedContext

logger = logging.getLogger(__name__)

class F5TtsModel(TTSModel):
    load_lock = threading.Lock()

    def __init__(self) -> None:
        super().__init__("f5-tts")

        v0_6_model_path = Path("./models/F5-TTS/F5TTS_Base/model_1200000.safetensors")
        v1_120_model_path = Path(
            "./models/F5-TTS/F5TTS_v1_Base/model_1200000.safetensors"
        )
        # NOTE: F5TTS 在 2025三月份发布的新 v1_base 模型
        v1_125_model_path = Path(
            "./models/F5-TTS/F5TTS_v1_Base/model_1250000.safetensors"
        )
        v1_model_path = (
            v1_125_model_path if v1_125_model_path.exists() else v1_120_model_path
        )

        self.model_path = v1_model_path
        self.model_version = "F5TTS_v1_Base"
        self.vocos_path = Path("./models/vocos-mel-24khz")

        if not v1_model_path.exists() and v0_6_model_path.exists():
            logger.warning(
                "F5-TTS v1 model not found, using v0.6.2 model instead. Please update your model."
            )
            self.model_path = v0_6_model_path
            self.model_version = "F5TTS_Base"

        self.model: Optional[F5TTS] = None

        self.annotation = F5Annotation()

    def is_downloaded(self):
        return self.model_path.exists() and self.vocos_path.exists()

    def is_loaded(self):
        return self.model is not None

    def check_files(self) -> None:
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        if not self.vocos_path.exists():
            raise FileNotFoundError(f"Vocos model file not found: {self.vocos_path}")

    def is_downloaded(self) -> bool:
        return self.model_path.exists() and self.vocos_path.exists()

    def _alert_ckpt_files(self):
        """
        提醒下载模型
        """
        if not self.model_path.exists():
            # NOTE: 不存在，提示使用脚本下载 `python -m scripts.downloader.f5_tts_v1 --source huggingface`
            if self.model_version == "F5TTS_Base":
                raise ValueError(
                    f"F5TTS model {self.model_version} not found, please download it manually using `python -m scripts.downloader.f5_tts --source huggingface`"
                )
            else:
                raise ValueError(
                    f"F5TTS model {self.model_version} not found, please download it manually using `python -m scripts.downloader.f5_tts_v1 --source huggingface`"
                )

    def load(self) -> F5TTS:
        self._alert_ckpt_files()
        self.check_files()

        with self.load_lock:
            if self.model is None:
                self.model = F5TTS(
                    ckpt_file=self.model_path,
                    model=self.model_version,
                    vocoder_local_path=str(self.vocos_path),
                    device=self.get_device(),
                    dtype=self.get_dtype(),
                    # TODO: 下面这两个也许可以配置一下？
                    ode_method="euler",
                    use_ema=True,
                )
        return self.model

    @devices.after_gc()
    def unload(self) -> None:
        if self.model is None:
            return
        del self.model
        self.model = None
        logger.info("F5-TTS model unloaded.")

    def get_sample_rate(self) -> int:
        # 来自 modules/repos_static/F5TTS/f5_tts/configs/F5TTS_v1_Base.yaml
        return 24000

    def get_ref_bytesio(self, seg: TTSSegment) -> tuple[io.BytesIO, str]:
        ref_wav, ref_txt = self.get_ref_wav(seg)
        if ref_wav is None:
            raise RuntimeError("Reference audio not found.")
        ref_wav_bytesio = io.BytesIO()
        sf.write(ref_wav_bytesio, ref_wav, self.get_sample_rate(), format="WAV")
        ref_wav_bytesio.seek(0)
        return ref_wav_bytesio, ref_txt

    def generate_batch(
        self, segments: list[TTSSegment], context: TTSPipelineContext
    ) -> list[NP_AUDIO]:
        cached = self.get_cache(segments=segments, context=context)
        if cached is not None:
            return cached

        self.load()

        sr = self.get_sample_rate()

        texts = [segment.text for segment in segments]

        seg0 = segments[0]
        # NOTE: 虽然用不到这些参数...但是还是列出来先
        # top_P = seg0.top_p
        # top_K = seg0.top_k
        # temperature = seg0.temperature
        # repetition_penalty = seg0.repetition_penalty
        # max_new_token = seg0.max_new_token
        prompt = seg0.prompt
        prompt1 = seg0.prompt1
        prompt2 = seg0.prompt2
        prefix = seg0.prefix
        # use_decoder = seg0.use_decoder
        seed = seg0.infer_seed
        chunk_size = context.infer_config.stream_chunk_size

        ref_bytesio, ref_txt = self.get_ref_bytesio(seg0)

        with SeedContext(seed=seed):
            generated_waves = self.model.infer_batch(
                ref_file=ref_bytesio,
                ref_text=ref_txt,
                gen_text_batches=texts,
                # TODO: 增加 diffusion 模型使用的参数，目前下面这些是写死的，需要增加 speaker 配置可以支持其他参数
                target_rms=0.1,
                cross_fade_duration=0.15,
                sway_sampling_coef=-1,
                cfg_strength=2,
                nfe_step=32,
                speed=1,
            )

        results = [(sr, wav) for wav, sr, _ in generated_waves]
        if not context.stop:
            self.set_cache(segments=segments, context=context, value=results)
        return results

    def generate_batch_stream(
        self, segments: list[TTSSegment], context: TTSPipelineContext
    ) -> Generator[list[NP_AUDIO], None, None]:
        # NOTE: F5ttsApi 里面有一个 infer_batch_stream ，但是和我们想象的 batch stream 不太一样，是通过多线程实现的...并且逻辑也和我们的 batch 不太一样，所以，约等于暂时无法 batch

        results = []
        for segment in segments:
            results.append(self.generate_batch([segment], context))
        yield results
