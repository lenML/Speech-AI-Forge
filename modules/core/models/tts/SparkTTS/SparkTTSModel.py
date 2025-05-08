import io
import logging
from pathlib import Path
from typing import Literal, Optional, Union
from modules.core.models.TTSModel import TTSModel
from modules.core.models.tts.SparkTTS.SparkTTS import SparkTTS
from modules.core.pipeline.dcls import TTSSegment
from modules.devices import devices
import soundfile as sf
import tempfile

from modules.utils.SeedContext import SeedContext

logger = logging.getLogger(__name__)


class SparkTTSModel(TTSModel):

    def __init__(self):
        model_id = "spark-tts"
        super().__init__(model_id)

        self.model: SparkTTS = None

        self.model_path = Path("./models/Spark-TTS-0.5B")
        self.device = devices.get_device_for(model_id)
        self.dtype = devices.dtype

    def check_files(self) -> None:
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

    def is_downloaded(self) -> bool:
        return self.model_path.exists()

    def load(self):
        if self.model is None:
            # TODO: 配置 dtype
            self.model = SparkTTS(model_dir=str(self.model_path), device=self.device)
        return self.model

    @devices.after_gc()
    def unload(self):
        if self.model is None:
            return
        del self.model.model
        del self.model.audio_tokenizer
        del self.model.tokenizer
        self.model = None
        logger.info("Unloaded SparkTTS model")

    def get_sample_rate(self):
        # 来自 models/Spark-TTS-0.5B/config.yaml
        return 16000

    def get_ref_bytesio(self, seg: TTSSegment) -> tuple[io.BytesIO, str]:
        ref_wav, ref_txt = self.get_ref_wav(seg)
        if ref_wav is None:
            raise RuntimeError("Reference audio not found.")
        ref_wav_bytesio = io.BytesIO()
        sf.write(ref_wav_bytesio, ref_wav, self.get_sample_rate(), format="WAV")
        ref_wav_bytesio.seek(0)
        return ref_wav_bytesio, ref_txt

    def get_ref_gender(self, seg: TTSSegment) -> Optional[Literal["male", "female"]]:
        # 从 speaker 中猜测是男还是女，猜不出来就返回 None
        spk = seg.spk
        if spk is None:
            return None
        gender = spk.gender
        if not isinstance(gender, str):
            return None
        if gender.lower() == "male":
            return "male"
        if gender.lower() == "female":
            return "female"
        if "男" in gender.lower():
            return "male"
        if "女" in gender.lower():
            return "female"
        return None

    def generate(self, segment, context):
        cached = self.get_cache(segments=[segment], context=context)
        if cached is not None:
            return cached[0]

        self.load()

        sr = self.get_sample_rate()

        seg0 = segment
        top_p = seg0.top_p
        top_k = seg0.top_k
        temperature = seg0.temperature
        text = seg0.text
        seed = seg0.infer_seed

        ref_wav, ref_txt = self.get_ref_wav(seg0)
        if ref_wav is None:
            raise RuntimeError("Reference audio not found.")

        gender = self.get_ref_gender(seg0)

        # 将 ref_wav 保存为临时文件
        # NOTE: 感觉不太好，可能有兼容问题，最好是可以直接传 bytesio 进去，但是得改 spark tts 库...暂时这样
        with tempfile.NamedTemporaryFile(suffix=".wav") as tmpfile:
            sf.write(tmpfile.name, ref_wav, sr, format="WAV")
            ref_wav_path = Path(tmpfile.name)

            with SeedContext(seed):
                wav = self.model.inference(
                    text=text,
                    prompt_speech_path=str(ref_wav_path),
                    prompt_text=ref_txt,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    # NOTE: 似乎和我想象的不太一样...设置gender等于是使用模型内部的音色
                    # gender=gender,
                    # NOTE: 这两个参数也和模型内部音色有关
                    # TODO: 也许，我们可以想办法提供 api 支持，或者 speakr 支持指定调用参数，以支持使用内置音色
                    # pitch="moderate",
                    # speed="moderate",
                )
                if not context.stop:
                    self.set_cache(
                        segments=[segment], context=context, value=[(sr, wav)]
                    )
                return (sr, wav)

    def generate_batch(self, segments, context):
        return [self.generate(segment, context) for segment in segments]

    def generate_batch_stream(self, segments, context):
        results = []
        for segment in segments:
            results.append(self.generate_batch([segment], context))
        yield results
