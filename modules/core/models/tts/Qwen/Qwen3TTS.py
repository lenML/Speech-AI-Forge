import logging
import torch
from modules.core.models.TTSModel import TTSModel
from modules.devices import devices
from modules.downloader.AutoModelDownloader import AutoModelDownloader


from modules.repos_static.Qwen3_TTS.qwen_tts import (
    Qwen3TTSModel as Qwen3TTS,
    Qwen3TTSTokenizer,
)

logger = logging.getLogger(__name__)

model_versions = [
    "1.7B-VoiceDesign",
    "1.7B-CustomVoice",
    "1.7B-Base",
    "0.6B-CustomVoice",
    "0.6B-Base",
]


class Qwen3TTSModel(TTSModel):
    def __init__(self, model_version="1.7B-CustomVoice", tokenizer_sr="12Hz"):
        super().__init__("qwen3-tts")
        self.model_version = model_version
        self.tokenizer_sr = tokenizer_sr
        if model_version not in model_versions:
            raise ValueError(
                f"Invalid model version: {model_version}. Available versions: {model_versions}"
            )

        self.model_name = f"Qwen3-TTS-{tokenizer_sr}-{model_version}"

        self.model: "None | Qwen3TTS" = None
        self.tokenizer: "None | Qwen3TTSTokenizer" = None

    def get_sample_rate(self):
        # 来自 models/Qwen3-TTS-12Hz-0.6B-CustomVoice/speech_tokenizer/config.json
        # input_sample_rate=24000
        # output_sample_rate=24000
        return 24000

    def get_dtype(self):
        dtype = super().get_dtype()
        if dtype == torch.float16:
            # NOTE: 实测用不了，会导致数值溢出
            logger.warning(
                "检测到 dtype 为 float16，但 Qwen3TTS 对 float16 支持很差，已强制切换为 float32。"
                "如需 f16 减少显存占用，请使用 --bf16 开启 bfloat16 模式以获得更好兼容性。"
            )
            return torch.float32
        return dtype

    def load(self):
        if self.model is not None:
            return self.model, self.tokenizer

        downloader = AutoModelDownloader()
        # TODO: 获取当前执行环境 type
        request_type = "api"

        model_path = downloader.download(
            model_name=self.model_name, request_type=request_type
        )

        device = self.get_device()
        dtype = self.get_dtype()

        self.model = Qwen3TTS.from_pretrained(
            str(model_path.absolute()),
            device_map=device,
            # dtype=dtype,
            # TODO: 支持 flash atten
            # attn_implementation="flash_attention_2",
        )
        self.tokenizer = Qwen3TTSTokenizer.from_pretrained(
            str(model_path.absolute() / "speech_tokenizer"),
            device_map=device,
        )

        return self.model, self.tokenizer

    @devices.after_gc()
    def unload(self):
        del self.model
        del self.tokenizer
        self.model = None
        self.tokenizer = None

    def generate_batch(self, segments, context):
        model, tokenizer = self.load()

        sr = self.get_sample_rate()

        seg0 = segments[0]
        top_p = seg0.top_p
        top_k = seg0.top_k
        temperature = seg0.temperature
        seed = seg0.infer_seed

        ref_wav, ref_txt = self.get_ref_wav(seg0)
        if ref_wav is None:
            raise RuntimeError("Reference audio not found.")

        # TODO: 不同type用不同的方法
        model_type = model.model.tts_model_type

        voice_clone_prompt = model.create_voice_clone_prompt(
            ref_audio=(ref_wav, sr),
            ref_text=ref_txt,
        )
        wavs, sr = model.generate_voice_clone(
            text=[seg.text for seg in segments],
            # FIXME: 似乎如果能提供 lang 会有助于推理，但是也支持 auto
            # language=None,
            voice_clone_prompt=voice_clone_prompt,
        )

        return [(sr, wav) for wav in wavs]

    def generate_batch_stream(self, segments, context):
        # NOTE: 现在的代码不支持，但是模型是支持的
        ret = self.generate_batch(segments, context)
        yield ret
