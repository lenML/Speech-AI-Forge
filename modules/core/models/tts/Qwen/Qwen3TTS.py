import logging
from typing import Literal
import torch
from modules.core.models.TTSModel import TTSModel
from modules.core.spk.SpkMgr import spk_mgr
from modules.core.spk.TTSSpeaker import TTSSpeaker
from modules.devices import devices
from modules.downloader.AutoModelDownloader import AutoModelDownloader


from modules.repos_static.Qwen3_TTS.qwen_tts import (
    Qwen3TTSModel as Qwen3TTS,
    Qwen3TTSTokenizer,
)
from modules.utils.SeedContext import SeedContext
from transformers.utils import is_flash_attn_2_available

logger = logging.getLogger(__name__)

model_versions = [
    "1.7B-VoiceDesign",
    "1.7B-CustomVoice",
    "1.7B-Base",
    "0.6B-CustomVoice",
    "0.6B-Base",
]

# For `Qwen3-TTS-12Hz-1.7B/0.6B-CustomVoice` models, the supported speaker list and speaker descriptions are provided below. We recommend using each speaker’s native language for the best quality. Of course, each speaker can speak any language supported by the model.

# | Speaker | Voice Description  |  Native language |
# | --- | --- | --- |
# | Vivian | Bright, slightly edgy young female voice. | Chinese |
# | Serena | Warm, gentle young female voice. | Chinese |
# | Uncle_Fu | Seasoned male voice with a low, mellow timbre. | Chinese |
# | Dylan | Youthful Beijing male voice with a clear, natural timbre. | Chinese (Beijing Dialect) |
# | Eric | Lively Chengdu male voice with a slightly husky brightness. | Chinese (Sichuan Dialect) |
# | Ryan | Dynamic male voice with strong rhythmic drive. | English |
# | Aiden | Sunny American male voice with a clear midrange. | English |
# | Ono_Anna | Playful Japanese female voice with a light, nimble timbre. | Japanese |
# | Sohee | Warm Korean female voice with rich emotion. | Korean |
custom_voices = [
    "Vivian",
    "Serena",
    "Uncle_Fu",
    "Dylan",
    "Eric",
    "Ryan",
    "Aiden",
    "Ono_Anna",
    "Sohee",
]
custom_spks = [
    TTSSpeaker.virtual(name=name, models=["qwen3-tts-*cv"]) for name in custom_voices
]
# 添加到 mgr
for spk in custom_spks:
    spk_mgr.ext_items.append(spk)

class Qwen3TTSModel(TTSModel):

    def __init__(self, model_version="1.7B-Base", tokenizer_sr="12Hz"):
        model_name = f"Qwen3-TTS-{tokenizer_sr}-{model_version}"

        super().__init__("qwen3-tts", model_name=model_name)

        self.model_version = model_version
        self.tokenizer_sr = tokenizer_sr
        if model_version not in model_versions:
            raise ValueError(
                f"Invalid model version: {model_version}. Available versions: {model_versions}"
            )

        self.model_name = model_name

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
            # NOTE: 实测用不了，会导致数值溢出 切换为 bf16
            if torch.cuda.is_bf16_supported():
                logger.warning("qwen3-tts: bf16 is used instead of fp16")
                dtype = torch.bfloat16
            else:
                logger.warning("qwen3-tts: fp16 无法在此模型上使用，自动切换为 fp32")
                dtype = torch.float32
        return dtype

    def load(self):
        if self.model is not None:
            return self.model, self.tokenizer
        # NOTE download 的语义其实写在 generate 里面好一点，但是...写在这里应该也没什么问题
        model_path = self.download()

        device = self.get_device()
        dtype = self.get_dtype()

        attn_impl = "flash_attention_2" if is_flash_attn_2_available() else "sdpa"

        self.model = Qwen3TTS.from_pretrained(
            str(model_path.absolute()),
            device_map=device,
            dtype=dtype,
            attn_implementation=attn_impl,
        )
        self.tokenizer = Qwen3TTSTokenizer.from_pretrained(
            str(model_path.absolute() / "speech_tokenizer"),
            device_map=device,
        )

        return self.model, self.tokenizer

    def is_loaded(self):
        return self.model is not None and self.tokenizer is not None

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

        model_type: Literal["base", "custom_voice", "voice_design"] = (
            model.model.tts_model_type
        )

        texts = [seg.text for seg in segments]
        instructs = [seg.prompt1 for seg in segments]

        with SeedContext(seed=seed):
            if model_type == "base":
                ref_wav, ref_txt = self.get_ref_wav(seg0)
                if ref_wav is None:
                    raise RuntimeError(
                        "Reference audio not found. Base model requires a reference audio."
                    )
                voice_clone_prompt = model.create_voice_clone_prompt(
                    ref_audio=(ref_wav, sr),
                    ref_text=ref_txt,
                )
                wavs, sr = model.generate_voice_clone(
                    text=texts,
                    top_k=top_k,
                    top_p=top_p,
                    temperature=temperature,
                    # FIXME: 似乎如果能提供 lang 会有助于推理，但是也支持 auto
                    # language=None,
                    voice_clone_prompt=voice_clone_prompt,
                )
            elif model_type == "custom_voice":
                speakers = [seg.spk.name for seg in segments]
                # 如果不是 custom_voices 里的就报错
                if not all(speaker in custom_voices for speaker in speakers):
                    raise ValueError(
                        f"Speaker(s) {', '.join(set(speakers) - set(custom_voices))} not found in custom_voices."
                    )
                wavs, sr = model.generate_custom_voice(
                    text=texts,
                    instruct=instructs,
                    speaker=speakers,
                    top_k=top_k,
                    top_p=top_p,
                    temperature=temperature,
                )
            elif model_type == "voice_design":
                wavs, sr = model.generate_voice_design(
                    text=texts,
                    instruct=instructs,
                    top_k=top_k,
                    top_p=top_p,
                    temperature=temperature,
                )
            else:
                raise ValueError(f"Unsupported model type: {model_type}")

        return [(sr, wav) for wav in wavs]

    def generate_batch_stream(self, segments, context):
        # NOTE: 现在的代码不支持，但是模型是支持的
        ret = self.generate_batch(segments, context)
        yield ret
