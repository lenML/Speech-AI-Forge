import threading
from typing import Generator, Union
from modules import config
from modules.core.models.TTSModel import TTSModel
from modules.core.models.tts.FishSpeechInfer import FishSpeechInfer
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

import logging

from modules.utils.SeedContext import SeedContext

FISH_SPEECH_LLAMA = Union[NaiveTransformer, DualARTransformer]

logger = logging.getLogger(__name__)


class FishSpeechModel(TTSModel):
    lock = threading.Lock()

    def __init__(self) -> None:
        super().__init__("fish_speech")

        self.model: FISH_SPEECH_LLAMA = None
        self.token_decoder: callable = None
        self.vqgan: FireflyArchitecture = None

        self.device = devices.get_device_for("fish_speech")
        self.dtype = devices.dtype

        self.encoded_prefix = []

    def reset(self):
        self.encoded_prefix = []

    def load(
        self, context: TTSPipelineContext
    ) -> tuple[FISH_SPEECH_LLAMA, FireflyArchitecture]:
        llama = self.load_llama()
        vqgan = self.load_vqgan()
        return llama, vqgan

    def load_llama(self) -> FISH_SPEECH_LLAMA:
        if self.model:
            return self.model

        with self.lock:
            logger.info(
                f"loading FishSpeech llama on device [{self.device}] with dtype [{self.dtype}]"
            )

            model, token_decoder = load_llama_model(
                checkpoint_path="./models/FishSpeech/",
                device=self.device,
                precision=self.dtype,
                compile=config.runtime_env_vars.compile,
            )

            logger.info("Loaded FishSpeech model")

            self.model = model
            self.token_decoder = token_decoder
            devices.torch_gc()
            return model

    def load_vqgan(self) -> FireflyArchitecture:
        if self.vqgan:
            return self.vqgan

        with self.lock:
            logger.info(
                f"loading FishSpeech vqgan on device [{self.device}] with dtype [{self.dtype}]"
            )
            config_name = "firefly_gan_vq"
            checkpoint_path = (
                "./models/FishSpeech/firefly-gan-vq-fsq-4x1024-42hz-generator.pth"
            )
            model: FireflyArchitecture = load_vqgan_model(
                config_name=config_name,
                checkpoint_path=checkpoint_path,
                device=self.device,
            )
            model = model.to(device=self.device, dtype=self.dtype)

            self.vqgan = model
            return model

    def unload(self, context: TTSPipelineContext) -> None:
        with self.lock:
            del self.model
            del self.token_decoder
            del self.vqgan
            self.model = None
            self.token_decoder = None
            self.vqgan = None
            devices.torch_gc()

    def generate_batch(
        self, segments: list[TTSSegment], context: TTSPipelineContext
    ) -> list[NP_AUDIO]:
        generator = self.generate_batch_stream(segments, context)
        return next(generator)

    # NOTE: 不支持batch生成 所以基本上是同步的
    def generate_batch_stream(
        self, segments: list[TTSSegment], context: TTSPipelineContext
    ) -> Generator[list[NP_AUDIO], None, None]:
        self.load(context=context)
        infer = FishSpeechInfer(
            llama=self.model,
            vqgan=self.vqgan,
            token_decoder=self.token_decoder,
        )
        seg0 = segments[0]
        infer_seed = seg0.infer_seed
        top_p = seg0.top_p
        emotion = seg0.emotion
        temperature = seg0.temperature
        # repetition_penalty = seg0.repetition_penalty

        sr = infer.sample_rate
        ret = []
        for segment in segments:
            with SeedContext(seed=infer_seed):
                decoded, generated = infer.infer(
                    text=segment.text,
                    temperature=temperature,
                    top_p=top_p,
                    emotion=emotion,
                    encoded_prefix=self.encoded_prefix[-2:],
                    max_new_tokens=2048,
                )

            self.encoded_prefix.append(decoded)
            ret.append((sr, generated))
        yield ret


if __name__ == "__main__":
    import numpy as np
    import soundfile as sf

    tts_model = FishSpeechModel()
    tts_model.load(context=None)

    seeds = [
        111,
        222,
        333,
        444,
        555,
        666,
        777,
        888,
        999,
    ]

    (
        t1,
        t2,
    ) = """
大家好，我是 Fish Audio 开发的开源文本转语音模型。经过十五万小时的数据训练，
我已经能够熟练掌握中文、日语和英语，我的语言处理能力接近人类水平，声音表现形式丰富多变。
    """.strip().split(
        "\n"
    )

    for seed in seeds:

        def create_seg(text: str):
            return TTSSegment(_type="text", text=text, infer_seed=seed)

        audio = np.empty(0)
        for sr, data in tts_model.generate_batch(
            segments=[create_seg(t1), create_seg(t2)],
            context=None,
        ):
            audio = np.concatenate((audio, data), axis=0)

        # audio_data = read_np_to_wav(audio_data=audio)
        sf.write(f"test_fish_speech_{seed}.wav", audio, sr)
