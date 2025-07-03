if __name__ == "__main__":
    from modules.repos_static.sys_paths import setup_repos_paths

    setup_repos_paths()

from pathlib import Path
from typing import Literal

import numpy as np
import torch
from modules.core.models.AudioReshaper import AudioReshaper
from modules.core.models.TTSModel import TTSModel
from modules.core.models.tts.GptSoVits.GptSoVitsTTS import GptSoVitsTTS
from modules.core.models.tts.GptSoVits.GptSoVitsTTSConfig import GptSoVitsTTSConfig
from modules.core.pipeline.dcls import TTSPipelineContext, TTSSegment
from modules.devices import devices
from modules.utils.SeedContext import SeedContext
from modules.utils.detect_lang import guess_lang

import tempfile
import soundfile as sf

class GptSoVitsModel(TTSModel):

    def __init__(self, version: Literal["v1", "v2", "v3", "v4"] = "v4"):
        model_id = f"gpt-so-vits-{version}"
        super().__init__(model_id)

        self.version = version

        self.model: GptSoVitsTTS = None

    def is_downloaded(self):
        return Path(f"models/gpt_sovits_{self.version}").exists()

    def get_device(self):
        return devices.get_device_for("gpt-so-vits")

    def get_sample_rate(self):
        # return self.config.sampling_rate
        # 来自 modules/repos_static/GPT_SoVITS/GPT_SoVITS/TTS_infer_pack/TTS.py
        return 32000

    def load(self):
        if self.model is None:
            configs = GptSoVitsTTSConfig(
                {
                    "custom": {
                        "device": self.get_device(),
                        "is_half": self.get_dtype() == torch.float16,
                    },
                    "version": self.version,
                }
            )
            # print("gpt-sovits configs:")
            # print(configs)
            self.model = GptSoVitsTTS(configs=configs)
        return self.model

    @devices.after_gc()
    def unload(self):
        if self.model is None:
            return
        # move to cpu
        self.model.set_device("cpu", False)
        del self.model
        self.model = None

    def generate(self, segment, context):
        cached = self.get_cache(segments=[segment], context=context)
        if cached is not None:
            return cached

        model = self.load()

        seg0 = segment
        ref_wav, ref_txt = self.get_ref_wav(seg0)

        if ref_wav is None:
            # 必须参考音频
            raise ValueError("Reference audio is required")

        top_p = seg0.top_p
        top_k = seg0.top_k
        temperature = seg0.temperature
        # repetition_penalty = seg0.repetition_penalty
        # max_new_token = seg0.max_new_token
        prompt = seg0.prompt
        prompt1 = seg0.prompt1
        prompt2 = seg0.prompt2
        prefix = seg0.prefix
        # use_decoder = seg0.use_decoder
        seed = seg0.infer_seed
        chunk_size = context.infer_config.stream_chunk_size

        sr = self.get_sample_rate()

        with tempfile.NamedTemporaryFile(suffix=".wav") as tmpfile:
            sf.write(tmpfile.name, ref_wav, sr, format="WAV")
            ref_wav_path = Path(tmpfile.name)

            with SeedContext(seed):
                result: tuple[int, np.ndarray] = next(
                    model.run(
                        {
                            "text": seg0.text,
                            # NOTE: 这里似乎可以填 "auto" ？
                            "text_lang": guess_lang(seg0.text),
                            # ref:
                            "ref_audio_path": str(ref_wav_path),
                            "prompt_text": ref_txt,
                            "prompt_lang": guess_lang(ref_txt),
                            # params:
                            "temperature": temperature,
                            "top_p": top_p,
                            "top_k": top_k,
                            "seed": seed,
                            # 不需要流式，因为内部实现其实就是分句子流式
                            "return_fragment": False,
                            "split_bucket": False,
                        }
                    )
                )
                # 这里输出的 data 是 int16 ，我们转为 float32 以适配整个系统
                sr, data = AudioReshaper.ensure_float32(result)

                if not context.stop:
                    self.set_cache(
                        segments=[segment], context=context, value=(sr, data)
                    )

                return sr, data

    def generate_batch(self, segments, context):
        # 官方代码里没有 input texts batch 的实现，所以，这里直接调 generate
        # NOTE: 其实应该是可以 batch 的，但是需要我们自己实现 run_batch ...
        return [self.generate(segment, context) for segment in segments]

    def generate_stream(self, segment, context):
        # 不支持 token 级别 stream
        yield self.generate(segment, context)

    def generate_batch_stream(self, segments, context):
        # 不支持 token 级别 stream
        for segment in segments:
            yield self.generate(segment, context)


if __name__ == "__main__":
    """
    测试
    """
    import soundfile as sf
    from modules.core.spk.SpkMgr import spk_mgr

    def create_seg(text: str):
        spk = spk_mgr.get_speaker("mona")
        return TTSSegment(_type="text", spk=spk, text=text, infer_seed=42)

    ctx = TTSPipelineContext()

    model_v4 = GptSoVitsModel("v4")
    test_input = create_seg("云想衣裳花想容，春风拂槛露华浓")
    test_output_filepath = "./gpt_sovits_v4_test.wav"

    sr, data = model_v4.generate(test_input, ctx)
    sf.write(test_output_filepath, data, sr, format="WAV")
