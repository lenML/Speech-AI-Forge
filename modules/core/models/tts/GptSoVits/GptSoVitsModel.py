if __name__ == "__main__":
    from modules.repos_static.sys_paths import setup_repos_paths

    setup_repos_paths()

from copy import deepcopy
import os
from pathlib import Path
from typing import Literal, Union

import numpy as np
import torch
from modules.core.models.TTSModel import TTSModel
from modules.core.pipeline.dcls import TTSSegment
from modules.devices import devices
from modules.repos_static.GPT_SoVITS.GPT_SoVITS.TTS_infer_pack.TTS import (
    TTS,
    TTS_Config,
)
from modules.repos_static.GPT_SoVITS.GPT_SoVITS.BigVGAN.bigvgan import BigVGAN
from modules.repos_static.GPT_SoVITS.GPT_SoVITS.module.models import Generator
from modules.utils.SeedContext import SeedContext
from modules.utils.detect_lang import guess_lang

import tempfile
import soundfile as sf


# 主要修改:
# 1. 改为 forge 路径
# 2. 永远不会读取和保存配置
class GptSoVitsTTSConfig(TTS_Config):
    default_configs = {
        "v1": {
            "device": "cpu",
            "is_half": False,
            "version": "v1",
            "t2s_weights_path": "models/GPT_SoVITS/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt",
            "vits_weights_path": "models/GPT_SoVITS/s2G488k.pth",
            "cnhuhbert_base_path": "models/GPT_SoVITS/chinese-hubert-base",
            "bert_base_path": "models/GPT_SoVITS/chinese-roberta-wwm-ext-large",
        },
        "v2": {
            "device": "cpu",
            "is_half": False,
            "version": "v2",
            "t2s_weights_path": "models/GPT_SoVITS/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt",
            "vits_weights_path": "models/GPT_SoVITS/gsv-v2final-pretrained/s2G2333k.pth",
            "cnhuhbert_base_path": "models/GPT_SoVITS/chinese-hubert-base",
            "bert_base_path": "models/GPT_SoVITS/chinese-roberta-wwm-ext-large",
        },
        "v3": {
            "device": "cpu",
            "is_half": False,
            "version": "v3",
            "t2s_weights_path": "models/GPT_SoVITS/s1v3.ckpt",
            "vits_weights_path": "models/GPT_SoVITS/s2Gv3.pth",
            "cnhuhbert_base_path": "models/GPT_SoVITS/chinese-hubert-base",
            "bert_base_path": "models/GPT_SoVITS/chinese-roberta-wwm-ext-large",
        },
        "v4": {
            "device": "cpu",
            "is_half": False,
            "version": "v4",
            "t2s_weights_path": "models/GPT_SoVITS/s1v3.ckpt",
            "vits_weights_path": "models/GPT_SoVITS/gsv-v4-pretrained/s2Gv4.pth",
            "cnhuhbert_base_path": "models/GPT_SoVITS/chinese-hubert-base",
            "bert_base_path": "models/GPT_SoVITS/chinese-roberta-wwm-ext-large",
        },
    }

    def __init__(self, configs: Union[dict, str] = None):
        assert isinstance(configs, dict)
        version = configs.get("version", "v2").lower()
        assert version in ["v1", "v2", "v3", "v4"]
        self.default_configs[version] = configs.get(
            version, self.default_configs[version]
        )
        self.configs: dict = configs.get(
            "custom", deepcopy(self.default_configs[version])
        )

        self.device = self.configs.get("device", torch.device("cpu"))
        if "cuda" in str(self.device) and not torch.cuda.is_available():
            print("Warning: CUDA is not available, set device to CPU.")
            self.device = torch.device("cpu")

        self.is_half = self.configs.get("is_half", False)
        # if str(self.device) == "cpu" and self.is_half:
        #     print(f"Warning: Half precision is not supported on CPU, set is_half to False.")
        #     self.is_half = False

        self.version = version
        self.t2s_weights_path = self.configs.get("t2s_weights_path", None)
        self.vits_weights_path = self.configs.get("vits_weights_path", None)
        self.bert_base_path = self.configs.get("bert_base_path", None)
        self.cnhuhbert_base_path = self.configs.get("cnhuhbert_base_path", None)
        self.languages = (
            self.v1_languages if self.version == "v1" else self.v2_languages
        )

        self.use_vocoder: bool = False

        if (self.t2s_weights_path in [None, ""]) or (
            not os.path.exists(self.t2s_weights_path)
        ):
            self.t2s_weights_path = self.default_configs[version]["t2s_weights_path"]
            print(f"fall back to default t2s_weights_path: {self.t2s_weights_path}")
        if (self.vits_weights_path in [None, ""]) or (
            not os.path.exists(self.vits_weights_path)
        ):
            self.vits_weights_path = self.default_configs[version]["vits_weights_path"]
            print(f"fall back to default vits_weights_path: {self.vits_weights_path}")
        if (self.bert_base_path in [None, ""]) or (
            not os.path.exists(self.bert_base_path)
        ):
            self.bert_base_path = self.default_configs[version]["bert_base_path"]
            print(f"fall back to default bert_base_path: {self.bert_base_path}")
        if (self.cnhuhbert_base_path in [None, ""]) or (
            not os.path.exists(self.cnhuhbert_base_path)
        ):
            self.cnhuhbert_base_path = self.default_configs[version][
                "cnhuhbert_base_path"
            ]
            print(
                f"fall back to default cnhuhbert_base_path: {self.cnhuhbert_base_path}"
            )
        self.update_configs()

        self.max_sec = None
        self.hz: int = 50
        self.semantic_frame_rate: str = "25hz"
        self.segment_size: int = 20480
        self.filter_length: int = 2048
        self.sampling_rate: int = 32000
        self.hop_length: int = 640
        self.win_length: int = 2048
        self.n_speakers: int = 300

    def _load_configs(self, configs_path):
        # 不支持
        raise NotImplementedError()

    def save_configs(self, configs_path=None):
        # 不支持
        # raise NotImplementedError()
        # 因为调用 set_devices 默认会调用save_configs... 所以这里不抛出错误
        pass


# 主要修改: 加载路径
class GptSoVitsTTS(TTS):
    def init_vocoder(self, version: str):
        if version == "v3":
            if (
                self.vocoder is not None
                and self.vocoder.__class__.__name__ == "BigVGAN"
            ):
                return
            if self.vocoder is not None:
                self.vocoder.cpu()
                del self.vocoder
                self.empty_cache()

            self.vocoder = BigVGAN.from_pretrained(
                "models/gpt_sovits_v3/pretrained_models/models--nvidia--bigvgan_v2_24khz_100band_256x",
                use_cuda_kernel=False,
            )  # if True, RuntimeError: Ninja is required to load C++ extensions
            # remove weight norm in the model and set to eval mode
            self.vocoder.remove_weight_norm()

            self.vocoder_configs["sr"] = 24000
            self.vocoder_configs["T_ref"] = 468
            self.vocoder_configs["T_chunk"] = 934
            self.vocoder_configs["upsample_rate"] = 256
            self.vocoder_configs["overlapped_len"] = 12

        elif version == "v4":
            if (
                self.vocoder is not None
                and self.vocoder.__class__.__name__ == "Generator"
            ):
                return
            if self.vocoder is not None:
                self.vocoder.cpu()
                del self.vocoder
                self.empty_cache()

            self.vocoder = Generator(
                initial_channel=100,
                resblock="1",
                resblock_kernel_sizes=[3, 7, 11],
                resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                upsample_rates=[10, 6, 2, 2, 2],
                upsample_initial_channel=512,
                upsample_kernel_sizes=[20, 12, 4, 4, 4],
                gin_channels=0,
                is_bias=True,
            )
            self.vocoder.remove_weight_norm()
            state_dict_g = torch.load(
                "models/gpt_sovits_v3/GPT_SoVITS/pretrained_models/gsv-v4-pretrained/vocoder.pth",
                map_location="cpu",
            )
            print("loading vocoder", self.vocoder.load_state_dict(state_dict_g))

            self.vocoder_configs["sr"] = 48000
            self.vocoder_configs["T_ref"] = 500
            self.vocoder_configs["T_chunk"] = 1000
            self.vocoder_configs["upsample_rate"] = 480
            self.vocoder_configs["overlapped_len"] = 12

        self.vocoder = self.vocoder.eval()
        if self.configs.is_half == True:
            self.vocoder = self.vocoder.half().to(self.configs.device)
        else:
            self.vocoder = self.vocoder.to(self.configs.device)


class GptSoVitsModel(TTSModel):

    def __init__(self, version: Literal["v1", "v2", "v3", "v4"] = "v4"):
        model_id = f"gpt-so-vits-{version}"
        super().__init__(model_id)

        self.model: GptSoVitsTTS = None
        # FIXME: 在这里初始化有可能导致提前加载 pytorch
        self.config: GptSoVitsTTSConfig = GptSoVitsTTSConfig({"version": version})

    def get_sample_rate(self):
        return self.config.sampling_rate

    def load(self):
        if self.model is None:
            self.model = GptSoVitsTTS(self.config)
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
        # TODO 增加 cache
        model = self.load()

        seg0 = segment
        ref_wav, ref_txt = self.get_ref_wav(seg0)
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
                result: tuple[int, np.ndarray] = model.run(
                    {
                        "text": seg0.text,
                        # NOTE: 这里似乎可以填 "auto" ？
                        "text_lang": guess_lang(seg0.text),
                        # ref:
                        "ref_audio_path": ref_wav_path,
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
                sr, data = result
                # maybe process something here?
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

    model_v4 = GptSoVitsModel("v4")
    test_input = "云想衣裳花想容，春风拂槛露华浓"
    test_output_filepath = "./gpt_sovits_v4_test.wav"

    sr, data = model_v4.generate(test_input)
    sf.write(test_output_filepath, data, sr, format="WAV")
