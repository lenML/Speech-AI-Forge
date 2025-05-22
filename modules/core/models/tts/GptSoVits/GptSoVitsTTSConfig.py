from copy import deepcopy
import os
from typing import Union

import torch
from modules.repos_static.GPT_SoVITS.GPT_SoVITS.TTS_infer_pack.TTS import (
    TTS_Config,
)


# 主要修改:
# 1. 改为 forge 路径
# 2. 永远不会读取和保存配置
class GptSoVitsTTSConfig(TTS_Config):
    default_configs = {
        "v1": {
            "device": "cpu",
            "is_half": False,
            "version": "v1",
            "t2s_weights_path": "models/gpt_sovits_v1/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt",
            "vits_weights_path": "models/gpt_sovits_v1/s2G488k.pth",
            "cnhuhbert_base_path": "models/gpt_sovits_v1/chinese-hubert-base",
            "bert_base_path": "models/gpt_sovits_v1/chinese-roberta-wwm-ext-large",
        },
        "v2": {
            "device": "cpu",
            "is_half": False,
            "version": "v2",
            "t2s_weights_path": "models/gpt_sovits_v2/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt",
            "vits_weights_path": "models/gpt_sovits_v2/gsv-v2final-pretrained/s2G2333k.pth",
            "cnhuhbert_base_path": "models/gpt_sovits_v2/chinese-hubert-base",
            "bert_base_path": "models/gpt_sovits_v2/chinese-roberta-wwm-ext-large",
        },
        "v3": {
            "device": "cpu",
            "is_half": False,
            "version": "v3",
            "t2s_weights_path": "models/gpt_sovits_v3/s1v3.ckpt",
            "vits_weights_path": "models/gpt_sovits_v3/s2Gv3.pth",
            "cnhuhbert_base_path": "models/gpt_sovits_v3/chinese-hubert-base",
            "bert_base_path": "models/gpt_sovits_v3/chinese-roberta-wwm-ext-large",
        },
        "v4": {
            "device": "cpu",
            "is_half": False,
            "version": "v4",
            "t2s_weights_path": "models/gpt_sovits_v4/s1v3.ckpt",
            "vits_weights_path": "models/gpt_sovits_v4/gsv-v4-pretrained/s2Gv4.pth",
            "cnhuhbert_base_path": "models/gpt_sovits_v4/chinese-hubert-base",
            "bert_base_path": "models/gpt_sovits_v4/chinese-roberta-wwm-ext-large",
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
