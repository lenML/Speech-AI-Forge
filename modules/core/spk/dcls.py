import time
from dataclasses import dataclass, field
from typing import List, Optional, Union

import torch


@dataclass(repr=False, eq=False)
class DcSpkMeta:
    name: str = ""
    desc: str = ""
    gender: str = ""

    author: str = ""
    version: str = ""

    # 时间戳 毫秒
    created_date: int = field(default_factory=lambda: int(time.time() * 1000))
    # 说话人头像 url 或者 base64
    avatar: str = ""
    # tag 用于 hub 中筛选
    tags: List[str] = field(default_factory=list)


# embedding
@dataclass(repr=False, eq=False)
class DcSpkVoiceToken:
    model_id: str
    model_hash: str = ""

    # 可以填入特殊 token 或者 tensor
    # 一般都是 tensor
    # egs: ["<EMOTION_START>","xxx","<EMOTION_END>", [...<torch.Tensor>...]]
    tokens: list[Union[str, list]] = field(default_factory=list)

    # cosyvoice 特有的属性
    embedding: Optional[list[torch.Tensor]] = None
    feat: Optional[list[torch.Tensor]] = None


@dataclass(repr=False, eq=False)
class DcSpkSample:
    wav: bytes
    text: str
    wav_sr: int


@dataclass(repr=False, eq=False)
class DcSpkReference:
    text: Optional[str] = None

    wav: Optional[bytes] = None
    wav_sr: Optional[int] = None

    # 标注情绪
    emotion: Optional[str] = None

    # 标记语言
    lang: Optional[str] = None


@dataclass(repr=False, eq=False)
class DcSpkTrainInfo:
    steps: int
    epochs: int
    dataset: str
    samples: int
    batch_size: int
    optimizer: str
    lr: float
    loss: str

    extra: Optional[dict]


# 这个说话人的推荐配置
@dataclass(repr=False, eq=False)
class DcSpkInferConfig:
    tempature: float
    top_k: int
    top_p: float
    max_tokens: int
    repetition_penalty: float

    # 应该没几个模型支持这个...
    emotion: str


@dataclass(repr=False, eq=False)
class DcSpk:
    id: str
    # 这里是 speaker file 版本
    version: str = "0.1"

    meta: DcSpkMeta = field(default_factory=DcSpkMeta)
    token: List[DcSpkVoiceToken] = field(default_factory=list)
    samples: List[DcSpkSample] = field(default_factory=list)
    refs: List[DcSpkReference] = field(default_factory=list)
    train_info: Optional[DcSpkTrainInfo] = None

    recommend_config: Optional[DcSpkInferConfig] = None
