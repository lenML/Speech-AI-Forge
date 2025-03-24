from dataclasses import dataclass, field
from typing import Literal, Optional

from modules.core.handler.datacls.audio_model import AdjustConfig
from modules.core.handler.datacls.enhancer_model import EnhancerConfig
from modules.core.handler.datacls.tn_model import TNConfig
from modules.core.handler.datacls.tts_model import InferConfig, TTSConfig
from modules.core.handler.datacls.vc_model import VCConfig
from modules.core.spk.TTSSpeaker import TTSSpeaker


@dataclass(repr=False, eq=False)
class TTSSegment:
    _type: Literal["break", "audio"]

    text: str = ""

    temperature: float = 0.3
    top_p: float = 0.9
    top_k: int = 20
    infer_seed: int = 42
    prompt: str = ""
    prompt1: str = ""
    prompt2: str = ""
    prefix: str = ""

    emotion: str = ""

    spk: TTSSpeaker = None

    ## (after process)
    # speed control
    duration_ms: Optional[int] = None
    speed_rate: Optional[float] = None

    ## NOTE: 其实还有个有这几个参数，但是应该不常用，暂时只支持单个 segment 控制速度，不支持其他调整
    # pitch: float = 0
    # volume_gain_db: float = 0


@dataclass(repr=False, eq=False)
class TTSPipelineContext:
    # text/ssml/texts 都是输入文本
    # 1. 只会使用其中一个，且必须有一个值
    # 2. text/ssml 均会启动自动 chunker 分割
    # 3. texts 不使用分割策略，传入什么文本就让模型合成什么文本
    text: Optional[str] = None
    texts: Optional[list[str]] = None
    ssml: Optional[str] = None

    spk: Optional[TTSSpeaker] = None
    tts_config: TTSConfig = field(default_factory=TTSConfig)
    infer_config: InferConfig = field(default_factory=InferConfig)
    adjust_config: AdjustConfig = field(default_factory=AdjustConfig)
    enhancer_config: EnhancerConfig = field(default_factory=EnhancerConfig)
    vc_config: VCConfig = field(default_factory=VCConfig)

    tn_config: TNConfig = field(default_factory=TNConfig)

    # TODO 这里写类型会循环引用...
    modules: list = field(default_factory=list)

    # 当调用 interrupt 时，此变量会被设置为 True
    stop: bool = False
